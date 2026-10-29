"""Deterministic, parallel Phase I particle-source ROOT tuple generation.

Workers generate NumPy chunks only.  ROOT is imported in the parent process and
all tree writes happen there, in chunk order, so output does not depend on
worker completion order.
"""

from __future__ import annotations

import argparse
from array import array
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any, Iterator


SCHEMA_VERSION = 1
PARTICLES = {"gamma": (22, 0.0), "e+": (-11, 0.51099895)}
DEFAULT_GEOMETRY = {
    "shape": "cylinder",
    "radius_mm": 30.0,
    "thickness_mm": 60.0,
    "center_mm": [0.0, 0.0, 0.0],
}
DEFAULT_GEOMETRY_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/app/simulation/MACE/PhaseI/SimMACEPhaseI/scripts/SimMACEPhaseI_geom.yaml"
)
BRANCH_CONTRACT = {
    "w": "float scalar",
    "t": "double scalar",
    "x": "float scalar [mm]",
    "y": "float scalar [mm]",
    "z": "float scalar [mm]",
    "pdgID": "vector<int>",
    "E": "vector<float> [MeV, total energy]",
    "px": "vector<float> [MeV/c]",
    "py": "vector<float> [MeV/c]",
    "pz": "vector<float> [MeV/c]",
}


def _simple_yaml_geometry(text: str) -> dict[str, Any]:
    """Read a small, deliberately restricted YAML geometry mapping.

    PyYAML is used when present.  The fallback accepts scalar keys and a
    three-element ``center_mm`` flow sequence; rejecting unknown syntax avoids
    silently hashing one geometry while interpreting another.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        # Minimal parser for the checked-in Mustard geometry's Target block,
        # plus the flat mapping used by small focused geometry files.
        if re.search(r"(?m)^Target\s*:\s*$", text):
            values: dict[str, Any] = {"Cylinder": {}}
            in_target = False
            in_cylinder = False
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                clean = line.strip()
                if indent == 0:
                    in_target = clean == "Target:"
                    in_cylinder = False
                    continue
                if not in_target:
                    continue
                match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*", clean)
                if not match:
                    continue
                key, value = match.groups()
                if indent == 2 and key == "Cylinder" and not value:
                    in_cylinder = True
                elif indent == 2:
                    in_cylinder = False
                    if key in {"Center", "Position"}:
                        try:
                            values[key] = json.loads(value.replace("'", '"'))
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Target.{key} must be a three-number list") from exc
                        continue
                    values[key] = value.strip("'\"")
                elif indent >= 4 and in_cylinder:
                    try:
                        values["Cylinder"][key] = float(value)
                    except ValueError:
                        values["Cylinder"][key] = value.strip("'\"")
            return values

        values = {}
        for line in text.splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*", clean)
            if not match:
                raise ValueError("geometry YAML needs PyYAML for nested mappings")
            key, value = match.groups()
            if key == "center_mm":
                try:
                    values[key] = json.loads(value.replace("'", '"'))
                except json.JSONDecodeError as exc:
                    raise ValueError("center_mm must be a three-number YAML list") from exc
            elif value.lower() in {"cylinder", "box", "sphere"}:
                values[key] = value.lower()
            else:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value.strip("'\"")
        if "geometry" in values and isinstance(values["geometry"], str) and not values["geometry"]:
            values.pop("geometry")
        return values

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("geometry YAML must contain a mapping")
    # Accept either the geometry fields at the root or under `geometry`.
    nested = loaded.get("Target", loaded.get("geometry", loaded))
    if not isinstance(nested, dict):
        raise ValueError("geometry must be a mapping")
    return nested


def _load_geometry(path: str | os.PathLike[str] | None) -> tuple[dict[str, Any], str, str]:
    geometry_path = Path(path).expanduser().resolve() if path is not None else DEFAULT_GEOMETRY_PATH
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Phase I geometry YAML not found: {geometry_path}")
    raw = geometry_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    parsed = _simple_yaml_geometry(raw.decode("utf-8"))
    # Generator support is intentionally limited to the named Phase I volume.
    if "ShapeType" in parsed or "Cylinder" in parsed:
        shape = parsed.get("ShapeType")
        cylinder = parsed.get("Cylinder")
        if not isinstance(cylinder, dict):
            raise ValueError("Target.Cylinder must be a mapping")
        radius = cylinder.get("Radius")
        thickness = cylinder.get("Thickness")
        center = parsed.get("Center", parsed.get("Position", [0.0, 0.0, 0.0]))
    else:
        shape = parsed.get("shape", parsed.get("type"))
        radius = parsed.get("radius_mm", parsed.get("radius"))
        thickness = parsed.get("thickness_mm", parsed.get("thickness"))
        center = parsed.get("center_mm", parsed.get("center", [0.0, 0.0, 0.0]))
    if isinstance(center, dict):
        center = [center.get(axis) for axis in ("x", "y", "z")]
    try:
        normalized = {
            "shape": str(shape).lower(),
            "radius_mm": float(radius),
            "thickness_mm": float(thickness),
            "center_mm": [float(component) for component in center],
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry must specify cylinder shape, radius, thickness, and center") from exc
    if len(normalized["center_mm"]) != 3:
        raise ValueError("geometry center must have three coordinates")
    if normalized != DEFAULT_GEOMETRY:
        raise ValueError(
            "only the Phase I cylinder (R=30 mm, thickness=60 mm, center=(0,0,0)) is supported"
        )
    return normalized, digest, str(geometry_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def _require_outputs_available(root_path: Path, manifest_path: Path) -> None:
    for path in (root_path, manifest_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _generate_chunk(
    chunk_id: int, count: int, seed: int, particle: str
) -> tuple[int, dict[str, Any]]:
    """Spawn-process entry point. Imports NumPy locally; never imports ROOT."""
    import numpy as np

    pdg_id, mass_mev = PARTICLES[particle]
    rng = np.random.default_rng(np.random.SeedSequence([seed, chunk_id]))
    # Uniform volume in a cylinder, isotropic direction, and uniform kinetic
    # energy over the requested interval.
    radius = 30.0 * np.sqrt(rng.random(count))
    position_phi = rng.uniform(0.0, 2.0 * math.pi, count)
    cosine = rng.uniform(-1.0, 1.0, count)
    direction_phi = rng.uniform(0.0, 2.0 * math.pi, count)
    kinetic = rng.uniform(0.001, 60.0, count)
    z = rng.uniform(-30.0, 30.0, count)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    total_energy = kinetic + mass_mev
    momentum = np.sqrt(np.maximum(0.0, total_energy * total_energy - mass_mev * mass_mev))
    return chunk_id, {
        "w": np.ones(count, dtype=np.float32),
        "t": np.zeros(count, dtype=np.float64),
        "x": (radius * np.cos(position_phi)).astype(np.float32),
        "y": (radius * np.sin(position_phi)).astype(np.float32),
        "z": z.astype(np.float32),
        "pdgID": np.full(count, pdg_id, dtype=np.int32),
        "E": total_energy.astype(np.float32),
        "px": (momentum * transverse * np.cos(direction_phi)).astype(np.float32),
        "py": (momentum * transverse * np.sin(direction_phi)).astype(np.float32),
        "pz": (momentum * cosine).astype(np.float32),
    }


def _ordered_chunks(
    events: int, seed: int, particle: str, workers: int, chunk_size: int
) -> Iterator[dict[str, Any]]:
    total_chunks = (events + chunk_size - 1) // chunk_size
    if total_chunks == 0:
        return
    context = multiprocessing.get_context("spawn")
    max_pending = max(1, workers * 2)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as pool:
        futures: dict[int, concurrent.futures.Future[tuple[int, dict[str, Any]]]] = {}
        submitted = 0
        while submitted < min(total_chunks, max_pending):
            n = min(chunk_size, events - submitted * chunk_size)
            futures[submitted] = pool.submit(_generate_chunk, submitted, n, seed, particle)
            submitted += 1
        for chunk_id in range(total_chunks):
            result_id, chunk = futures.pop(chunk_id).result()
            if result_id != chunk_id:
                raise RuntimeError("worker returned a mismatched chunk ID")
            yield chunk
            if submitted < total_chunks:
                n = min(chunk_size, events - submitted * chunk_size)
                futures[submitted] = pool.submit(_generate_chunk, submitted, n, seed, particle)
                submitted += 1


def generate_source(
    particle: str = "gamma",
    events: int = 10000,
    seed: int = 1,
    output: str | os.PathLike[str] = "phasei_source.root",
    tree: str = "PhaseISource",
    workers: int | None = None,
    chunk_size: int = 10000,
    geometry: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Generate a one-particle-per-entry Phase I source ROOT tree.

    Returns the JSON-serializable manifest written beside the ROOT file.
    Requires NumPy in worker processes and PyROOT in the parent process.
    """
    if particle not in PARTICLES:
        raise ValueError("particle must be 'gamma' or 'e+'")
    if events <= 0 or chunk_size <= 0 or seed < 0:
        raise ValueError("events and chunk_size must be positive; seed must be nonnegative")
    if not tree or any(character.isspace() for character in tree):
        raise ValueError("tree must be nonempty and contain no whitespace")
    if not os.fspath(output) or any(character.isspace() for character in os.fspath(output)):
        raise ValueError("output path must be nonempty and contain no whitespace")
    if workers is None:
        workers = min(4, _available_cpu_count())
    if workers < 1:
        raise ValueError("workers must be positive")
    geometry_value, geometry_hash, geometry_path = _load_geometry(geometry)
    destination = Path(output).expanduser().resolve()
    if any(character.isspace() for character in str(destination)):
        raise ValueError("resolved output path must contain no whitespace")
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    _require_outputs_available(destination, manifest_path)
    try:
        import ROOT  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyROOT is required to write the source ROOT file") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile.Open(str(destination), "CREATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"could not create ROOT file: {destination}")
    try:
        output_tree = ROOT.TTree(tree, "Phase I single-particle source")
        record: dict[str, Any] = {
            "w": array("f", [1.0]),
            "t": array("d", [0.0]),
            "x": array("f", [0.0]),
            "y": array("f", [0.0]),
            "z": array("f", [0.0]),
        }
        for name, code in (("w", "F"), ("t", "D"), ("x", "F"), ("y", "F"), ("z", "F")):
            output_tree.Branch(name, record[name], f"{name}/{code}")
        vectors = {name: ROOT.std.vector(ctype)() for name, ctype in (
            ("pdgID", "int"), ("E", "float"), ("px", "float"),
            ("py", "float"), ("pz", "float")
        )}
        for name, vector in vectors.items():
            output_tree.Branch(name, vector)

        written = 0
        for chunk in _ordered_chunks(events, seed, particle, workers, chunk_size):
            count = len(chunk["w"])
            for index in range(count):
                for scalar in record:
                    record[scalar][0] = float(chunk[scalar][index])
                for name, vector in vectors.items():
                    vector.clear()
                    vector.push_back(chunk[name][index].item())
                output_tree.Fill()
            written += count
        output_tree.Write()
    finally:
        root_file.Close()

    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "particle": particle,
        "events": written,
        "tree": tree,
        "source_file": str(destination),
        "sha256": source_hash,
        "generator_source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        },
        "settings": {
            "particle": particle,
            "pdg_id": PARTICLES[particle][0],
            "events": events,
            "seed": seed,
            "workers": workers,
            "chunk_size": chunk_size,
            "tree": tree,
            "output": str(destination),
            "geometry_path": geometry_path,
            "sampling": {
                "position": "uniform volume in cylinder",
                "direction": "isotropic (uniform cos(theta) in [-1,1], phi in [0,2pi))",
                "kinetic_energy_mev": [0.001, 60.0],
                "time_ns": 0.0,
                "weight": 1.0,
            },
        },
        "units": {"position": "mm", "time": "ns", "energy": "MeV", "momentum": "MeV/c"},
        "geometry": geometry_value,
        "geometry_sha256": geometry_hash,
        "count": written,
        "branch_contract": BRANCH_CONTRACT,
    }
    with manifest_path.open("x", encoding="utf-8") as manifest_stream:
        manifest_stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle", choices=tuple(PARTICLES), required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree", default="PhaseISource")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: min(4, CPUs available to this process))",
    )
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--geometry", help="YAML describing the supported Phase I cylinder")
    args = parser.parse_args(argv)
    manifest = generate_source(
        particle=args.particle,
        events=args.events,
        seed=args.seed,
        output=args.output,
        tree=args.tree,
        workers=args.workers,
        chunk_size=args.chunk_size,
        geometry=args.geometry,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
