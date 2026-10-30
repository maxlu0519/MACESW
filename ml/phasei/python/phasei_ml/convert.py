"""Fast conversion of unmerged Phase-I ROOT shards into event-indexed arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import dataset


class ConvertError(dataset.DatasetError):
    """Input data cannot be safely converted."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _root_version(ROOT: Any) -> str:
    return str(ROOT.gROOT.GetVersion())


def _validate_event_coverage(event_ids: np.ndarray, n_events: int, context: str) -> np.ndarray:
    """Return stable event order or fail on duplicate/missing/out-of-range IDs."""
    evt = np.asarray(event_ids, dtype=np.int64)
    if np.any((evt < 0) | (evt >= n_events)):
        raise ConvertError(f"{context} has out-of-range EvtID")
    order = np.argsort(evt, kind="stable")
    ordered = evt[order]
    if not np.array_equal(ordered, np.arange(n_events, dtype=np.int64)):
        unique, counts = np.unique(ordered, return_counts=True)
        dups = unique[counts > 1][:8].tolist()
        missing = np.setdiff1d(np.arange(n_events, dtype=np.int64), unique)[:8].tolist()
        raise ConvertError(f"{context} duplicate/missing EvtID; duplicates={dups}, missing={missing}")
    return order


def _asnumpy(ROOT: Any, paths: list[Path], tree_path: str, defines: dict[str, str], columns: list[str], context: str):
    chain = ROOT.TChain(tree_path)
    for path in paths:
        if chain.Add(str(path)) != 1:
            raise ConvertError(f"cannot add ROOT shard {path} for {tree_path}")
    if chain.GetEntries() <= 0:
        # The tree schemas were already checked in each shard. Empty detector trees are valid.
        return {column: np.empty((0,), dtype=np.int64 if column in ("EvtID", "channel_scalar") else np.float32)
                for column in columns}
    df = ROOT.RDataFrame(chain)
    for name, expression in defines.items():
        df = df.Define(name, expression)
    try:
        return df.AsNumpy(columns)
    except Exception as exc:
        raise ConvertError(f"failed extracting {context} columns {columns}: {exc}") from exc


def _check_tree_schemas(ROOT: Any, files: list[Path], run_id: int) -> None:
    trees = [("SimPrimaryVertex", ("EvtID", *dataset.LABEL_FIELDS), dataset._PRIMARY_TYPES)]
    for detector, fields in dataset.FEATURES.items():
        trees.append(({"TTC": "TTCSimHit", "SciFi": "SciFiSimHit", "ECAL": "ECALSimHit"}[detector],
                      ("EvtID", *fields), dataset._HIT_TYPES))
    for path in files:
        handle = ROOT.TFile.Open(str(path), "READ")
        if not handle or handle.IsZombie():
            raise ConvertError(f"cannot open simulation shard {path}")
        try:
            for name, fields, types in trees:
                tree = handle.Get(f"G4Run{run_id}/{name}")
                if not tree:
                    raise ConvertError(f"missing G4Run{run_id}/{name} in {path}")
                dataset._require_fields(tree, fields, types, f"{path}:G4Run{run_id}/{name}")
        finally:
            handle.Close()


def _extract_source(ROOT: Any, source: dict, base: Path, sim: dict, sim_base: Path,
                    source_manifest_path: Path | None, n: int, run_id: int,
                    evt: np.ndarray, labels: dict[str, np.ndarray]) -> None:
    source_handle, source_tree = dataset._source_tree(source, base, n, ROOT)
    try:
        row = np.arange(n, dtype=np.int64)
        # The current Mustard reader maps EvtID to the next source row, wrapping at N.
        index = (evt + 1) % n
        if len(np.unique(evt)) != n or not np.array_equal(np.sort(evt), row):
            raise ConvertError("SimPrimaryVertex EvtID coverage is not exactly [0, N)")
        source_tree_path = str(source["tree"])
        cols = ["pdg_scalar", "energy_scalar", "px_scalar", "py_scalar", "pz_scalar", "t", "x", "y", "z"]
        defines = {"pdg_scalar": "int(pdgID[0])", "energy_scalar": "float(E[0])",
                   "px_scalar": "float(px[0])", "py_scalar": "float(py[0])", "pz_scalar": "float(pz[0])"}
        chain = ROOT.TChain(source_tree_path)
        source_file = dataset._resolve(base, str(source["source_file"]))
        chain.Add(str(source_file))
        df = ROOT.RDataFrame(chain)
        for name, expr in defines.items():
            df = df.Define(name, expr)
        data = df.AsNumpy(cols)
        if len(data["pdg_scalar"]) != n:
            raise ConvertError(f"source has {len(data['pdg_scalar'])} rows, expected {n}")
        # Compare all primary observables in bulk; no Python event loop.
        mass = 0.51099895 if source["particle"] == "e+" else 0.0
        checks = [
            (np.asarray(data["pdg_scalar"], dtype=np.int32)[index], labels["PDGID"][evt], "PDGID", 0),
            (np.asarray(data["energy_scalar"], dtype=np.float64)[index] - mass, labels["Ek0"], "Ek0", 2e-5),
            (np.asarray(data["t"], dtype=np.float64)[index], labels["t0"], "t0", 2e-5),
        ]
        for actual, expected, name, atol in checks:
            if not np.allclose(actual, expected, rtol=2e-5, atol=atol):
                bad = int(np.flatnonzero(~np.isclose(actual, expected, rtol=2e-5, atol=atol))[0])
                raise ConvertError(f"source row mapping mismatch at EvtID {bad}: {name}")
        for scalar in ("x", "y", "z"):
            pos = "xyz".index(scalar)
            actual = np.asarray(data[scalar], dtype=np.float64)[index]
            if not np.allclose(actual, labels["x0"][:, pos], rtol=2e-5, atol=2e-5):
                raise ConvertError(f"source row mapping mismatch: {scalar} at EvtID {int(evt[0])}")
        for scalar in ("px_scalar", "py_scalar", "pz_scalar"):
            pos = ("px_scalar", "py_scalar", "pz_scalar").index(scalar)
            actual = np.asarray(data[scalar], dtype=np.float64)[index]
            if not np.allclose(actual, labels["p0"][:, pos], rtol=2e-5, atol=2e-5):
                raise ConvertError(f"source row mapping mismatch: {scalar}")
    finally:
        source_handle.Close()


def _extract_pair(source_value: str | Path, sim_value: str | Path):
    source, source_base = dataset._manifest(source_value)
    sim, sim_base = dataset._manifest(sim_value)
    source_path = Path(source_value).expanduser().resolve()
    sim_path = Path(sim_value).expanduser().resolve()
    dataset._check_manifest(source, "source")
    dataset._check_manifest(sim, "simulation")
    if sim.get("completed") is not True:
        raise ConvertError("simulation manifest must be completed")
    n = int(source.get("events", 0))
    if n < 1 or int(sim.get("events", -1)) != n:
        raise ConvertError("source/simulation event count mismatch")
    for key in ("particle", "tree", "geometry_sha256"):
        if source.get(key) != sim.get(key, sim.get("geometry_hash")):
            raise ConvertError(f"simulation {key} does not match source")
    dataset._verify_sim_source_link(sim, sim_base, source, source_base, source_path)
    source_id = str(source["sha256"]).lower()
    source_file = dataset._resolve(source_base, str(source["source_file"]))
    if _sha256(source_file) != source_id:
        raise ConvertError(f"source ROOT SHA-256 mismatch: {source_file}")
    files = dataset._simulation_files(sim, sim_base)
    for path in files:
        if not path.is_file():
            raise ConvertError(f"missing simulation shard: {path}")
    ROOT = dataset._root()
    run_ids = set()
    for path in files:
        ids = dataset._run_ids(ROOT, path)
        if not ids:
            raise ConvertError(f"no G4Run in {path}")
        run_ids.update(ids)
    if len(run_ids) != 1:
        raise ConvertError(f"expected one G4Run, found {sorted(run_ids)}")
    run_id = next(iter(run_ids))
    _check_tree_schemas(ROOT, files, run_id)
    primary = _asnumpy(ROOT, files, f"G4Run{run_id}/SimPrimaryVertex",
                       {"x0x": "float(x0[0])", "x0y": "float(x0[1])", "x0z": "float(x0[2])",
                        "p0x": "float(p0[0])", "p0y": "float(p0[1])", "p0z": "float(p0[2])"},
                       ["EvtID", "PDGID", "t0", "Ek0", "x0x", "x0y", "x0z", "p0x", "p0y", "p0z"],
                       "SimPrimaryVertex")
    evt = np.asarray(primary["EvtID"], dtype=np.int64)
    order = _validate_event_coverage(evt, n, "SimPrimaryVertex")
    evt = evt[order]
    labels = {name: np.asarray(primary[name])[order] for name in ("PDGID", "t0", "Ek0")}
    labels["x0"] = np.column_stack([np.asarray(primary[f"x0{c}"])[order] for c in "xyz"]).astype(np.float32)
    labels["p0"] = np.column_stack([np.asarray(primary[f"p0{c}"])[order] for c in "xyz"]).astype(np.float32)
    labels["PDGID"] = labels["PDGID"].astype(np.int32)
    for name in ("t0", "Ek0"):
        labels[name] = labels[name].astype(np.float32)
    _extract_source(ROOT, source, source_base, sim, sim_base, source_path, n, run_id, evt, labels)

    hits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for detector, fields in dataset.FEATURES.items():
        tree_name = {"TTC": "TTCSimHit", "SciFi": "SciFiSimHit", "ECAL": "ECALSimHit"}[detector]
        channel, _, _ = fields
        raw = _asnumpy(ROOT, files, f"G4Run{run_id}/{tree_name}",
                       {"channel_scalar": f"int({channel})"},
                       ["EvtID", "channel_scalar", "t", "Edep"], f"{tree_name}")
        hit_evt = np.asarray(raw["EvtID"], dtype=np.int64)
        if np.any((hit_evt < 0) | (hit_evt >= n)):
            raise ConvertError(f"{tree_name} has out-of-range EvtID")
        horder = np.argsort(hit_evt, kind="stable")
        hit_evt = hit_evt[horder]
        channels = np.asarray(raw["channel_scalar"], dtype=np.int32)[horder]
        times = np.asarray(raw["t"], dtype=np.float32)[horder]
        edeps = np.asarray(raw["Edep"], dtype=np.float32)[horder]
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(hit_evt, minlength=n), out=offsets[1:])
        hits[detector] = (channels, times, edeps, offsets)
    return source, sim, source_path, sim_path, source_base, sim_base, files, labels, hits


def _stratified_split(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Split each of 16 species×response-mask strata as 80/10/10."""
    rng = np.random.default_rng(seed)
    result = np.empty(len(mask), dtype=np.uint8)
    for stratum in np.unique(mask):
        indices = np.flatnonzero(mask == stratum)
        rng.shuffle(indices)
        n = len(indices)
        train = int(np.floor(.8 * n))
        val = int(np.floor(.1 * n))
        result[indices[:train]] = 0
        result[indices[train:train + val]] = 1
        result[indices[train + val:]] = 2
    return result


def _write_pair(out: Path, particle: str, source, sim, source_path: Path, sim_path: Path,
                source_base: Path, sim_base: Path, files: list[Path], labels, hits) -> dict:
    n = len(labels["PDGID"])
    directory = out / particle
    directory.mkdir(parents=True)
    np.save(directory / "evt_id.npy", np.arange(n, dtype=np.int64), allow_pickle=False)
    target = np.zeros(n, dtype=np.int8) if particle == "gamma" else np.ones(n, dtype=np.int8)
    np.save(directory / "pid.npy", target, allow_pickle=False)
    np.save(directory / "pdgid.npy", labels["PDGID"], allow_pickle=False)
    for name in ("t0", "Ek0", "x0", "p0"):
        np.save(directory / f"{name}.npy", labels[name], allow_pickle=False)
    pattern = np.zeros(n, dtype=np.uint8)
    hit_counts = {}
    for bit, detector in enumerate(("TTC", "SciFi", "ECAL")):
        ch, t, edep, offsets = hits[detector]
        for name, array in (("channel", ch), ("t", t), ("Edep", edep), ("offsets", offsets)):
            np.save(directory / f"{detector.lower()}_{name}.npy", array, allow_pickle=False)
        counts = np.diff(offsets)
        pattern |= (counts > 0).astype(np.uint8) << bit
        hit_counts[detector] = int(len(ch))
    np.save(directory / "response_mask.npy", pattern, allow_pickle=False)
    split = _stratified_split((target.astype(np.uint8) << 3) | pattern)
    np.save(directory / "split.npy", split, allow_pickle=False)
    # Train-only normalization for raw time and log(1+Edep/MeV), pooled across each detector.
    normalization = {}
    for detector in ("TTC", "SciFi", "ECAL"):
        ch, times, edeps, _ = hits[detector]
        train_evt = split == 0
        train_hit = train_evt[np.repeat(np.arange(n, dtype=np.int64), np.diff(hits[detector][3]))]
        train_t = times[train_hit].astype(np.float64)
        train_e = np.log1p(np.maximum(edeps[train_hit].astype(np.float64), 0.0))
        t_mean = float(train_t.mean()) if len(train_t) else 0.0
        t_std = float(train_t.std()) if len(train_t) else 1.0
        e_mean = float(train_e.mean()) if len(train_e) else 0.0
        e_std = float(train_e.std()) if len(train_e) else 1.0
        normalization[detector] = {"t_mean": t_mean, "t_std": max(t_std, 1e-8),
                                   "log1p_edep_mean": e_mean, "log1p_edep_std": max(e_std, 1e-8),
                                   "channel_count": int(ch.max()) + 1 if len(ch) else 1}
    norm_file = directory / "normalization.json"
    norm_file.write_text(json.dumps(normalization, indent=2) + "\n")
    checksums = {str(path): _sha256(path) for path in files}
    command = sim.get("command", [])
    exe_hash = None
    if isinstance(command, list):
        candidates = [Path(str(token)) for token in command if Path(str(token)).name == "MACE"]
        if not candidates:
            candidates = [Path(str(token)) for token in command if Path(str(token)).is_file() and os.access(str(token), os.X_OK)]
        if candidates and candidates[0].is_file():
            exe_hash = {"path": str(candidates[0]), "sha256": _sha256(candidates[0])}
    return {"particle": particle, "events": n, "response_masks": {str(i): int((pattern == i).sum()) for i in range(8)},
            "hit_counts": hit_counts, "source_manifest": str(source_path),
            "source_manifest_sha256": _sha256(source_path), "source_sha256": source["sha256"],
            "simulation_manifest": str(sim_path), "simulation_manifest_sha256": _sha256(sim_path),
            "simulation_shards": [{"path": str(p), "sha256": checksums[str(p)]} for p in files],
            "binary_sha256": exe_hash, "geometry_sha256": source["geometry_sha256"],
            "normalization": normalization, "split_counts": {str(i): int((split == i).sum()) for i in range(3)},
            "root_version": _root_version(dataset._root())}


def convert(pairs: list[tuple[str | Path, str | Path]], output: str | Path) -> dict:
    if not pairs:
        raise ConvertError("provide at least one --pair SOURCE_MANIFEST SIM_MANIFEST")
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ConvertError(f"output already exists: {out}")
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    started = time.perf_counter()
    try:
        seen_particles = set()
        entries = []
        for source_value, sim_value in pairs:
            result = _extract_pair(source_value, sim_value)
            source, sim, source_path, sim_path, source_base, sim_base, files, labels, hits = result
            particle = str(source["particle"])
            if particle in seen_particles:
                raise ConvertError(f"duplicate particle dataset {particle}")
            seen_particles.add(particle)
            entries.append(_write_pair(tmp, particle, source, sim, source_path, sim_path,
                                       source_base, sim_base, files, labels, hits))
        manifest = {"schema_version": 1, "feature_whitelist": {name: list(fields) for name, fields in dataset.FEATURES.items()},
                    "truth_fields_excluded_from_features": ["PDGID", "t0", "x0", "Ek0", "p0", "EvtID"],
                    "split_seed": 42, "split_definition": {"train": .8, "validation": .1, "test": .1},
                    "particles": entries, "created_unix": time.time(), "elapsed_seconds": time.perf_counter() - started,
                    "python": platform.python_version()}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        return manifest
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE_MANIFEST", "SIM_MANIFEST"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = convert([tuple(pair) for pair in args.pair], args.output)
    except (ConvertError, OSError, ValueError) as exc:
        parser.error(str(exc))
    for item in result["particles"]:
        print(f"{item['particle']}: {item['events']} events, masks={item['response_masks']}")
    print(f"Published {args.output} in {result['elapsed_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
