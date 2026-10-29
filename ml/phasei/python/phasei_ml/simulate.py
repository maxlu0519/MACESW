"""Launch reproducible MPI SimMACEPhaseI runs from generated event data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Sequence


class SimulationError(RuntimeError):
    """Raised when a simulation cannot be prepared or completed."""


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], Path, Path]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SimulationError("source manifest must contain a JSON object")
    if manifest.get("schema_version") != 1:
        raise SimulationError("source manifest schema_version must be 1")
    tree = manifest.get("tree")
    if not isinstance(tree, str) or not tree or any(c.isspace() for c in tree):
        raise SimulationError("source manifest tree must be a nonempty whitespace-free string")
    events = manifest.get("events")
    if isinstance(events, bool) or not isinstance(events, int) or events <= 0:
        raise SimulationError("source manifest events must be a positive integer")
    raw_file = manifest.get("source_file")
    if not isinstance(raw_file, str) or not raw_file:
        raise SimulationError("source manifest source_file must be a path")
    source_file = Path(raw_file).expanduser()
    if not source_file.is_absolute():
        source_file = (path.parent / source_file).resolve()
    else:
        source_file = source_file.resolve()
    if not source_file.is_file():
        raise SimulationError(f"source event file does not exist: {source_file}")
    if any(c.isspace() for c in str(source_file)):
        raise SimulationError("source event file path contains whitespace; the MACE data reader cannot parse it")
    expected_hash = manifest.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SimulationError("source manifest sha256 must be a 64-character SHA-256 digest")
    import hashlib

    digest = hashlib.sha256()
    with source_file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != expected_hash.lower():
        raise SimulationError(f"source event file hash does not match manifest: {source_file}")
    settings = manifest.get("settings")
    raw_geometry = settings.get("geometry_path") if isinstance(settings, dict) else None
    geometry_hash = manifest.get("geometry_sha256")
    if not isinstance(raw_geometry, str) or not raw_geometry:
        raise SimulationError("source manifest settings.geometry_path is required for simulation")
    if not isinstance(geometry_hash, str) or len(geometry_hash) != 64:
        raise SimulationError("source manifest geometry_sha256 must be a 64-character SHA-256 digest")
    geometry_path = Path(raw_geometry).expanduser()
    if not geometry_path.is_absolute():
        geometry_path = (path.parent / geometry_path).resolve()
    else:
        geometry_path = geometry_path.resolve()
    if not geometry_path.is_file():
        raise SimulationError(f"geometry file does not exist: {geometry_path}")
    if any(c.isspace() for c in str(geometry_path)):
        raise SimulationError("geometry file path contains whitespace; the MACE macro parser cannot parse it")
    geometry_digest = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    if geometry_digest.lower() != geometry_hash.lower():
        raise SimulationError(f"geometry file hash does not match source manifest: {geometry_path}")
    return manifest, source_file, geometry_path


def _scheduler_info() -> dict[str, Any]:
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "PBS_JOBID", "LSB_JOBID", "JOB_ID")
    found = {key: os.environ.get(key) for key in keys if os.environ.get(key)}
    mustard_scheduler = os.environ.get("MUSTARD_EXECUTION_SCHEDULER")
    return {
        "kind": "batch" if found else "local",
        "environment": found or None,
        "mustard_execution_scheduler": mustard_scheduler or "Mustard default",
    }


def _available_cpu_count() -> int:
    affinity_getter = getattr(os, "sched_getaffinity", None)
    if affinity_getter is not None:
        try:
            available = len(affinity_getter(0))
            if available > 0:
                return available
        except OSError:
            pass
    return os.cpu_count() or 1


def _command_parts(value: str | os.PathLike[str] | Sequence[str]) -> list[str]:
    if isinstance(value, (str, os.PathLike)):
        parts = shlex.split(os.fspath(value))
    else:
        parts = [os.fspath(part) for part in value]
    if not parts:
        raise SimulationError("mpiexec command cannot be empty")
    return parts


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_simulation(
    source_manifest: str | os.PathLike[str],
    output_base: str | os.PathLike[str],
    ranks: int | None = None,
    mace_bin: str | os.PathLike[str] = "MACE",
    mpiexec: str | os.PathLike[str] | Sequence[str] = "mpiexec",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare and optionally run MPI simulation; return its simulation manifest.

    ``output_base`` is a new or empty directory. Per-rank ROOT files and the
    ``simulation_manifest.json`` record are written below it.
    """
    source_manifest_path = Path(source_manifest).expanduser().resolve()
    source, source_file, geometry_path = _load_source_manifest(source_manifest_path)
    out_dir = Path(output_base).expanduser().resolve()
    if any(c.isspace() for c in str(out_dir)):
        raise SimulationError("output path contains whitespace; the MACE macro parser cannot parse it")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise SimulationError(f"output directory must be empty to prevent stale ROOT files being accepted: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    available_cpu_count = _available_cpu_count()
    if ranks is None:
        ranks = min(4, available_cpu_count, source["events"])
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise SimulationError("ranks must be a positive integer")
    if ranks > source["events"]:
        raise SimulationError(f"ranks ({ranks}) cannot exceed source event count ({source['events']})")

    macro_path = out_dir / "simulate.mac"
    output_stem = "SimMACEPhaseI.root"
    macro_lines = [
        f"/Mustard/Detector/Description/Import {geometry_path}",
        "/run/initialize",
        f"/MACE/Analysis/FilePath {output_stem}",
        "/MACE/Analysis/FileMode RECREATE",
        "/MACE/Generator/SwitchToFromDataPrimaryGenerator",
        f"/Mustard/Generator/DataReaderPrimaryGenerator/EventData {source_file} {source['tree']}",
        "/Mustard/Generator/DataReaderPrimaryGenerator/NVertex 1",
        "/MACE/Analysis/SavePrimaryVertexData yes",
        "/MACE/Analysis/CoincidenceWithECAL no",
        f"/run/beamOn {source['events']}",
        "/Mustard/Run/PrintRunSummary",
    ]
    macro_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    exe_parts = _command_parts(mace_bin)
    mpi_parts = _command_parts(mpiexec)
    command = [*mpi_parts, "-n", str(ranks), *exe_parts, "PhaseI", "SimMACEPhaseI", str(macro_path)]
    record_path = out_dir / "simulation_manifest.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": __import__("hashlib").sha256(source_manifest_path.read_bytes()).hexdigest(),
        "source_file": str(source_file),
        "source_sha256": source["sha256"],
        "particle": source.get("particle"),
        "events": source["events"],
        "tree": source["tree"],
        "geometry_hash": source.get("geometry_hash", source.get("geometry_sha256")),
        "geometry_sha256": source["geometry_sha256"],
        "geometry_path": str(geometry_path),
        "geometry": source.get("geometry"),
        "settings": source.get("settings"),
        "mpi_ranks": ranks,
        "available_cpu_count": available_cpu_count,
        "scheduler": _scheduler_info(),
        "command": command,
        "macro_file": str(macro_path),
        "output_base": str(out_dir),
        "output_files": [],
        "completed": False,
        "dry_run": bool(dry_run),
        "returncode": None,
    }
    if dry_run:
        record["status"] = "dry_run"
        _write_json(record_path, record)
        return record

    executable = shutil.which(mpi_parts[0]) if not Path(mpi_parts[0]).is_absolute() else mpi_parts[0]
    if not executable:
        record["status"] = "failed"
        record["error"] = f"MPI launcher not found: {mpi_parts[0]}"
        _write_json(record_path, record)
        raise SimulationError(f"MPI launcher not found: {mpi_parts[0]}; install MPI or pass --mpiexec")
    command[0] = executable
    record["command"] = command
    binary = shutil.which(exe_parts[0]) if not Path(exe_parts[0]).is_absolute() else exe_parts[0]
    if not binary:
        record["status"] = "failed"
        record["error"] = f"MACE executable not found: {exe_parts[0]}"
        _write_json(record_path, record)
        raise SimulationError(f"MACE executable not found: {exe_parts[0]}; pass --mace-bin")
    command[len(mpi_parts) + 2] = binary
    record["command"] = command
    try:
        completed = subprocess.run(command, cwd=out_dir, check=False)
        record["returncode"] = completed.returncode
        if completed.returncode != 0:
            record["status"] = "failed"
            record["error"] = f"MPI simulation exited with status {completed.returncode}"
            raise SimulationError(record["error"])
        root_files = sorted(
            file.relative_to(out_dir).as_posix()
            for file in out_dir.rglob("*.root")
            if file.is_file() and file.stat().st_size > 0
        )
        if not root_files:
            record["status"] = "failed"
            record["error"] = "simulation exited successfully but produced no nonempty ROOT files"
            raise SimulationError(record["error"])
        record["output_files"] = root_files
        record["completed"] = True
        record["status"] = "complete"
    except OSError as exc:
        record["status"] = "failed"
        record["error"] = f"could not launch simulation: {exc}"
        raise SimulationError(record["error"]) from exc
    finally:
        _write_json(record_path, record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimMACEPhaseI from generated event data")
    parser.add_argument("source_manifest", help="JSON manifest produced by phasei_ml.source")
    parser.add_argument("output_base", help="new or empty directory for this simulation")
    parser.add_argument("--ranks", type=int, help="MPI ranks (default: min(4, CPU count, event count))")
    parser.add_argument("--mace-bin", default="MACE", help="MACE executable (default: MACE)")
    parser.add_argument("--mpiexec", default="mpiexec", help="MPI launcher command (default: mpiexec)")
    parser.add_argument("--dry-run", action="store_true", help="write the macro and launch record without running")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_simulation(
            args.source_manifest,
            args.output_base,
            ranks=args.ranks,
            mace_bin=args.mace_bin,
            mpiexec=args.mpiexec,
            dry_run=args.dry_run,
        )
    except SimulationError as exc:
        print(f"phasei simulation: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
