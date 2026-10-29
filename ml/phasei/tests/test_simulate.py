import hashlib
import json
from pathlib import Path

import pytest

from phasei_ml.simulate import SimulationError, run_simulation


def _source(tmp_path: Path) -> Path:
    source_file = tmp_path / "generated.root"
    source_file.write_bytes(b"fake root input")
    geometry_file = tmp_path / "geometry.yaml"
    geometry_file.write_text("geometry: test\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "particle": "gamma",
        "events": 7,
        "tree": "EventData",
        "source_file": str(source_file),
        "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(geometry_file.read_bytes()).hexdigest(),
        "settings": {"seed": 42, "geometry_path": str(geometry_file)},
    }
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_dry_run_writes_macro_and_manifest_without_execution(tmp_path):
    source = _source(tmp_path)
    result = run_simulation(source, tmp_path / "simulation", dry_run=True)

    assert result["completed"] is False
    assert result["status"] == "dry_run"
    assert result["mpi_ranks"] >= 1
    assert result["output_files"] == []
    macro = (tmp_path / "simulation" / "simulate.mac").read_text(encoding="utf-8")
    assert "/MACE/Generator/SwitchToFromDataPrimaryGenerator" in macro
    assert "/Mustard/Generator/DataReaderPrimaryGenerator/NVertex 1" in macro
    assert "/MACE/Analysis/SavePrimaryVertexData yes" in macro
    assert "/MACE/Analysis/CoincidenceWithECAL no" in macro
    assert f"/Mustard/Detector/Description/Import {tmp_path / 'geometry.yaml'}" in macro
    assert "/MACE/Analysis/FilePath SimMACEPhaseI.root" in macro
    assert "/run/beamOn 7" in macro
    assert macro.index("/Mustard/Detector/Description/Import") < macro.index("/run/initialize")
    assert macro.index("/run/initialize") < macro.index("/MACE/Analysis/FilePath")


def test_default_rank_count_uses_available_cpu_count(tmp_path, monkeypatch):
    monkeypatch.setattr("phasei_ml.simulate._available_cpu_count", lambda: 32)
    source = _source(tmp_path)
    result = run_simulation(source, tmp_path / "simulation", dry_run=True)
    assert result["mpi_ranks"] == 4
    assert result["available_cpu_count"] == 32


def test_explicit_rank_count_cannot_exceed_event_count(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(SimulationError, match="cannot exceed source event count"):
        run_simulation(source, tmp_path / "simulation", ranks=8, dry_run=True)


def test_existing_nonempty_output_directory_is_rejected(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "simulation"
    output.mkdir()
    (output / "old.root").write_bytes(b"stale")

    with pytest.raises(SimulationError, match="must be empty"):
        run_simulation(source, output, dry_run=True)


def test_mpi_success_records_nested_rank_files(tmp_path, monkeypatch):
    source = _source(tmp_path)

    class Result:
        returncode = 0

    def fake_run(command, cwd, check):
        rank_dir = Path(cwd) / "node0"
        rank_dir.mkdir()
        (rank_dir / "SimMACEPhaseI.root").write_bytes(b"rank output")
        return Result()

    monkeypatch.setattr("phasei_ml.simulate.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("phasei_ml.simulate.subprocess.run", fake_run)
    result = run_simulation(source, tmp_path / "simulation")

    assert result["completed"] is True
    assert result["status"] == "complete"
    assert result["output_files"] == ["node0/SimMACEPhaseI.root"]
    assert result["source_manifest"] == str(source.resolve())
    assert result["geometry_path"] == str((tmp_path / "geometry.yaml").resolve())
    saved = json.loads((tmp_path / "simulation" / "simulation_manifest.json").read_text())
    assert saved["completed"] is True


def test_mpi_failure_writes_incomplete_manifest(tmp_path, monkeypatch):
    source = _source(tmp_path)

    class Result:
        returncode = 9

    monkeypatch.setattr("phasei_ml.simulate.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("phasei_ml.simulate.subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(SimulationError, match="status 9"):
        run_simulation(source, tmp_path / "simulation")
    saved = json.loads((tmp_path / "simulation" / "simulation_manifest.json").read_text())
    assert saved["completed"] is False
    assert saved["output_files"] == []


def test_source_hash_mismatch_is_rejected(tmp_path):
    source = _source(tmp_path)
    data = json.loads(source.read_text())
    data["sha256"] = "0" * 64
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SimulationError, match="hash does not match"):
        run_simulation(source, tmp_path / "simulation", dry_run=True)


def test_geometry_hash_mismatch_is_rejected(tmp_path):
    source = _source(tmp_path)
    data = json.loads(source.read_text())
    data["geometry_sha256"] = "0" * 64
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SimulationError, match="geometry file hash does not match"):
        run_simulation(source, tmp_path / "simulation", dry_run=True)
