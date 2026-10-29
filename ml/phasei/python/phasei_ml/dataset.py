"""Read Phase-I simulation shards into event records without truth leakage.

The reader deliberately uses only PyROOT.  Simulation files are kept as
unmerged MPI shards: event IDs, rather than file names or entry order, define
the join between hit tuples and primary vertices.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


FEATURES = {
    "TTC": ("TileID", "t", "Edep"),
    "SciFi": ("FiberID", "t", "Edep"),
    "ECAL": ("ModID", "t", "Edep"),
}
LABEL_FIELDS = ("PDGID", "t0", "x0", "Ek0", "p0")
_PRIMARY_TYPES = {"EvtID": {"Int_t", "int"}, "PDGID": {"Int_t", "int"},
                  "t0": {"Double_t", "double"}, "Ek0": {"Float_t", "float"}}
_HIT_TYPES = {"TileID": {"Short_t", "short"}, "FiberID": {"Short_t", "short"},
              "ModID": {"Short_t", "short"}, "t": {"Double_t", "double"},
              "Edep": {"Float_t", "float"}, "EvtID": {"Int_t", "int"}}


class DatasetError(ValueError):
    """Raised when manifests or ROOT data cannot be joined safely."""


def _manifest(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    if isinstance(value, Mapping):
        return dict(value), Path.cwd()
    path = Path(value).expanduser().resolve()
    try:
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read manifest {path}: {exc}") from exc


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _check_manifest(m: Mapping[str, Any], kind: str) -> None:
    if m.get("schema_version") != 1:
        raise DatasetError(f"{kind} manifest schema_version must be 1")


def _root():
    try:
        import ROOT  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyROOT is required to read Phase-I ROOT files") from exc
    return ROOT


def _open_tree(ROOT: Any, path: Path, tree_path: str):
    handle = ROOT.TFile.Open(str(path), "READ")
    if not handle or handle.IsZombie():
        raise DatasetError(f"cannot open ROOT file {path}")
    tree = handle.Get(tree_path)
    if not tree:
        handle.Close()
        raise DatasetError(f"missing TTree {tree_path!r} in {path}")
    return handle, tree


def _leaf_type(tree: Any, name: str) -> str | None:
    branch = tree.GetBranch(name)
    if not branch:
        return None
    leaf = branch.GetLeaf(name)
    if not leaf and branch.GetListOfLeaves() and branch.GetListOfLeaves().GetEntries() == 1:
        leaf = branch.GetListOfLeaves().At(0)
    return str(leaf.GetTypeName()) if leaf else None


def _require_fields(tree: Any, required: Sequence[str], types: Mapping[str, set[str]], context: str) -> None:
    missing = [name for name in required if not tree.GetBranch(name)]
    if missing:
        raise DatasetError(f"{context} missing required branches: {', '.join(missing)}")
    for name, accepted in types.items():
        if name not in required:
            continue
        actual = _leaf_type(tree, name)
        if actual is not None and actual not in accepted:
            raise DatasetError(f"{context}.{name} has ROOT type {actual!r}, expected {sorted(accepted)}")


def _require_class_branches(tree: Any, classes: Mapping[str, set[str]], context: str) -> None:
    for name, accepted in classes.items():
        branch = tree.GetBranch(name)
        actual = str(branch.GetClassName()) if branch and hasattr(branch, "GetClassName") else ""
        if actual and actual not in accepted:
            raise DatasetError(f"{context}.{name} has ROOT class {actual!r}, expected {sorted(accepted)}")


def _value(tree: Any, name: str) -> Any:
    value = getattr(tree, name)
    # Convert std::array, vector, and PyROOT proxies to plain Python values.
    if isinstance(value, (str, bytes)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _vec3(value: Any, context: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{context} is not a 3-vector") from exc
    if len(result) != 3:
        raise DatasetError(f"{context} is not a 3-vector")
    return result  # type: ignore[return-value]


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=2e-5, abs_tol=2e-5)


_PDG_MASS_MEV = {22: 0.0, -11: 0.51099895}
_PARTICLE_PDG = {"gamma": 22, "e+": -11}


def _validate_source_row(row: Mapping[str, Any], vertex: Mapping[str, Any], particle_name: str, context: str) -> None:
    """Compare the generator row (after SetEntry/Next) with saved truth."""
    try:
        pdg = int(row["pdgID"][0])
        energy = float(row["E"][0])
        momentum = tuple(float(row[k][0]) for k in ("px", "py", "pz"))
        position = tuple(float(row[k]) for k in ("x", "y", "z"))
        time = float(row["t"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DatasetError(f"{context}: malformed source generator row") from exc
    expected_pdg = _PARTICLE_PDG.get(particle_name)
    if expected_pdg is None:
        raise DatasetError(f"unsupported source particle {particle_name!r}; v1 supports gamma and e+")
    if pdg != expected_pdg:
        raise DatasetError(f"{context}: source PDGID {pdg} disagrees with manifest particle {particle_name!r}")
    if pdg != int(vertex["PDGID"]):
        raise DatasetError(f"{context}: source PDGID {pdg} disagrees with SimPrimaryVertex {vertex['PDGID']}")
    mass = _PDG_MASS_MEV[pdg]
    if not _close(energy - mass, vertex["Ek0"]):
        raise DatasetError(f"{context}: source kinetic energy disagrees with SimPrimaryVertex Ek0")
    if any(not _close(a, b) for a, b in zip(momentum, vertex["p0"])):
        raise DatasetError(f"{context}: source momentum disagrees with SimPrimaryVertex p0")
    if any(not _close(a, b) for a, b in zip(position, vertex["x0"])) or not _close(time, vertex["t0"]):
        raise DatasetError(f"{context}: source vertex disagrees with SimPrimaryVertex x0/t0")


def _source_tree(source: Mapping[str, Any], source_base: Path, n_events: int, ROOT: Any):
    source_file = _resolve(source_base, str(source["source_file"]))
    expected_hash = str(source.get("sha256", "")).lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise DatasetError("source manifest sha256 must be a 64-character hexadecimal digest")
    digest = hashlib.sha256()
    try:
        with source_file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DatasetError(f"cannot read source ROOT file {source_file}: {exc}") from exc
    if digest.hexdigest() != expected_hash:
        raise DatasetError(f"source ROOT file SHA-256 mismatch: {source_file}")
    tree_name = str(source["tree"])
    handle, tree = _open_tree(ROOT, source_file, tree_name)
    fields = ("w", "t", "x", "y", "z", "pdgID", "E", "px", "py", "pz")
    context = f"source {tree_name}"
    _require_fields(tree, fields, {"w": {"Float_t", "float"}, "t": {"Double_t", "double"},
                                   "x": {"Float_t", "float"}, "y": {"Float_t", "float"},
                                   "z": {"Float_t", "float"}}, context)
    _require_class_branches(tree, {
        "pdgID": {"vector<int>", "std::vector<int>"},
        "E": {"vector<float>", "std::vector<float>"},
        "px": {"vector<float>", "std::vector<float>"},
        "py": {"vector<float>", "std::vector<float>"},
        "pz": {"vector<float>", "std::vector<float>"},
    }, context)
    if int(tree.GetEntries()) != n_events:
        handle.Close()
        raise DatasetError(f"source tree has {tree.GetEntries()} rows, manifest declares {n_events}")
    return handle, tree


def _read_source_row(tree: Any, index: int) -> dict[str, Any]:
    fields = ("w", "t", "x", "y", "z", "pdgID", "E", "px", "py", "pz")
    if tree.GetEntry(index) <= 0:
        raise DatasetError(f"failed reading source entry {index}")
    row = {name: _value(tree, name) for name in fields}
    for name in ("pdgID", "E", "px", "py", "pz"):
        if not row[name] or len(row[name]) != 1:
            raise DatasetError(f"source entry {index} must have exactly one particle in {name}")
    return row


def _simulation_files(sim: Mapping[str, Any], sim_base: Path) -> list[Path]:
    raw = sim.get("output_files")
    if not isinstance(raw, list) or not raw:
        raise DatasetError("simulation manifest output_files must be a non-empty list")
    files = []
    for item in raw:
        name = item if isinstance(item, str) else item.get("path", item.get("file")) if isinstance(item, Mapping) else None
        if not name:
            raise DatasetError(f"invalid output_files item: {item!r}")
        files.append(_resolve(sim_base, str(name)))
    if len(set(files)) != len(files):
        raise DatasetError("simulation manifest lists a ROOT shard more than once")
    return files


def _verify_sim_source_link(sim: Mapping[str, Any], sim_base: Path,
                           source: Mapping[str, Any], source_base: Path,
                           source_manifest_path: Path | None) -> None:
    linked = sim.get("source_manifest")
    if linked is None:
        raise DatasetError("simulation manifest is missing source_manifest linkage")
    if isinstance(linked, Mapping):
        linked_source = dict(linked)
        linked_base = sim_base
        linked_manifest_path = None
    else:
        linked_path = _resolve(sim_base, str(linked))
        linked_source, linked_base = _manifest(linked_path)
        linked_manifest_path = linked_path
    _check_manifest(linked_source, "linked source")
    if str(linked_source.get("sha256", "")).lower() != str(source.get("sha256", "")).lower():
        raise DatasetError("simulation manifest links a different source SHA-256")
    if str(sim.get("source_sha256", "")).lower() != str(source.get("sha256", "")).lower():
        raise DatasetError("simulation manifest source_sha256 does not match source manifest")
    linked_file = _resolve(linked_base, str(linked_source.get("source_file", "")))
    source_file = _resolve(source_base, str(source.get("source_file", "")))
    if linked_file != source_file:
        raise DatasetError("simulation manifest links a different source ROOT file")
    for name in ("events", "tree", "particle", "geometry_sha256"):
        if name not in sim or name not in source or sim.get(name) != source.get(name):
            raise DatasetError(f"simulation manifest {name} does not match source manifest")
    geometry_hash = str(source.get("geometry_sha256", "")).lower()
    if len(geometry_hash) != 64 or any(char not in "0123456789abcdef" for char in geometry_hash):
        raise DatasetError("source manifest geometry_sha256 must be a 64-character hexadecimal digest")

    manifest_digest = str(sim.get("source_manifest_sha256", "")).lower()
    if len(manifest_digest) != 64 or any(char not in "0123456789abcdef" for char in manifest_digest):
        raise DatasetError("simulation manifest source_manifest_sha256 must be a 64-character hexadecimal digest")
    manifest_paths_to_hash = {path for path in (source_manifest_path, linked_manifest_path) if path is not None}
    for manifest_path_to_hash in manifest_paths_to_hash:
        digest = hashlib.sha256()
        try:
            with manifest_path_to_hash.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise DatasetError(f"cannot read source manifest {manifest_path_to_hash}: {exc}") from exc
        if digest.hexdigest() != manifest_digest:
            raise DatasetError(f"source manifest SHA-256 mismatch: {manifest_path_to_hash}")


def _primary_record(tree: Any) -> tuple[int, dict[str, Any]]:
    evt = int(_value(tree, "EvtID"))
    vertex = {
        "PDGID": int(_value(tree, "PDGID")),
        "t0": float(_value(tree, "t0")),
        "x0": _vec3(_value(tree, "x0"), "SimPrimaryVertex.x0"),
        "Ek0": float(_value(tree, "Ek0")),
        "p0": _vec3(_value(tree, "p0"), "SimPrimaryVertex.p0"),
    }
    return evt, vertex


class _HitCursor:
    """One-event lookahead over hit groups in the rank's event execution order."""

    def __init__(self, tree: Any, detector: str, context: str):
        self.tree = tree
        self.detector = detector
        self.context = context
        self.entry = 0
        self.evt: int | None = None
        self.hit: dict[str, Any] | None = None
        self._advance()

    def _advance(self) -> None:
        if self.entry >= int(self.tree.GetEntries()):
            self.evt = None
            self.hit = None
            return
        index = self.entry
        self.entry += 1
        if self.tree.GetEntry(index) <= 0:
            raise DatasetError(f"failed reading {self.context} entry {index}")
        evt = int(_value(self.tree, "EvtID"))
        self.evt = evt
        self.hit = {field: _value(self.tree, field) for field in FEATURES[self.detector]}

    def take(self, evt: int) -> list[dict[str, Any]]:
        result = []
        while self.evt == evt:
            result.append(self.hit or {})
            self._advance()
        return result

    def ensure_exhausted(self) -> None:
        if self.evt is not None:
            raise DatasetError(f"{self.context} has unconsumed hits for EvtID {self.evt}")


def iter_events(source_manifest: str | Path | Mapping[str, Any],
                sim_manifest_or_paths: str | Path | Mapping[str, Any] | Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield joined events in shard order and primary-tree entry order.

    Each record has ``key=(source_id, run_id, evt_id)``, ``labels`` containing
    only primary-vertex truth, and ``hits`` containing only whitelisted raw
    observables. A hitless event is yielded with empty detector lists. Memory
    use is bounded by a coverage bitmap and the current event's hits; hit rows
    must follow that rank's primary-tree execution order, with each event's rows
    contiguous. MPI event IDs need not be numerically ordered.
    """
    source, source_base = _manifest(source_manifest)
    source_manifest_path = None if isinstance(source_manifest, Mapping) else Path(source_manifest).expanduser().resolve()
    _check_manifest(source, "source")
    for key in ("particle", "events", "tree", "source_file", "sha256"):
        if key not in source:
            raise DatasetError(f"source manifest missing {key!r}")
    n_events = int(source["events"])
    if n_events < 1:
        raise DatasetError("source manifest events must be positive")

    if isinstance(sim_manifest_or_paths, (str, Path, Mapping)):
        sim, sim_base = _manifest(sim_manifest_or_paths)  # type: ignore[arg-type]
        _check_manifest(sim, "simulation")
        if sim.get("completed") is not True:
            raise DatasetError("simulation manifest must be marked completed")
        _verify_sim_source_link(sim, sim_base, source, source_base, source_manifest_path)
        files = _simulation_files(sim, sim_base)
    else:
        sim_base = Path.cwd()
        files = [_resolve(sim_base, str(p)) for p in sim_manifest_or_paths]
        if not files:
            raise DatasetError("at least one simulation ROOT shard is required")

    particle_name = str(source["particle"])
    if particle_name not in _PARTICLE_PDG:
        raise DatasetError(f"unsupported source particle {particle_name!r}; v1 supports gamma and e+")
    ROOT = _root()
    runs: set[int] = set()
    detector_tree = {"TTC": "TTCSimHit", "SciFi": "SciFiSimHit", "ECAL": "ECALSimHit"}
    seen = bytearray(n_events)
    seen_count = 0
    source_handle, source_tree = _source_tree(source, source_base, n_events, ROOT)

    try:
        for file_path in files:
            file_run_ids = _run_ids(ROOT, file_path)
            if not file_run_ids:
                raise DatasetError(f"simulation shard {file_path} has no G4Run directory")
            for run_id in file_run_ids:
                runs.add(run_id)
                prefix = f"G4Run{run_id}"
                h, tree = _open_tree(ROOT, file_path, f"{prefix}/SimPrimaryVertex")
                try:
                    _require_fields(tree, ("EvtID", *LABEL_FIELDS), _PRIMARY_TYPES,
                                    f"{file_path}:{prefix}/SimPrimaryVertex")
                    for entry in range(int(tree.GetEntries())):
                        if tree.GetEntry(entry) <= 0:
                            raise DatasetError(f"failed reading {file_path}:{prefix}/SimPrimaryVertex entry {entry}")
                        evt, vertex = _primary_record(tree)
                        if evt < 0 or evt >= n_events:
                            raise DatasetError(f"unexpected EvtID {evt} in {file_path}:{prefix}")
                        if seen[evt]:
                            raise DatasetError(f"duplicate primary event (G4Run{run_id}, EvtID {evt}) across shards")
                        seen[evt] = 1
                        seen_count += 1
                        row_index = (evt + 1) % n_events
                        source_row = _read_source_row(source_tree, row_index)
                        _validate_source_row(source_row, vertex, particle_name,
                                             f"G4Run{run_id}/EvtID{evt} (source row {row_index})")
                finally:
                    h.Close()
    finally:
        source_handle.Close()

    if len(runs) != 1:
        raise DatasetError(f"expected exactly one G4Run for one source invocation; found {sorted(runs)}")
    run_id = next(iter(runs))
    if seen_count != n_events:
        first_missing = []
        for evt, present in enumerate(seen):
            if not present:
                first_missing.append(evt)
                if len(first_missing) == 8:
                    break
        raise DatasetError(f"primary event coverage mismatch: missing={first_missing}")

    source_id = str(source["sha256"]).lower()
    for file_path in files:
        for shard_run in _run_ids(ROOT, file_path):
            if shard_run != run_id:
                raise DatasetError(f"unexpected G4Run{shard_run} in {file_path}")
            prefix = f"G4Run{shard_run}"
            handle = ROOT.TFile.Open(str(file_path), "READ")
            if not handle or handle.IsZombie():
                raise DatasetError(f"cannot open ROOT file {file_path}")
            try:
                primary_tree = handle.Get(f"{prefix}/SimPrimaryVertex")
                if not primary_tree:
                    raise DatasetError(f"missing TTree {prefix}/SimPrimaryVertex in {file_path}")
                _require_fields(primary_tree, ("EvtID", *LABEL_FIELDS), _PRIMARY_TYPES,
                                f"{file_path}:{prefix}/SimPrimaryVertex")
                cursors = {}
                for detector, tree_name in detector_tree.items():
                    tree = handle.Get(f"{prefix}/{tree_name}")
                    if not tree:
                        raise DatasetError(f"missing TTree {prefix}/{tree_name} in {file_path}")
                    _require_fields(tree, ("EvtID", *FEATURES[detector]), _HIT_TYPES,
                                    f"{file_path}:{prefix}/{tree_name}")
                    cursors[detector] = _HitCursor(tree, detector, f"{file_path}:{prefix}/{tree_name}")

                for entry in range(int(primary_tree.GetEntries())):
                    if primary_tree.GetEntry(entry) <= 0:
                        raise DatasetError(f"failed reading {file_path}:{prefix}/SimPrimaryVertex entry {entry}")
                    evt, vertex = _primary_record(primary_tree)
                    if evt < 0 or evt >= n_events:
                        raise DatasetError(f"unexpected EvtID {evt} in {file_path}:{prefix}")
                    event_hits = {detector: cursors[detector].take(evt) for detector in FEATURES}
                    yield {"key": (source_id, run_id, evt), "labels": vertex, "hits": event_hits}
                for cursor in cursors.values():
                    cursor.ensure_exhausted()
            finally:
                handle.Close()


def _run_ids(ROOT: Any, path: Path) -> list[int]:
    handle = ROOT.TFile.Open(str(path), "READ")
    if not handle or handle.IsZombie():
        raise DatasetError(f"cannot open ROOT file {path}")
    ids: set[int] = set()
    keys = handle.GetListOfKeys()
    if keys:
        for i in range(keys.GetEntries()):
            name = str(keys.At(i).GetName())
            if name.startswith("G4Run"):
                suffix = name[5:]
                if suffix.isdigit():
                    ids.add(int(suffix))
    handle.Close()
    return sorted(ids)
