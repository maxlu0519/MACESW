"""Synthetic ROOT-shaped tests for the streaming dataset contract."""

from __future__ import annotations

import hashlib
import json
import math
import pytest

from phasei_ml import dataset


class _Branch:
    def __init__(self, name, type_name="Float_t"):
        self.name = name
        self.type_name = type_name

    def GetLeaf(self, _name):
        return self

    def GetTypeName(self):
        return self.type_name


class _Tree:
    def __init__(self, rows, types=None):
        self.rows = rows
        self.types = types or {}
        self.current = None

    def GetBranch(self, name):
        if not self.rows and name not in self.types:
            return None
        if self.rows and name not in self.rows[0] and name not in self.types:
            return None
        return _Branch(name, self.types.get(name, "Float_t"))

    def GetEntries(self):
        return len(self.rows)

    def GetEntry(self, index):
        self.current = self.rows[index]
        for name, value in self.current.items():
            setattr(self, name, value)
        return 1


class _Key:
    def __init__(self, name):
        self.name = name

    def GetName(self):
        return self.name


class _Keys(list):
    def GetEntries(self):
        return len(self)

    def At(self, index):
        return self[index]


class _Handle:
    def __init__(self, trees):
        self.trees = trees

    def IsZombie(self):
        return False

    def Get(self, name):
        return self.trees.get(name)

    def GetListOfKeys(self):
        return _Keys(_Key("G4Run0") for name in self.trees if name.startswith("G4Run0/"))

    def Close(self):
        pass


class _TFile:
    registry = {}

    @classmethod
    def Open(cls, name, _mode):
        return cls.registry.get(name)


class _ROOT:
    TFile = _TFile


def _source_rows():
    # Mustard's SetEntry(i) followed by Next() makes event 0 consume row 1.
    mass = 0.51099895  # e+ mass in MeV, matching the generator's MeV units.
    p_for_kinetic = lambda kinetic: math.sqrt((mass + kinetic) ** 2 - mass**2)
    return [
        {"w": 1.0, "t": 0.0, "x": 20.0, "y": 0.0, "z": 0.0, "pdgID": [-11], "E": [mass + 3.0], "px": [p_for_kinetic(3.0)], "py": [0.0], "pz": [0.0]},
        {"w": 1.0, "t": 0.0, "x": 10.0, "y": 0.0, "z": 0.0, "pdgID": [-11], "E": [mass + 2.0], "px": [p_for_kinetic(2.0)], "py": [0.0], "pz": [0.0]},
    ]


def _tree(rows, types=None):
    return _Tree(rows, types)


def test_iter_events_joins_by_evtid_and_keeps_empty_events(monkeypatch, tmp_path):
    source_file = tmp_path / "source.root"
    source_file.write_bytes(b"synthetic source ROOT payload")
    source_path = str(source_file.resolve())
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    sim_path = str((tmp_path / "rank0.root").resolve())
    source_types = {"w": "Float_t", "t": "Double_t"}
    source_tree = _tree(_source_rows(), source_types)
    primary = _tree([
        {"EvtID": 0, "PDGID": -11, "t0": 0.0, "x0": (10.0, 0.0, 0.0), "Ek0": 2.0,
         "p0": (math.sqrt((0.51099895 + 2.0) ** 2 - 0.51099895**2), 0.0, 0.0)},
        {"EvtID": 1, "PDGID": -11, "t0": 0.0, "x0": (20.0, 0.0, 0.0), "Ek0": 3.0,
         "p0": (math.sqrt((0.51099895 + 3.0) ** 2 - 0.51099895**2), 0.0, 0.0)},
    ], {"EvtID": "Int_t", "PDGID": "Int_t", "t0": "Double_t", "Ek0": "Float_t"})
    trees = {
        "G4Run0/SimPrimaryVertex": primary,
        "G4Run0/TTCSimHit": _tree([{"EvtID": 1, "TileID": 7, "t": 4.5, "Edep": 0.25}], {"EvtID": "Int_t", "TileID": "Short_t", "t": "Double_t", "Edep": "Float_t"}),
        "G4Run0/SciFiSimHit": _tree([], {"EvtID": "Int_t", "FiberID": "Short_t", "t": "Double_t", "Edep": "Float_t"}),
        "G4Run0/ECALSimHit": _tree([], {"EvtID": "Int_t", "ModID": "Short_t", "t": "Double_t", "Edep": "Float_t"}),
    }
    _TFile.registry = {source_path: _Handle({"Source": source_tree}), sim_path: _Handle(trees)}
    monkeypatch.setattr(dataset, "_root", lambda: _ROOT)
    source = {"schema_version": 1, "particle": "e+", "events": 2, "tree": "Source",
              "source_file": source_path, "sha256": source_hash}

    events = list(dataset.iter_events(source, [sim_path]))

    assert [event["key"][2] for event in events] == [0, 1]
    assert events[0]["hits"] == {"TTC": [], "SciFi": [], "ECAL": []}
    assert events[1]["hits"]["TTC"] == [{"TileID": 7, "t": 4.5, "Edep": 0.25}]
    assert set(events[0]["labels"]) == set(dataset.LABEL_FIELDS)
    assert all(set(hit) <= set(dataset.FEATURES[detector])
               for event in events for detector, hits in event["hits"].items() for hit in hits)


def test_source_row_validation_rejects_truth_mismatch():
    row = _source_rows()[1]
    vertex = {"PDGID": -11, "t0": 0.0, "x0": (10.0, 0.0, 0.0), "Ek0": 9.0,
              "p0": (math.sqrt((0.51099895 + 2.0) ** 2 - 0.51099895**2), 0.0, 0.0)}
    with pytest.raises(dataset.DatasetError, match="kinetic energy"):
        dataset._validate_source_row(row, vertex, "e+", "event")


def test_vector_schema_is_checked():
    assert dataset._vec3([1, 2, 3], "p0") == (1.0, 2.0, 3.0)
    with pytest.raises(dataset.DatasetError, match="3-vector"):
        dataset._vec3([1, 2], "p0")


def test_hit_cursor_consumes_contiguous_event_groups():
    tree = _tree([
        {"EvtID": 5, "TileID": 8, "t": 3.0, "Edep": 0.3},
        {"EvtID": 2, "TileID": 3, "t": 1.0, "Edep": 0.1},
        {"EvtID": 2, "TileID": 4, "t": 2.0, "Edep": 0.2},
    ])
    cursor = dataset._HitCursor(tree, "TTC", "synthetic TTC")
    # MPI execution order may move backward numerically within a rank.
    assert cursor.take(5) == [{"TileID": 8, "t": 3.0, "Edep": 0.3}]
    assert cursor.take(2) == [
        {"TileID": 3, "t": 1.0, "Edep": 0.1},
        {"TileID": 4, "t": 2.0, "Edep": 0.2},
    ]
    cursor.ensure_exhausted()


def test_sim_manifest_checks_source_provenance_hashes(tmp_path):
    source_file = tmp_path / "source.root"
    source_file.write_bytes(b"source")
    source_manifest = {
        "schema_version": 1,
        "particle": "gamma",
        "events": 4,
        "tree": "Generated",
        "source_file": "source.root",
        "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "geometry_sha256": "a" * 64,
    }
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sim_manifest = {
        "source_manifest": "source.json",
        "source_manifest_sha256": manifest_hash,
        "source_sha256": source_manifest["sha256"],
        "events": source_manifest["events"],
        "tree": source_manifest["tree"],
        "particle": source_manifest["particle"],
        "geometry_sha256": source_manifest["geometry_sha256"],
    }

    dataset._verify_sim_source_link(sim_manifest, tmp_path, source_manifest, tmp_path, manifest_path)

    sim_manifest["geometry_sha256"] = "b" * 64
    with pytest.raises(dataset.DatasetError, match="geometry_sha256"):
        dataset._verify_sim_source_link(sim_manifest, tmp_path, source_manifest, tmp_path, manifest_path)
