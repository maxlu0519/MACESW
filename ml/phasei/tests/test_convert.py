from __future__ import annotations

import json
import hashlib
from array import array

import numpy as np
import pytest

from phasei_ml import convert, dataset


def test_primary_coverage_accepts_shuffled_ids_and_returns_order():
    ids = np.asarray([2, 0, 1])
    order = convert._validate_event_coverage(ids, 3, "test")
    assert ids[order].tolist() == [0, 1, 2]


@pytest.mark.parametrize("ids, message", [
    ([0, 1, 1], "duplicate"),
    ([0, 2], "missing"),
    ([0, 3], "out-of-range"),
])
def test_primary_coverage_rejects_duplicate_missing_and_out_of_range(ids, message):
    with pytest.raises(convert.ConvertError, match=message):
        convert._validate_event_coverage(np.asarray(ids), 3, "test")


def test_split_is_deterministic_and_has_all_three_parts():
    strata = np.repeat(np.arange(16, dtype=np.uint8), 100)
    first = convert._stratified_split(strata, 42)
    second = convert._stratified_split(strata, 42)
    assert np.array_equal(first, second)
    for label in range(16):
        counts = np.bincount(first[strata == label], minlength=3)
        assert counts.tolist() == [80, 10, 10]


def test_failed_conversion_does_not_publish_output(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    output = tmp_path / "published"
    with pytest.raises(dataset.DatasetError):
        convert.convert([(bad, bad)], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".published.tmp-*"))


def test_feature_whitelist_has_no_label_fields():
    allowed = {field for fields in dataset.FEATURES.values() for field in fields}
    assert allowed == {"TileID", "FiberID", "ModID", "t", "Edep"}
    assert not (allowed & set(dataset.LABEL_FIELDS))


def test_zero_hit_events_are_represented_by_equal_offsets():
    hit_evt = np.asarray([0, 0, 3])
    offsets = np.zeros(5, dtype=np.int64)
    np.cumsum(np.bincount(hit_evt, minlength=4), out=offsets[1:])
    assert offsets.tolist() == [0, 2, 2, 2, 3]


def _write_small_root_case(tmp_path, *, shard_events=((1, 0),), source_x=(20., 10.), illegal_branch=None):
    ROOT = pytest.importorskip("ROOT")
    source_root = tmp_path / "source.root"
    source_tree = ROOT.TTree("Source", "Source")
    sf = ROOT.TFile(str(source_root), "RECREATE")
    source_tree.SetDirectory(sf)
    w, t = array("f", [1.]), array("d", [0.])
    x, y, z = array("f", [0.]), array("f", [0.]), array("f", [0.])
    pdgid = ROOT.std.vector("int")(); energy = ROOT.std.vector("float")()
    px = ROOT.std.vector("float")(); py = ROOT.std.vector("float")(); pz = ROOT.std.vector("float")()
    for name, value, leaf in (("w", w, "w/F"), ("t", t, "t/D"), ("x", x, "x/F"),
                              ("y", y, "y/F"), ("z", z, "z/F")):
        source_tree.Branch(name, value, leaf)
    for name, value in (("pdgID", pdgid), ("E", energy), ("px", px), ("py", py), ("pz", pz)):
        source_tree.Branch(name, value)
    # Mustard maps EvtID to the following source row, wrapping at N.
    for xpos in source_x:
        x[0] = xpos
        pdgid.clear(); energy.clear(); px.clear(); py.clear(); pz.clear()
        pdgid.push_back(22); energy.push_back(5.0); px.push_back(0.0); py.push_back(0.0); pz.push_back(5.0)
        source_tree.Fill()
    sf.Write(); sf.Close()
    source_sha = hashlib.sha256(source_root.read_bytes()).hexdigest()
    geom_sha = "a" * 64
    source_manifest = {"schema_version": 1, "particle": "gamma", "events": 2, "tree": "Source",
                       "source_file": source_root.name, "sha256": source_sha, "geometry_sha256": geom_sha}
    source_path = tmp_path / "source.root.json"
    source_path.write_text(json.dumps(source_manifest))
    source_manifest_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

    sim_dir = tmp_path / "sim"; sim_dir.mkdir()
    shard_names = []
    hit_rows = {"TTCSimHit": [(1, 7, 2.5, .25)],
                "SciFiSimHit": [(0, 4, 1.5, .125)],
                "ECALSimHit": [(0, 9, 3., .5), (1, 10, 4., .75)]}
    id_names = {"TTCSimHit": "TileID", "SciFiSimHit": "FiberID", "ECALSimHit": "ModID"}
    for shard_index, events in enumerate(shard_events):
        rank = sim_dir / f"rank{shard_index}.root"
        shard_names.append(rank.name)
        rf = ROOT.TFile(str(rank), "RECREATE")
        run = rf.mkdir("G4Run0"); run.cd()
        primary = ROOT.TTree("SimPrimaryVertex", "SimPrimaryVertex")
        evt, primary_pdg = array("i", [0]), array("i", [22])
        t0, ek = array("d", [0.]), array("f", [5.])
        x0, p0 = array("f", [0., 0., 0.]), array("f", [0., 0., 5.])
        primary.Branch("EvtID", evt, "EvtID/I"); primary.Branch("PDGID", primary_pdg, "PDGID/I")
        primary.Branch("t0", t0, "t0/D"); primary.Branch("Ek0", ek, "Ek0/F")
        primary.Branch("x0", x0, "x0[3]/F"); primary.Branch("p0", p0, "p0[3]/F")
        for ev in events:
            evt[0] = ev; x0[0] = (10. if ev == 0 else 20.); primary.Fill()
        primary.Write()
        for tree_name, id_name in id_names.items():
            tree = ROOT.TTree(tree_name, tree_name)
            hevt = array("i", [0])
            ht, edep = array("d", [0.]), array("f", [0.])
            tree.Branch("EvtID", hevt, "EvtID/I"); tree.Branch("t", ht, "t/D")
            tree.Branch("Edep", edep, "Edep/F")
            if tree_name == illegal_branch:
                channel = array("i", [0])
                tree.Branch(id_name, channel, f"{id_name}/I")
            else:
                channel = array("h", [0])
                tree.Branch(id_name, channel, f"{id_name}/S")
            ordered_hits = [hit for ev in events for hit in hit_rows[tree_name] if hit[0] == ev]
            for hit_ev, cid, tm, en in ordered_hits:
                hevt[0], channel[0], ht[0], edep[0] = hit_ev, cid, tm, en
                tree.Fill()
            tree.Write()
        rf.Close()
    sim_manifest = {"schema_version": 1, "completed": True, "events": 2, "particle": "gamma",
                    "tree": "Source", "geometry_sha256": geom_sha, "geometry_hash": geom_sha,
                    "source_sha256": source_sha, "source_manifest": str(source_path),
                    "source_manifest_sha256": source_manifest_sha, "output_files": shard_names}
    sim_path = sim_dir / "simulation_manifest.json"
    sim_path.write_text(json.dumps(sim_manifest))
    return source_path, sim_path


def _load_converted_events(output, particle="gamma"):
    directory = output / particle
    n = len(np.load(directory / "evt_id.npy", mmap_mode="r"))
    result = []
    for evt in range(n):
        entry = {name: np.load(directory / f"{name}.npy", mmap_mode="r")[evt]
                 for name in ("pdgid", "t0", "Ek0", "x0", "p0")}
        entry["hits"] = {}
        for detector, prefix, channel_name in (("TTC", "ttc", "TileID"),
                                                ("SciFi", "scifi", "FiberID"),
                                                ("ECAL", "ecal", "ModID")):
            offsets = np.load(directory / f"{prefix}_offsets.npy", mmap_mode="r")
            entry["hits"][detector] = [{channel_name: int(channel), "t": float(hit_time), "Edep": float(edep)}
                                         for channel, hit_time, edep in zip(
                                             np.load(directory / f"{prefix}_channel.npy", mmap_mode="r")[offsets[evt]:offsets[evt + 1]],
                                             np.load(directory / f"{prefix}_t.npy", mmap_mode="r")[offsets[evt]:offsets[evt + 1]],
                                             np.load(directory / f"{prefix}_Edep.npy", mmap_mode="r")[offsets[evt]:offsets[evt + 1]])]
        result.append(entry)
    return result


def test_small_root_conversion_matches_iter_events_all_detectors_and_labels(tmp_path):
    source_path, sim_path = _write_small_root_case(tmp_path)
    output = tmp_path / "converted"
    convert.convert([(source_path, sim_path)], output)
    expected = sorted(dataset.iter_events(source_path, sim_path), key=lambda event: event["key"][2])
    actual = _load_converted_events(output)
    assert len(actual) == len(expected) == 2
    for got, want in zip(actual, expected):
        for field in ("PDGID", "t0", "Ek0", "x0", "p0"):
            key = "pdgid" if field == "PDGID" else field
            np.testing.assert_allclose(got[key], want["labels"][field])
        assert got["hits"] == want["hits"]


def test_root_converter_rejects_source_row_mapping_mismatch(tmp_path):
    source_path, sim_path = _write_small_root_case(tmp_path, source_x=(20., 11.))
    with pytest.raises(convert.ConvertError, match="source row mapping mismatch"):
        convert.convert([(source_path, sim_path)], tmp_path / "output")


def test_root_converter_rejects_illegal_detector_branch_type(tmp_path):
    source_path, sim_path = _write_small_root_case(tmp_path, illegal_branch="TTCSimHit")
    with pytest.raises(dataset.DatasetError, match="ROOT type"):
        convert.convert([(source_path, sim_path)], tmp_path / "output")


def test_root_converter_rejects_cross_shard_duplicate_event(tmp_path):
    source_path, sim_path = _write_small_root_case(tmp_path, shard_events=((0,), (0, 1)))
    with pytest.raises(convert.ConvertError, match="duplicate.*EvtID"):
        convert.convert([(source_path, sim_path)], tmp_path / "output")


def test_root_converter_rejects_missing_event_across_shards(tmp_path):
    source_path, sim_path = _write_small_root_case(tmp_path, shard_events=((0,),))
    with pytest.raises(convert.ConvertError, match="missing.*EvtID"):
        convert.convert([(source_path, sim_path)], tmp_path / "output")
