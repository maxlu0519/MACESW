import hashlib
import json

import numpy as np
import pytest

from phasei_ml.gnn_data import GraphDataError, GraphEventDataset, fit_normalization, load_graph_data, split_indices
from phasei_ml.geometry import EXPECTED_COUNTS, GeometryError, adjacency, load_geometry


def _geometry(tmp_path):
    yaml = tmp_path / "geom.yaml"
    yaml.write_text("Phase-I test geometry\n", encoding="utf-8")
    detectors = {}
    for detector, count in EXPECTED_COUNTS.items():
        channels = []
        for i in range(count):
            if detector == "scifi":
                pass
            elif detector == "ttc":
                channels.append({"id": i, "position_mm": [float(i), 0., 0.], "normal": [0., 1., 0.]})
            else:
                channels.append({"id": i, "position_mm": [float(i), 0., 0.], "normal": [1., 0., 0.],
                                 "neighbors": [i - 1] if i else [1]})
        detectors[detector] = {"channels": channels}
    fiber_counts = [140, 140, 120, 120, 160, 160, 120, 120,
                    180, 180, 140, 140, 180, 180, 140, 140]
    scifi_channels = []
    channel_id = 0
    for layer, count in enumerate(fiber_counts):
        for local in range(count):
            phi = (local + (0.5 if layer % 2 else 0.)) / count * 2 * np.pi
            scifi_channels.append({"id": channel_id, "layer_id": layer, "local_id": local,
                                   "layer_type": "Transverse", "radius_mm": 45. + layer,
                                   "pitch_rad": 0., "rotation_angle_rad": phi})
            channel_id += 1
    detectors["scifi"] = {"channels": scifi_channels}
    data = {"schema_version": 1, "units": "mm", "coordinate_frame": "MACE world", "detectors": detectors}
    geom_path = tmp_path / "geometry.json"
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"
    geom_path.write_text(payload, encoding="utf-8")
    manifest = {"schema_version": 1, "geometry_yaml": str(yaml),
                "geometry_yaml_sha256": hashlib.sha256(yaml.read_bytes()).hexdigest(),
                "geometry_export_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "counts": EXPECTED_COUNTS}
    manifest_path = tmp_path / "geometry_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return geom_path, manifest_path


def _dataset(tmp_path, geom_yaml_hash):
    root = tmp_path / "data"
    event_dir = root / "e+"
    event_dir.mkdir(parents=True)
    for name, values in {
        "pid": [1, 1], "Ek0": [2., 3.], "x0": [[0., 1., 2.], [1., 2., 3.]],
        "response_mask": [2, 2],
        "p0": [[1., 0., 0.], [0., 1., 0.]], "split": [0, 2],
    }.items():
        np.save(event_dir / f"{name}.npy", np.asarray(values))
    np.save(event_dir / "scifi_channel.npy", np.asarray([0, 0, 1, 0, 1], np.int16))
    np.save(event_dir / "scifi_t.npy", np.asarray([3., 1., 2., 6., 7.], np.float32))
    np.save(event_dir / "scifi_Edep.npy", np.asarray([.1, .2, .3, .4, .5], np.float32))
    np.save(event_dir / "scifi_offsets.npy", np.asarray([0, 3, 5], np.int64))
    for detector in ("ttc", "ecal"):
        np.save(event_dir / f"{detector}_channel.npy", np.asarray([], np.int16))
        np.save(event_dir / f"{detector}_t.npy", np.asarray([], np.float32))
        np.save(event_dir / f"{detector}_Edep.npy", np.asarray([], np.float32))
        np.save(event_dir / f"{detector}_offsets.npy", np.asarray([0, 0, 0], np.int64))
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "particles": [
        {"particle": "e+", "geometry_sha256": geom_yaml_hash}]}), encoding="utf-8")
    return root


def test_geometry_hash_and_graph_edges_are_bound_and_event_local(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, manifest = load_geometry(geom_path, manifest_path)
    data_dir = _dataset(tmp_path, manifest["geometry_yaml_sha256"])
    particles = load_graph_data(data_dir, geometry, manifest)
    packed = particles[0].detector["scifi"]
    assert packed.channel.tolist() == [0, 1, 0, 1]
    assert packed.edep.tolist() == pytest.approx([.3, .3, .4, .5])
    assert packed.relative_time.tolist() == pytest.approx([0., 1., 0., 1.])
    norm, target_norm = fit_normalization(particles)
    indices = split_indices(particles)["train"] + split_indices(particles)["test"]
    adjacency_map = {name: adjacency(geometry, name) for name in EXPECTED_COUNTS}
    dataset = GraphEventDataset(particles, indices, norm, target_norm, adjacency_map)
    graphs, _ = dataset.collate([dataset[0], dataset[1]])
    graph = graphs["scifi"]
    assert graph["counts"].tolist() == [2., 2.]
    assert graph["edge_index"].shape[1] == 4
    assert graph["edge_attr"].shape == (4, 9)
    assert torch_equal(graph["batch"][graph["edge_index"][0]],
                       graph["batch"][graph["edge_index"][1]])


def torch_equal(a, b):
    return bool((a == b).all())


def test_geometry_yaml_mismatch_is_rejected(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, manifest = load_geometry(geom_path, manifest_path)
    data_dir = _dataset(tmp_path, "0" * 64)
    with pytest.raises(GraphDataError, match="geometry hash"):
        load_graph_data(data_dir, geometry, manifest)


def test_geometry_manifest_resolves_yaml_relative_to_itself(tmp_path, monkeypatch):
    geom_path, manifest_path = _geometry(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["geometry_yaml"] = "geom.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    _, loaded = load_geometry(geom_path, manifest_path)
    assert loaded["geometry_yaml_sha256"] == manifest["geometry_yaml_sha256"]


def test_ecal_crystal_normal_is_required(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    payload = json.loads(geom_path.read_text(encoding="utf-8"))
    del payload["detectors"]["ecal"]["channels"][0]["normal"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    geom_path.write_text(encoded, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["geometry_export_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GeometryError, match="ECAL channel 0 needs center and normal"):
        load_geometry(geom_path, manifest_path)


def test_train_validation_load_does_not_inspect_test_hit_values(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, manifest = load_geometry(geom_path, manifest_path)
    data_dir = _dataset(tmp_path, manifest["geometry_yaml_sha256"])
    channel_path = data_dir / "e+" / "scifi_channel.npy"
    channel = np.load(channel_path)
    channel[3] = 9999  # deliberately invalid, but belongs to split==2
    np.save(channel_path, channel)
    training_graphs = load_graph_data(data_dir, geometry, manifest, allowed_split_ids=(0, 1))
    assert training_graphs[0].detector["scifi"].channel.tolist() == [0, 1]
    with pytest.raises(GraphDataError, match="outside geometry"):
        load_graph_data(data_dir, geometry, manifest)


def test_scifi_adjacency_uses_local_and_nearby_layers_only(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, _ = load_geometry(geom_path, manifest_path)
    rows = geometry["detectors"]["scifi"]["channels"]
    graph = adjacency(geometry, "scifi")
    layer_sizes = {}
    for row in rows:
        layer_sizes[row["layer_id"]] = max(layer_sizes.get(row["layer_id"], 0), row["local_id"] + 1)
    for source, neighbors in enumerate(graph):
        a = rows[source]
        for target in neighbors:
            b = rows[target]
            delta_layer = abs(a["layer_id"] - b["layer_id"])
            assert delta_layer <= 4
            if delta_layer == 0:
                delta_local = abs(a["local_id"] - b["local_id"])
                delta_local = min(delta_local, layer_sizes[a["layer_id"]] - delta_local)
                assert delta_local <= 2


def test_hit_order_does_not_change_channel_aggregation(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, manifest = load_geometry(geom_path, manifest_path)
    data_dir = _dataset(tmp_path, manifest["geometry_yaml_sha256"])
    before = load_graph_data(data_dir, geometry, manifest)[0].detector["scifi"]
    event_dir = data_dir / "e+"
    for suffix in ("channel", "t", "Edep"):
        path = event_dir / f"scifi_{suffix}.npy"
        rows = np.load(path)
        rows[:3] = rows[:3][[2, 0, 1]]
        np.save(path, rows)
    after = load_graph_data(data_dir, geometry, manifest)[0].detector["scifi"]
    np.testing.assert_array_equal(before.channel, after.channel)
    np.testing.assert_allclose(before.edep, after.edep)
    np.testing.assert_allclose(before.relative_time, after.relative_time)


def test_zero_hit_event_is_preserved_in_graph_batch(tmp_path):
    geom_path, manifest_path = _geometry(tmp_path)
    geometry, manifest = load_geometry(geom_path, manifest_path)
    data_dir = _dataset(tmp_path, manifest["geometry_yaml_sha256"])
    event_dir = data_dir / "e+"
    for suffix in ("channel", "t", "Edep"):
        path = event_dir / f"scifi_{suffix}.npy"
        np.save(path, np.load(path)[:3])
    np.save(event_dir / "scifi_offsets.npy", np.asarray([0, 3, 3], np.int64))
    np.save(event_dir / "response_mask.npy", np.asarray([2, 0], np.int8))
    particles = load_graph_data(data_dir, geometry, manifest)
    norm, target_norm = fit_normalization(particles)
    graph_data = GraphEventDataset(particles, split_indices(particles)["test"], norm, target_norm,
                                   {name: adjacency(geometry, name) for name in EXPECTED_COUNTS})
    graphs, target = graph_data.collate([graph_data[0]])
    assert int(target["response_mask"][0]) == 0
    for detector in EXPECTED_COUNTS:
        assert graphs[detector]["counts"].tolist() == [0.]
        assert graphs[detector]["edge_index"].shape == (2, 0)
