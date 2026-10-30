"""Validation and adjacency helpers for exported Phase-I geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_COUNTS = {"scifi": 2360, "ttc": 108, "ecal": 488}


class GeometryError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_geometry(geometry_path: str | Path, manifest_path: str | Path | None = None):
    geometry_path = Path(geometry_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else geometry_path.with_name("geometry_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = geometry_path.read_bytes()
        geometry = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometryError(f"cannot read geometry artifact: {exc}") from exc
    if manifest.get("schema_version") != 1 or geometry.get("schema_version") != 1:
        raise GeometryError("geometry schema_version must be 1")
    if hashlib.sha256(payload).hexdigest() != manifest.get("geometry_export_sha256"):
        raise GeometryError("geometry export SHA-256 mismatch")
    yaml_path = Path(manifest.get("geometry_yaml", ""))
    if not yaml_path.is_absolute():
        yaml_path = (manifest_path.parent / yaml_path).resolve()
    if not yaml_path.is_absolute():
        yaml_path = (manifest_path.parent / yaml_path).resolve()
    if not yaml_path.is_file() or sha256_file(yaml_path) != manifest.get("geometry_yaml_sha256"):
        raise GeometryError("geometry YAML missing or SHA-256 mismatch")
    if geometry.get("units") != "mm" or geometry.get("coordinate_frame") != "MACE world":
        raise GeometryError("geometry units/frame do not match the Phase-I schema")
    detectors = geometry.get("detectors", {})
    for detector, expected in EXPECTED_COUNTS.items():
        channels = detectors.get(detector, {}).get("channels", [])
        if len(channels) != expected or manifest.get("counts", {}).get(detector) != expected:
            raise GeometryError(f"{detector} channel count must be {expected}")
        ids = [int(row["id"]) for row in channels]
        if ids != list(range(expected)):
            raise GeometryError(f"{detector} channel IDs must be contiguous and sorted from zero")
        if detector == "ecal":
            id_set = set(ids)
            for channel in channels:
                if len(channel.get("position_mm", [])) != 3 or len(channel.get("normal", [])) != 3:
                    raise GeometryError(f"ECAL channel {channel['id']} needs center and normal")
                neighbors = channel.get("neighbors", [])
                if neighbors != sorted(set(neighbors)) or any(n not in id_set or n == channel["id"] for n in neighbors):
                    raise GeometryError(f"invalid ECAL neighbors for channel {channel['id']}")
    return geometry, manifest


def adjacency(geometry: dict, detector: str, k: int = 12) -> list[list[int]]:
    """Return ECAL source adjacency or deterministic geometric kNN adjacency."""
    channels = geometry["detectors"][detector]["channels"]
    if detector == "ecal":
        return [list(row["neighbors"]) for row in channels]
    if detector == "scifi":
        # Match channel topology and actual FiberInformation fields: near
        # neighbors on each cyclic layer plus angularly close fibers in at most
        # four adjacent layers. No point is invented for a helical fiber.
        layer_channels: dict[int, list[dict]] = {}
        for channel in channels:
            layer_channels.setdefault(int(channel["layer_id"]), []).append(channel)
        for rows in layer_channels.values():
            rows.sort(key=lambda row: int(row["local_id"]))
        result = []
        for channel in channels:
            layer = int(channel["layer_id"])
            local = int(channel["local_id"])
            current_phi = float(channel["rotation_angle_rad"])
            current_radius = float(channel["radius_mm"])
            current_pitch = float(channel["pitch_rad"])
            selected = set()
            own = layer_channels[layer]
            for delta in (-2, -1, 1, 2):
                selected.add(int(own[(local + delta) % len(own)]["id"]))
            candidates: dict[int, list[tuple[float, int]]] = {}
            for other_layer, rows in layer_channels.items():
                if other_layer == layer or abs(other_layer - layer) > 4:
                    continue
                layer_candidates = candidates.setdefault(other_layer, [])
                for other in rows:
                    phi_delta = abs((current_phi - float(other["rotation_angle_rad"]) + np.pi) % (2 * np.pi) - np.pi)
                    radius = (current_radius + float(other["radius_mm"])) / 2
                    pitch_delta = current_pitch - float(other["pitch_rad"])
                    distance2 = ((current_radius - float(other["radius_mm"])) ** 2
                                 + (radius * phi_delta) ** 2
                                 + (radius * pitch_delta) ** 2)
                    layer_candidates.append((distance2, int(other["id"])))
            # Add the two closest fibers from each eligible neighboring layer;
            # sorting by (distance, channel id) makes ties deterministic.
            for other_layer in range(max(0, layer - 4), min(max(layer_channels) + 1, layer + 5)):
                if other_layer == layer or other_layer not in candidates:
                    continue
                layer_candidates = candidates[other_layer]
                layer_candidates.sort()
                selected.update(channel_id for _, channel_id in layer_candidates[:2])
            result.append(sorted(selected))
        return result

    xyz = [row["position_mm"] for row in channels]
    coords = np.asarray(xyz, dtype=np.float64)
    result = []
    for i in range(len(coords)):
        d2 = np.sum((coords - coords[i]) ** 2, axis=1)
        d2[i] = np.inf
        order = np.argsort(d2, kind="stable")[: min(k, len(coords) - 1)]
        result.append(order.astype(int).tolist())
    return result
