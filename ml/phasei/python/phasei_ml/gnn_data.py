"""Sparse event graph construction for the Phase-I detector GNN.

Hit rows are aggregated by (event, channel): deposited energy is summed and
the earliest hit time is retained. No event is capped or truncated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import EXPECTED_COUNTS


DETECTORS = ("scifi", "ttc", "ecal")
_NAMES = {"scifi": "SciFi", "ttc": "TTC", "ecal": "ECAL"}


class GraphDataError(ValueError):
    pass


def _geometry_features(geometry: dict, detector: str) -> np.ndarray:
    channels = geometry["detectors"][detector]["channels"]
    if detector == "scifi":
        rows = []
        nlayer = max(int(c["layer_id"]) for c in channels) + 1
        nlocal = {}
        for c in channels:
            nlocal[int(c["layer_id"])] = max(nlocal.get(int(c["layer_id"]), 0), int(c["local_id"]) + 1)
        for c in channels:
            rows.append((int(c["layer_id"]) / max(nlayer - 1, 1),
                         int(c["local_id"]) / max(nlocal[int(c["layer_id"])] - 1, 1),
                         float(c["radius_mm"]), float(c["pitch_rad"]),
                         np.sin(float(c["rotation_angle_rad"])), np.cos(float(c["rotation_angle_rad"])),
                         float(c["layer_type"] != "Transverse")))
    elif detector == "ttc":
        rows = [tuple(c["position_mm"]) + tuple(c["normal"]) for c in channels]
    else:
        rows = [tuple(c["position_mm"]) + tuple(c["normal"]) + (len(c["neighbors"]),) for c in channels]
    return np.asarray(rows, dtype=np.float32)


@dataclass
class PackedDetector:
    channel: np.ndarray
    relative_time: np.ndarray
    edep: np.ndarray
    offsets: np.ndarray
    static: np.ndarray


@dataclass
class ParticleGraphs:
    particle: str
    detector: dict[str, PackedDetector]
    labels: dict[str, np.ndarray]
    split: np.ndarray


def _pack_detector(directory: Path, detector: str, n_events: int, event_min: np.ndarray,
                   static: np.ndarray, allowed_events: np.ndarray) -> PackedDetector:
    channel_raw = np.load(directory / f"{detector}_channel.npy", mmap_mode="r")
    times_raw = np.load(directory / f"{detector}_t.npy", mmap_mode="r")
    edep_raw = np.load(directory / f"{detector}_Edep.npy", mmap_mode="r")
    offsets = np.asarray(np.load(directory / f"{detector}_offsets.npy", mmap_mode="r"), dtype=np.int64)
    counts = np.diff(offsets)
    all_event_ids = np.repeat(np.arange(n_events, dtype=np.int64), counts)
    keep = allowed_events[all_event_ids]
    event_ids = all_event_ids[keep]
    channel = np.asarray(channel_raw[keep], dtype=np.int64)
    times = np.asarray(times_raw[keep], dtype=np.float32)
    edep = np.asarray(edep_raw[keep], dtype=np.float32)
    channel = np.asarray(channel, dtype=np.int64)
    if len(channel) != len(times) or len(channel) != len(edep):
        raise GraphDataError(f"{detector} channel/time/energy lengths disagree in {directory.name}")
    if np.any((channel < 0) | (channel >= EXPECTED_COUNTS[detector])):
        raise GraphDataError(f"{detector} channel outside geometry range in {directory.name}")
    keys = event_ids * EXPECTED_COUNTS[detector] + channel
    unique, inverse = np.unique(keys, return_inverse=True)
    node_energy = np.bincount(inverse, weights=np.maximum(edep, 0.0)).astype(np.float32)
    node_time = np.full(len(unique), np.inf, dtype=np.float32)
    np.minimum.at(node_time, inverse, times)
    node_events = unique // EXPECTED_COUNTS[detector]
    node_channels = (unique % EXPECTED_COUNTS[detector]).astype(np.int16)
    if np.any(~np.isfinite(event_min[node_events])):
        raise GraphDataError("nonempty detector event has no global time reference")
    node_time -= event_min[node_events]
    node_offsets = np.concatenate(([0], np.cumsum(np.bincount(node_events, minlength=n_events), dtype=np.int64)))
    return PackedDetector(node_channels, node_time, node_energy, node_offsets, static)


def load_graph_data(data_dir: str | Path, geometry: dict, geometry_manifest: dict,
                    allowed_split_ids: tuple[int, ...] = (0, 1, 2)) -> list[ParticleGraphs]:
    data_dir = Path(data_dir).expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    geom_hash = geometry_manifest["geometry_yaml_sha256"]
    for item in manifest.get("particles", []):
        if item.get("geometry_sha256") != geom_hash:
            raise GraphDataError(f"{item.get('particle')} dataset geometry hash does not match exported YAML")
    packed = []
    for item in manifest["particles"]:
        particle = item["particle"]
        directory = data_dir / particle
        split = np.asarray(np.load(directory / "split.npy", mmap_mode="r"), dtype=np.int8)
        n_events = len(split)
        allowed_events = np.isin(split, allowed_split_ids)
        event_min = np.full(n_events, np.inf, dtype=np.float32)
        for detector in DETECTORS:
            offsets = np.asarray(np.load(directory / f"{detector}_offsets.npy", mmap_mode="r"), dtype=np.int64)
            all_event_ids = np.repeat(np.arange(n_events, dtype=np.int64), np.diff(offsets))
            keep = allowed_events[all_event_ids]
            times = np.asarray(np.load(directory / f"{detector}_t.npy", mmap_mode="r")[keep], dtype=np.float32)
            event_ids = all_event_ids[keep]
            np.minimum.at(event_min, event_ids, times)
        detectors = {detector: _pack_detector(directory, detector, n_events, event_min,
                                               _geometry_features(geometry, detector), allowed_events)
                     for detector in DETECTORS}
        labels = {name: np.load(directory / f"{name}.npy", mmap_mode="r")
                  for name in ("pid", "Ek0", "x0", "p0", "response_mask")}
        packed.append(ParticleGraphs(particle, detectors, labels, split))
    return packed


def fit_normalization(particles: list[ParticleGraphs]) -> tuple[dict, dict]:
    norm = {}
    energy = []
    position = []
    for particle in particles:
        mask = particle.split == 0
        energy.append(np.asarray(particle.labels["Ek0"][mask], dtype=np.float64))
        position.append(np.asarray(particle.labels["x0"][mask], dtype=np.float64))
    all_energy = np.concatenate(energy)
    all_position = np.concatenate(position)
    target = {"energy_mean": float(all_energy.mean()), "energy_std": float(max(all_energy.std(), 1e-6)),
              "position_mean": all_position.mean(axis=0).astype(float).tolist(),
              "position_std": np.maximum(all_position.std(axis=0), 1e-6).astype(float).tolist()}
    for detector in DETECTORS:
        all_values = []
        for particle in particles:
            packed = particle.detector[detector]
            keep = np.repeat(particle.split == 0, np.diff(packed.offsets))
            edep_log = np.log1p(np.maximum(packed.edep[keep], 0.0))[:, None]
            raw = np.column_stack((packed.relative_time[keep], edep_log[:, 0], packed.static[packed.channel[keep]]))
            all_values.append(raw)
        values = np.concatenate(all_values, axis=0)
        if not len(values):
            width = 2 + _geometry_features_placeholder_width(detector)
            mean = np.zeros(width, dtype=np.float64)
            std = np.ones(width, dtype=np.float64)
        else:
            mean = values.mean(axis=0)
            std = np.maximum(values.std(axis=0), 1e-6)
        norm[detector] = {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}
    return norm, target


def _geometry_features_placeholder_width(detector: str) -> int:
    # A detector absent from all training events has no data-derived feature
    # statistics; zero/one is the neutral standardization for its idle branch.
    return {"scifi": 7, "ttc": 6, "ecal": 7}[detector]


class GraphEventDataset(Dataset):
    def __init__(self, particles: list[ParticleGraphs], indices: list[tuple[int, int]],
                 norm: dict, target_norm: dict, adjacency: dict[str, list[list[int]]]):
        self.particles = particles
        self.indices = indices
        self.norm = norm
        self.target_norm = target_norm
        self.adjacency = adjacency

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]

    def collate(self, rows):
        batch_size = len(rows)
        output = {}
        for detector in DETECTORS:
            node_x, node_channel, node_batch, edges = [], [], [], []
            ptr = [0]
            for batch_id, (particle_id, event_id) in enumerate(rows):
                p = self.particles[particle_id]
                packed = p.detector[detector]
                lo, hi = int(packed.offsets[event_id]), int(packed.offsets[event_id + 1])
                channels = np.asarray(packed.channel[lo:hi], dtype=np.int64)
                time = packed.relative_time[lo:hi]
                energy = np.log1p(np.maximum(packed.edep[lo:hi], 0.0))
                raw = np.column_stack((time, energy, packed.static[channels])).astype(np.float32, copy=False)
                stat = self.norm[detector]
                raw = (raw - np.asarray(stat["mean"], dtype=np.float32)) / np.asarray(stat["std"], dtype=np.float32)
                node_x.append(raw)
                node_channel.append(channels)
                node_batch.append(np.full(len(channels), batch_id, dtype=np.int64))
                local = {int(channel): j for j, channel in enumerate(channels)}
                base = ptr[-1]
                for src, channel in enumerate(channels):
                    for neighbor in self.adjacency[detector][int(channel)]:
                        dst = local.get(int(neighbor))
                        if dst is not None:
                            edges.append((base + src, base + dst))
                ptr.append(base + len(channels))
            x = np.concatenate(node_x, axis=0) if node_x else np.zeros((0, len(self.norm[detector]["mean"])), np.float32)
            channels = np.concatenate(node_channel) if node_channel else np.zeros(0, np.int64)
            batch = np.concatenate(node_batch) if node_batch else np.zeros(0, np.int64)
            edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
            if edge_index.size and np.any(batch[edge_index[0]] != batch[edge_index[1]]):
                raise GraphDataError("cross-event graph edge detected")
            edge_attr = x[edge_index[0]] - x[edge_index[1]] if edge_index.size else np.zeros(
                (0, x.shape[1]), dtype=np.float32)
            output[detector] = {"x": torch.from_numpy(x), "channel": torch.from_numpy(channels),
                                "batch": torch.from_numpy(batch), "ptr": torch.as_tensor(ptr, dtype=torch.int64),
                                "edge_index": torch.from_numpy(edge_index), "edge_attr": torch.from_numpy(edge_attr),
                                "counts": torch.as_tensor(np.diff(ptr), dtype=torch.float32)}

        pid, energy, position, direction, response_mask = [], [], [], [], []
        for particle_id, event_id in rows:
            labels = self.particles[particle_id].labels
            pid.append(int(labels["pid"][event_id]))
            response_mask.append(int(labels["response_mask"][event_id]))
            energy.append(float(labels["Ek0"][event_id]))
            position.append(np.asarray(labels["x0"][event_id], dtype=np.float32))
            momentum = np.asarray(labels["p0"][event_id], dtype=np.float32)
            direction.append(momentum / max(float(np.linalg.norm(momentum)), 1e-12))
        energy = (np.asarray(energy, dtype=np.float32) - self.target_norm["energy_mean"]) / self.target_norm["energy_std"]
        pmean = np.asarray(self.target_norm["position_mean"], dtype=np.float32)
        pstd = np.asarray(self.target_norm["position_std"], dtype=np.float32)
        target = {"pid": torch.as_tensor(pid, dtype=torch.long),
                  "response_mask": torch.as_tensor(response_mask, dtype=torch.long),
                  "energy": torch.from_numpy(energy),
                  "position": torch.from_numpy((np.stack(position) - pmean) / pstd),
                  "direction": torch.from_numpy(np.stack(direction))}
        return output, target


def split_indices(particles: list[ParticleGraphs]) -> dict[str, list[tuple[int, int]]]:
    result = {"train": [], "validation": [], "test": []}
    ids = {0: "train", 1: "validation", 2: "test"}
    for particle_id, particle in enumerate(particles):
        for split_id, name in ids.items():
            result[name].extend((particle_id, int(i)) for i in np.flatnonzero(particle.split == split_id))
    return result
