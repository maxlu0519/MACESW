"""Vectorized batches from the event-indexed Phase-I NumPy dataset.

Only the converted detector fields are copied into model inputs. Primary
vertex arrays are read separately and used exclusively as training targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_DETECTORS = ("ttc", "scifi", "ecal")
_NORM_NAMES = {"ttc": "TTC", "scifi": "SciFi", "ecal": "ECAL"}


def identity_collate(batch):
    """DataLoader's batched fetch has already assembled the tensors."""
    return batch


class VectorizedEventDataset(Dataset):
    """Gather a mixed-species batch using NumPy array indexing, not hit loops."""

    def __init__(self, data_dir: str | Path, indices_by_particle: dict[str, np.ndarray],
                 target_scaler: dict, detector_names: tuple[str, ...] = _DETECTORS):
        self.data_dir = Path(data_dir)
        self.target_scaler = target_scaler
        self.detector_names = tuple(detector_names)
        if not self.detector_names or any(name not in _DETECTORS for name in self.detector_names):
            raise ValueError("detector_names must select one or more Phase-I detectors")
        self.particles = []
        lengths = []
        for particle, indices in indices_by_particle.items():
            selected = np.asarray(indices, dtype=np.int64)
            if not len(selected):
                continue
            directory = self.data_dir / particle
            labels = {name: np.load(directory / f"{name}.npy", mmap_mode="r")
                      for name in ("pid", "Ek0", "x0", "p0", "response_mask")}
            hit = {detector: {field: np.load(directory / f"{detector}_{field}.npy", mmap_mode="r")
                              for field in ("channel", "t", "Edep", "offsets")}
                   for detector in self.detector_names}
            norm = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
            self.particles.append((selected, labels, hit, norm))
            lengths.append(len(selected))
        if not lengths:
            raise ValueError("vectorized dataset has no selected events")
        self.bounds = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    def __len__(self):
        return int(self.bounds[-1])

    def __getitem__(self, index):
        # DataLoader uses __getitems__ for automatic batching. Keep this method
        # available for callers that request a one-event batch explicitly.
        return self.__getitems__([index])

    def __getitems__(self, indices):
        global_indices = np.asarray(indices, dtype=np.int64)
        if global_indices.ndim != 1 or not len(global_indices):
            raise ValueError("expected a nonempty batch of event indices")
        if np.any((global_indices < 0) | (global_indices >= len(self))):
            raise IndexError("event index outside selected dataset")
        batch_size = len(global_indices)
        particle_ids = np.searchsorted(self.bounds[1:], global_indices, side="right")
        pid = np.empty(batch_size, dtype=np.int64)
        energy = np.empty(batch_size, dtype=np.float32)
        position = np.empty((batch_size, 3), dtype=np.float32)
        momentum = np.empty((batch_size, 3), dtype=np.float32)
        response_mask = np.empty(batch_size, dtype=np.float32)
        groups = []
        max_hits = {name: 1 for name in self.detector_names}

        for particle_id, (selected, labels, hit, norm) in enumerate(self.particles):
            rows = np.flatnonzero(particle_ids == particle_id)
            if not len(rows):
                continue
            events = selected[global_indices[rows] - self.bounds[particle_id]]
            pid[rows] = labels["pid"][events]
            energy[rows] = labels["Ek0"][events]
            position[rows] = labels["x0"][events]
            momentum[rows] = labels["p0"][events]
            response_mask[rows] = labels["response_mask"][events]
            detector_slices = {}
            for name in self.detector_names:
                offsets = hit[name]["offsets"]
                starts = np.asarray(offsets[events], dtype=np.int64)
                counts = np.asarray(offsets[events + 1] - starts, dtype=np.int64)
                max_hits[name] = max(max_hits[name], int(counts.max(initial=0)))
                detector_slices[name] = (starts, counts)
            groups.append((rows, hit, norm, detector_slices))

        scaler = self.target_scaler
        energy = (energy - scaler["energy"]["mean"]) / scaler["energy"]["std"]
        position = (position - np.asarray(scaler["position"]["mean"], dtype=np.float32)) / np.asarray(
            scaler["position"]["std"], dtype=np.float32)
        direction = momentum / np.maximum(np.linalg.norm(momentum, axis=1, keepdims=True), 1e-12)
        batch = {"response_mask": torch.from_numpy(response_mask)}
        target = {"pid": torch.from_numpy(pid), "energy": torch.from_numpy(energy),
                  "position": torch.from_numpy(position), "direction": torch.from_numpy(direction)}

        for name in self.detector_names:
            width = max_hits[name]
            fields = {"channel": np.zeros((batch_size, width), dtype=np.int64),
                      "t": np.zeros((batch_size, width), dtype=np.float32),
                      "log_edep": np.zeros((batch_size, width), dtype=np.float32),
                      "weight": np.zeros((batch_size, width), dtype=np.float32),
                      "mask": np.zeros((batch_size, width), dtype=np.float32)}
            columns = np.arange(width, dtype=np.int64)
            for rows, hit, norm, detector_slices in groups:
                starts, counts = detector_slices[name]
                row_indices, column_indices = np.nonzero(columns[None, :] < counts[:, None])
                if not len(row_indices):
                    continue
                source_indices = starts[row_indices] + column_indices
                dest_rows = rows[row_indices]
                raw = hit[name]
                stats = norm[_NORM_NAMES[name]]
                edep = np.maximum(np.asarray(raw["Edep"][source_indices], dtype=np.float32), 0.0)
                fields["channel"][dest_rows, column_indices] = raw["channel"][source_indices]
                fields["t"][dest_rows, column_indices] = (
                    raw["t"][source_indices] - stats["t_mean"]) / stats["t_std"]
                fields["log_edep"][dest_rows, column_indices] = (
                    np.log1p(edep) - stats["log1p_edep_mean"]) / stats["log1p_edep_std"]
                fields["weight"][dest_rows, column_indices] = edep
                fields["mask"][dest_rows, column_indices] = 1.0
            batch[name] = {field: torch.from_numpy(values) for field, values in fields.items()}
        return batch, target
