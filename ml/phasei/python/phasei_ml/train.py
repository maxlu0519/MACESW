"""Train the Phase-I probabilistic reconstruction baseline."""
from __future__ import annotations

import argparse
import copy
from functools import partial
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .fast_loader import VectorizedEventDataset, identity_collate
from .model import PhaseIModel, losses, parameter_count


class EventData(Dataset):
    def __init__(self, directory: Path, indices: np.ndarray, target_scaler: dict | None = None,
                 detector_names: tuple[str, ...] = ("ttc", "scifi", "ecal")):
        self.directory = directory
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_scaler = target_scaler
        self.arrays: dict[str, np.ndarray] = {}
        for name in ("pid", "pdgid", "t0", "Ek0", "x0", "p0", "response_mask", "split"):
            self.arrays[name] = np.load(directory / f"{name}.npy", mmap_mode="r")
        self.norm = json.loads((directory / "normalization.json").read_text())
        self.hit = {}
        self.detector_names = tuple(detector_names)
        for name in self.detector_names:
            self.hit[name] = {field: np.load(directory / f"{name}_{field}.npy", mmap_mode="r")
                              for field in ("channel", "t", "Edep", "offsets")}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        event = int(self.indices[item])
        result = {key: np.asarray(self.arrays[key][event]) for key in ("pid", "Ek0", "x0", "p0", "response_mask")}
        for detector in self.detector_names:
            arrays = self.hit[detector]
            start, stop = map(int, arrays["offsets"][event:event + 2])
            norm = self.norm[{"ttc": "TTC", "scifi": "SciFi", "ecal": "ECAL"}[detector]]
            time = np.asarray(arrays["t"][start:stop], dtype=np.float32)
            energy = np.maximum(np.asarray(arrays["Edep"][start:stop], dtype=np.float32), 0.0)
            loge = np.log1p(energy).astype(np.float32)
            time = (time - norm["t_mean"]) / norm["t_std"]
            loge = (loge - norm["log1p_edep_mean"]) / norm["log1p_edep_std"]
            result[detector] = {"channel": np.asarray(arrays["channel"][start:stop], dtype=np.int64),
                                "t": time, "log_edep": loge, "weight": energy,
                                "mask": np.ones(stop - start, dtype=np.float32)}
        p = np.asarray(result["p0"], dtype=np.float32)
        result["direction"] = p / max(float(np.linalg.norm(p)), 1e-12)
        return result


def _collate(items: list[dict[str, Any]], target_scaler: dict | None = None,
             detector_names: tuple[str, ...] = ("ttc", "scifi", "ecal")) -> tuple[dict, dict]:
    batch = {key: torch.as_tensor(np.asarray([item[key] for item in items]), dtype=torch.float32)
             for key in ("response_mask",)}
    energy = np.asarray([float(item["Ek0"]) for item in items], dtype=np.float32)
    position = np.asarray([item["x0"] for item in items], dtype=np.float32)
    if target_scaler is not None:
        energy = (energy - target_scaler["energy"]["mean"]) / target_scaler["energy"]["std"]
        position = (position - np.asarray(target_scaler["position"]["mean"], dtype=np.float32)) / np.asarray(target_scaler["position"]["std"], dtype=np.float32)
    target = {"pid": torch.as_tensor([int(item["pid"]) for item in items], dtype=torch.long),
              "energy": torch.as_tensor(energy, dtype=torch.float32),
              "position": torch.as_tensor(position, dtype=torch.float32),
              "direction": torch.as_tensor(np.asarray([item["direction"] for item in items]), dtype=torch.float32)}
    for detector in detector_names:
        max_hits = max(1, max(len(item[detector]["channel"]) for item in items))
        fields = {key: np.zeros((len(items), max_hits), dtype=np.int64 if key == "channel" else np.float32)
                  for key in ("channel", "t", "log_edep", "weight", "mask")}
        for i, item in enumerate(items):
            data = item[detector]
            count = len(data["channel"])
            for key in fields:
                fields[key][i, :count] = data[key]
        batch[detector] = {key: torch.from_numpy(value) for key, value in fields.items()}
    return batch, target


def _select_indices(directory: Path, split_id: int, limit: int | None, seed: int) -> np.ndarray:
    split = np.load(directory / "split.npy", mmap_mode="r")
    pid = np.load(directory / "pid.npy", mmap_mode="r")
    indices = np.flatnonzero(split == split_id)
    if limit is not None:
        rng = np.random.default_rng(seed + split_id)
        selected = []
        for class_id in (0, 1):
            candidates = indices[pid[indices] == class_id].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[:limit].tolist())
        indices = np.asarray(selected, dtype=np.int64)
    return indices


def _load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fit_target_scaler(data_dir: str | Path, train_indices: dict[str, np.ndarray]) -> dict:
    """Fit target standardization from training rows only, pooled across species."""
    energies, positions = [], []
    for particle, indices in train_indices.items():
        directory = Path(data_dir) / particle
        energies.append(np.asarray(np.load(directory / "Ek0.npy", mmap_mode="r")[indices], dtype=np.float64))
        positions.append(np.asarray(np.load(directory / "x0.npy", mmap_mode="r")[indices], dtype=np.float64))
    if not energies or not sum(len(x) for x in energies):
        raise ValueError("cannot fit target scaler on an empty training set")
    e = np.concatenate(energies)
    x = np.concatenate(positions)
    e_mean, e_std = float(e.mean()), float(e.std())
    x_mean, x_std = x.mean(0), x.std(0)
    return {"energy": {"mean": e_mean, "std": max(e_std, 1e-8)},
            "position": {"mean": x_mean.tolist(), "std": np.maximum(x_std, 1e-8).tolist()}}


def _indices_by_particle(data_dir: Path, split_id: int, limit: int | None, seed: int) -> dict[str, np.ndarray]:
    return {particle["particle"]: _select_indices(data_dir / particle["particle"], split_id, limit, seed)
            for particle in _load_manifest(data_dir / "manifest.json")["particles"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_snapshot_sha256() -> str:
    package_dir=Path(__file__).resolve().parent
    files=sorted(package_dir.glob("*.py"))+[package_dir.parents[1]/"config"/"default.yaml"]
    digest=hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(data_dir: str | Path, output: str | Path, *, epochs=30, batch_size=256,
          learning_rate=3e-4, train_limit_per_class=None, eval_limit_per_class=None,
          seed=42, use_experts=True, device=None, patience=5,
          last_output: str | Path | None = None, resume: str | Path | None = None,
          loss_weights: dict[str, float] | None = None,
          input_pipeline: str = "vectorized", num_workers: int = 0,
          detector_names: tuple[str, ...] = ("ttc", "scifi", "ecal")) -> dict:
    data_dir = Path(data_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    last_output = Path(last_output).expanduser().resolve() if last_output else output.with_name("last.pt")
    manifest = _load_manifest(data_dir / "manifest.json")
    detector_names = tuple(detector_names)
    if not detector_names or any(name not in ("ttc", "scifi", "ecal") for name in detector_names):
        raise ValueError("detector_names must select one or more of ttc, scifi, ecal")
    manifest_hash = _sha256(data_dir / "manifest.json")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable; found {torch.cuda.device_count()} device(s)")
    weights = {"pid": 1.0, "energy": 1.0, "position": 1.0, "direction": 1.0}
    if loss_weights:
        weights.update({key: float(value) for key, value in loss_weights.items()})
    if set(weights) != {"pid", "energy", "position", "direction"} or any(v <= 0 for v in weights.values()):
        raise ValueError("loss_weights must contain positive weights for pid, energy, position, and direction")
    channel_counts = {}
    for particle in manifest["particles"]:
        for detector, info in particle["normalization"].items():
            if detector.lower() in detector_names:
                channel_counts[detector.lower()] = max(channel_counts.get(detector.lower(), 1), int(info["channel_count"]))
    model = PhaseIModel(channel_counts, use_experts=use_experts, detector_names=detector_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_indices = {}
    validation_indices = {}
    for particle in manifest["particles"]:
        directory = data_dir / particle["particle"]
        for split_id, name in ((0, "train"), (1, "validation")):
            limit = train_limit_per_class if split_id == 0 else eval_limit_per_class
            indices = _select_indices(directory, split_id, limit, seed)
            if split_id == 0:
                train_indices[particle["particle"]] = indices
            else:
                validation_indices[particle["particle"]] = indices
    scaler = fit_target_scaler(data_dir, train_indices)
    index_digest = hashlib.sha256()
    for particle in sorted(train_indices):
        index_digest.update(particle.encode("utf-8"))
        index_digest.update(np.asarray(train_indices[particle], dtype=np.int64).tobytes())
    train_index_sha256 = index_digest.hexdigest()
    if input_pipeline == "vectorized":
        datasets = {"train": VectorizedEventDataset(data_dir, train_indices, scaler, detector_names),
                    "validation": VectorizedEventDataset(data_dir, validation_indices, scaler, detector_names)}
        collate_fn = identity_collate
    elif input_pipeline == "legacy":
        datasets = {
            "train": torch.utils.data.ConcatDataset([
                EventData(data_dir / particle, indices, scaler, detector_names)
                for particle, indices in train_indices.items() if len(indices)]),
            "validation": torch.utils.data.ConcatDataset([
                EventData(data_dir / particle, indices, scaler)
                for particle, indices in validation_indices.items() if len(indices)]),
        }
        collate_fn = partial(_collate, target_scaler=scaler, detector_names=detector_names)
    else:
        raise ValueError("input_pipeline must be 'vectorized' or 'legacy'")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    loader_options = {"batch_size": batch_size, "num_workers": num_workers,
                      "collate_fn": collate_fn, "pin_memory": device.type == "cuda"}
    if num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    sampler_generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=sampler_generator, **loader_options),
        "validation": DataLoader(datasets["validation"], shuffle=False, **loader_options),
    }
    if "train" not in loaders:
        raise ValueError("training split is empty")
    history = []
    start_epoch = 0
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("data_manifest_sha256") != manifest_hash:
            raise ValueError("resume checkpoint dataset manifest hash does not match")
        if state.get("training_config", {}).get("seed") != seed or state.get("model_config", {}).get("use_experts") != use_experts:
            raise ValueError("resume checkpoint seed or model mode does not match this run")
        old_config=state.get("training_config",{})
        if (old_config.get("batch_size")!=batch_size or old_config.get("learning_rate")!=learning_rate or
                old_config.get("loss_weights")!=weights or state.get("train_indices_sha256")!=train_index_sha256 or
                old_config.get("input_pipeline", "legacy")!=input_pipeline or
                old_config.get("num_workers", 0)!=num_workers):
            raise ValueError("resume checkpoint batch size, learning rate, loss weights, input pipeline, worker count, or training event indices differ")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state.get("history", [])
        start_epoch = int(state["epoch"])
        best_val = float(state.get("best_val", math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        bad_epochs = int(state.get("bad_epochs", 0))
        if "python_rng" in state:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng"]])
    started = time.perf_counter()
    events_this_run = 0
    epochs_this_run = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_sums = {key: 0.0 for key in weights}; n_batches = 0; n_events = 0
        for batch, target in loaders["train"]:
            batch = _move(batch, device); target = _move(target, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            parts = losses(out, target)
            loss = sum(weights[key] * parts[key].mean() for key in weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            for key in train_sums:
                train_sums[key] += float(parts[key].mean().detach())
            n_batches += 1; n_events += len(target["pid"])
        events_this_run += n_events
        epochs_this_run += 1
        train_parts = {key: value / max(n_batches, 1) for key, value in train_sums.items()}
        validation = _evaluate_losses(model, loaders.get("validation"), device)
        val_score = sum(weights[key] * validation[key] for key in weights)
        if not math.isfinite(val_score) or any(not math.isfinite(value) for value in validation.values()):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch + 1}")
        elapsed = time.perf_counter() - started
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_row = {"epoch": epoch + 1, "train_loss": sum(weights[k] * train_parts[k] for k in weights),
                     "train_by_head": train_parts, "validation_loss": val_score,
                     "validation_by_head": validation, "elapsed_seconds": elapsed,
                     "epoch_seconds": epoch_seconds,
                     "events_per_second": n_events / max(epoch_seconds, 1e-9)}
        history.append(epoch_row)
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        improved = val_score < best_val - 1e-4
        if improved:
            best_val, best_epoch, bad_epochs = val_score, epoch + 1, 0
        else:
            bad_epochs += 1
        common = {"schema_version": 2, "model_config": {"channel_counts": channel_counts, "use_experts": use_experts,
                  "detector_names": detector_names},
                  "training_config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                                      "seed": seed, "dtype": "float32", "optimizer": "AdamW", "patience": patience,
                                      "loss_weights": weights, "input_pipeline": input_pipeline,
                                      "num_workers": num_workers, "detector_names": detector_names},
                  "target_scaler": scaler, "normalization": {p["particle"]: p["normalization"] for p in manifest["particles"]},
                  "data_manifest_sha256": manifest_hash, "train_indices_sha256": train_index_sha256,
                  "code_snapshot_sha256": _code_snapshot_sha256(), "history": history,
                  "best_val": best_val, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                  "model_state": model.state_dict()}
        if improved:
            _atomic_torch_save(common, output)
        last_state = copy.deepcopy(common)
        last_state.update({"epoch": epoch + 1, "optimizer_state": optimizer.state_dict(),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        _atomic_torch_save(last_state, last_output)
        if bad_epochs >= patience:
            break
    elapsed = time.perf_counter() - started
    if not output.exists():
        raise RuntimeError("training ended without a finite best validation checkpoint")
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    checkpoint["runtime"] = {"elapsed_seconds": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                             "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    _atomic_torch_save(checkpoint, output)
    result = {"parameters": parameter_count(model), "best_epoch": best_epoch,
              "best_validation_loss": best_val, "epochs_completed": len(history),
              "elapsed_seconds": elapsed, "events_per_second": events_this_run / max(elapsed, 1e-9),
              "epochs_this_run": epochs_this_run,
              "peak_cuda_allocated_bytes": checkpoint["runtime"]["peak_cuda_allocated_bytes"],
              "peak_cuda_reserved_bytes": checkpoint["runtime"]["peak_cuda_reserved_bytes"],
              "checkpoint": str(output), "last_checkpoint": str(last_output)}
    print(json.dumps(result, sort_keys=True))
    return result


@torch.no_grad()
def _evaluate_losses(model, loader, device):
    if loader is None:
        raise ValueError("validation loader is empty")
    model.eval(); sums = {"pid": 0.0, "energy": 0.0, "position": 0.0, "direction": 0.0}; count = 0
    for batch, target in loader:
        batch, target = _move(batch, device), _move(target, device)
        parts = losses(model(batch), target)
        n = len(target["pid"]); count += n
        for key in sums:
            sums[key] += float(parts[key].mean()) * n
    return {key: value / max(count, 1) for key, value in sums.items()}


def _move(batch, device):
    return {key: ({subkey: value.to(device, non_blocking=True) for subkey, value in item.items()}
                  if isinstance(item, dict) else item.to(device, non_blocking=True))
            for key, item in batch.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config" / "default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--last-output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--input-pipeline", choices=("vectorized", "legacy"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--train-limit-per-class", type=int)
    parser.add_argument("--eval-limit-per-class", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-experts", action="store_true", help="train the no-router ablation")
    parser.add_argument("--device")
    parser.add_argument("--detectors", nargs="+", choices=("ttc", "scifi", "ecal"))
    args = parser.parse_args(argv)
    try:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        parser.error(f"cannot read training config {args.config}: {exc}")
    training = config.get("training", {})
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    device = args.device if args.device is not None else config.get("device")
    train(args.data, args.output, epochs=args.epochs if args.epochs is not None else int(training.get("epochs", 30)),
          batch_size=args.batch_size if args.batch_size is not None else int(training.get("batch_size", 256)),
          learning_rate=args.learning_rate if args.learning_rate is not None else float(training.get("learning_rate", 3e-4)),
          patience=args.patience if args.patience is not None else int(training.get("patience", 5)),
          train_limit_per_class=args.train_limit_per_class,
          eval_limit_per_class=args.eval_limit_per_class,
          seed=seed, use_experts=not args.no_experts,
          device=device, last_output=args.last_output, resume=args.resume,
          loss_weights=training.get("loss_weights"),
          input_pipeline=args.input_pipeline if args.input_pipeline is not None else training.get("input_pipeline", "vectorized"),
          num_workers=args.num_workers if args.num_workers is not None else int(training.get("num_workers", 0)),
          detector_names=tuple(args.detectors if args.detectors is not None else training.get("detectors", ("ttc", "scifi", "ecal"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
