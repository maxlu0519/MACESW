"""Train the opt-in sparse Phase-I GNN without reading held-out test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .geometry import adjacency, load_geometry
from .gnn_data import GraphEventDataset, fit_normalization, load_graph_data, split_indices
from .gnn_model import PhaseIGNN, losses, parameter_count


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hash() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("gnn_model.py", "gnn_data.py", "geometry.py", "gnn_train.py"):
        path = package / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _eval(model, loader, device) -> dict[str, float]:
    model.eval()
    sums = {name: 0. for name in ("pid", "energy", "position", "direction")}
    count = 0
    with torch.no_grad():
        for graph, target in loader:
            graph = {d: {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in values.items()} for d, values in graph.items()}
            target = {k: v.to(device, non_blocking=True) for k, v in target.items()}
            output = model(graph)
            component = losses(output, target)
            n = len(target["pid"])
            count += n
            for name, values in component.items():
                sums[name] += float(values.sum())
    if not count:
        raise ValueError("validation split is empty")
    result = {name: value / count for name, value in sums.items()}
    result["total"] = sum(result.values())
    return result


def train(data_dir: str | Path, geometry_path: str | Path, output: str | Path, *,
          geometry_manifest: str | Path | None = None, last_output: str | Path | None = None,
          epochs: int = 80, patience: int = 10, batch_size: int = 256,
          grad_accum: int = 8, seed: int = 42, device: str = "cuda:0",
          max_train_events: int | None = None, max_validation_events: int | None = None,
          num_workers: int = 0) -> dict:
    data_dir = Path(data_dir).expanduser().resolve()
    geometry_path = Path(geometry_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    last_output = Path(last_output).expanduser().resolve() if last_output else output.with_name("last.pt")
    manifest_path = data_dir / "manifest.json"
    dataset_sha = _sha256(manifest_path)
    geometry, geometry_manifest = load_geometry(geometry_path, geometry_manifest)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Construct only train and validation input graphs. Test hit values are not
    # read until the separately invoked evaluation command.
    particles = load_graph_data(data_dir, geometry, geometry_manifest, allowed_split_ids=(0, 1))
    split = split_indices(particles)
    if not split["train"] or not split["validation"]:
        raise ValueError("training and validation splits must both be nonempty")
    if max_train_events is not None:
        rng = np.random.default_rng(seed)
        chosen = rng.permutation(len(split["train"]))[:max_train_events]
        split["train"] = [split["train"][int(i)] for i in chosen]
    if max_validation_events is not None:
        rng = np.random.default_rng(seed + 1)
        chosen = rng.permutation(len(split["validation"]))[:max_validation_events]
        split["validation"] = [split["validation"][int(i)] for i in chosen]
    input_norm, target_norm = fit_normalization(particles)
    adj = {name: adjacency(geometry, name) for name in ("scifi", "ttc", "ecal")}
    train_data = GraphEventDataset(particles, split["train"], input_norm, target_norm, adj)
    val_data = GraphEventDataset(particles, split["validation"], input_norm, target_norm, adj)
    loader_options = {"num_workers": num_workers, "persistent_workers": num_workers > 0,
                      "prefetch_factor": 2 if num_workers > 0 else None}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=train_data.collate, pin_memory=dev.type == "cuda",
                              generator=torch.Generator().manual_seed(seed), **loader_options)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            collate_fn=val_data.collate, pin_memory=dev.type == "cuda", **loader_options)
    model = PhaseIGNN().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = output.parent / "history.jsonl"
    best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    started = time.monotonic()
    accum_batches = max(1, grad_accum)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        pending = []
        train_sums = {name: 0. for name in ("pid", "energy", "position", "direction")}
        train_events = 0

        def step_group(group):
            divisor = len(group)
            for graph, target in group:
                graph = {d: {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
                             for k, v in values.items()} for d, values in graph.items()}
                target = {k: v.to(dev, non_blocking=True) for k, v in target.items()}
                component = losses(model(graph), target)
                total = sum(values.mean() for values in component.values())
                (total / divisor).backward()
                n = len(target["pid"])
                for name, values in component.items():
                    train_sums[name] += float(values.detach().sum())
                nonlocal train_events
                train_events += n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for graph, target in train_loader:
            pending.append((graph, target))
            if len(pending) == accum_batches:
                step_group(pending)
                pending = []
        if pending:
            step_group(pending)

        validation = _eval(model, val_loader, dev)
        train_metrics = {name: value / train_events for name, value in train_sums.items()}
        train_metrics["total"] = sum(train_metrics.values())
        row = {"epoch": epoch, "train": train_metrics, "validation": validation,
               "seconds": time.monotonic() - epoch_start}
        history.append(row)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        improved = validation["total"] < best - 1e-5
        if improved:
            best = validation["total"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        state = {"schema_version": 1, "model_type": "phasei_gnn", "model_state": model.state_dict(),
                 "model_config": {"spec": "phasei_gnn_v1", "parameters": parameter_count(model)},
                 "training_config": {"seed": seed, "epochs_requested": epochs, "batch_size": batch_size,
                                     "gradient_accumulation": grad_accum, "effective_batch_size": batch_size * grad_accum,
                                     "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
                                     "input_features": "channel,time,edep,exported geometry; no truth inputs",
                                     "test_read_during_training": False,
                                     "num_workers": num_workers,
                                     "max_train_events": max_train_events,
                                     "max_validation_events": max_validation_events},
                 "best_epoch": best_epoch, "best_validation_loss": best,
                 "validation_metrics": validation, "input_normalization": input_norm,
                 "target_normalization": target_norm, "data_manifest_sha256": dataset_sha,
                 "geometry_yaml_sha256": geometry_manifest["geometry_yaml_sha256"],
                 "geometry_export_sha256": geometry_manifest["geometry_export_sha256"],
                 "code_snapshot_sha256": _code_hash(), "history": history}
        torch.save(state, last_output)
        if improved:
            torch.save(state, output)
        if dev.type == "cuda":
            row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        print(json.dumps(row, sort_keys=True), flush=True)
        if bad_epochs >= patience:
            break

    summary = {"best_checkpoint": str(output), "last_checkpoint": str(last_output),
               "best_epoch": best_epoch, "best_validation_loss": best,
               "epochs_completed": len(history), "elapsed_seconds": time.monotonic() - started,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else None,
               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if dev.type == "cuda" else None,
               "dataset_manifest_sha256": dataset_sha,
               "geometry_yaml_sha256": geometry_manifest["geometry_yaml_sha256"],
               "geometry_export_sha256": geometry_manifest["geometry_export_sha256"],
               "train_events": len(train_data), "validation_events": len(val_data),
               "num_workers": num_workers,
               "test_read_during_training": False}
    (output.parent / "training_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--last-output", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-events", type=int)
    parser.add_argument("--max-validation-events", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    result = train(args.data, args.geometry, args.output, geometry_manifest=args.geometry_manifest,
                   last_output=args.last_output, epochs=args.epochs, patience=args.patience,
                   batch_size=args.batch_size, grad_accum=args.grad_accum, seed=args.seed, device=args.device,
                   max_train_events=args.max_train_events, max_validation_events=args.max_validation_events,
                   num_workers=args.num_workers)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
