"""Evaluate a selected Phase-I GNN checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .geometry import adjacency, load_geometry
from .gnn_data import GraphEventDataset, load_graph_data, split_indices
from .gnn_model import PhaseIGNN, losses


@torch.no_grad()
def evaluate(checkpoint_path: str | Path, data_dir: str | Path, geometry_path: str | Path,
             output_dir: str | Path, *, geometry_manifest: str | Path | None = None,
             batch_size: int = 256, device: str = "cuda:0",
             max_test_events: int | None = None, baseline_checkpoint: str | Path | None = None) -> dict:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_sha = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if checkpoint.get("data_manifest_sha256") != dataset_sha:
        raise ValueError("checkpoint and evaluation data manifest hashes differ")
    geometry, geometry_meta = load_geometry(geometry_path, geometry_manifest)
    if checkpoint.get("geometry_yaml_sha256") != geometry_meta["geometry_yaml_sha256"] or checkpoint.get(
            "geometry_export_sha256") != geometry_meta["geometry_export_sha256"]:
        raise ValueError("checkpoint and geometry hashes differ")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    particles = load_graph_data(data_dir, geometry, geometry_meta, allowed_split_ids=(2,))
    indices = split_indices(particles)["test"]
    if max_test_events is not None and len(indices) > max_test_events:
        rng = np.random.default_rng(61471)
        selected = rng.choice(len(indices), size=max_test_events, replace=False)
        indices = [indices[int(i)] for i in selected]
    norms = checkpoint["input_normalization"]
    target_norm = checkpoint["target_normalization"]
    adj = {name: adjacency(geometry, name) for name in ("scifi", "ttc", "ecal")}
    dataset = GraphEventDataset(particles, indices, norms, target_norm, adj)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=dataset.collate, pin_memory=dev.type == "cuda")
    model = PhaseIGNN().to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    totals = {name: 0. for name in ("pid", "energy", "position", "direction")}
    n = 0
    correct = 0
    parts, event_indices, pid_true, pid_pred, masks, energies, e_err = [], [], [], [], [], [], []
    x_err, theta_err, event_nll = [], [], []
    e_mean = target_norm["energy_mean"]
    e_std = target_norm["energy_std"]
    x_mean = np.asarray(target_norm["position_mean"], dtype=np.float64)
    x_std = np.asarray(target_norm["position_std"], dtype=np.float64)
    for graph, target in loader:
        graph = {d: {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in values.items()} for d, values in graph.items()}
        target_dev = {k: v.to(dev, non_blocking=True) for k, v in target.items()}
        output = model(graph)
        components = losses(output, target_dev)
        batch_n = len(target["pid"])
        n += batch_n
        for key, value in components.items():
            totals[key] += float(value.sum())
        correct += int((output["pid"].argmax(-1) == target_dev["pid"]).sum())
        event_nll.append(sum(value.detach().cpu().numpy() for value in components.values()))
        pid_true.extend(target["pid"].cpu().numpy().tolist())
        pid_pred.extend(output["pid"].argmax(-1).cpu().numpy().tolist())
        masks.extend(target["response_mask"].cpu().numpy().tolist())
        # DataLoader order follows split_indices' particle blocks.
        batch_start = n - batch_n
        parts.extend(dataset.particles[indices[batch_start + i][0]].particle for i in range(batch_n))
        event_indices.extend(int(indices[batch_start + i][1]) for i in range(batch_n))

        energy = output["energy"]
        ew = energy[:, :5].softmax(-1)
        epred = (ew * energy[:, 5:10]).sum(-1).cpu().numpy() * e_std + e_mean
        etrue = target["energy"].cpu().numpy() * e_std + e_mean
        energies.extend(etrue.tolist())
        e_err.extend((epred - etrue).tolist())

        position = output["position"]
        xw = position[:, :4].softmax(-1)
        xpred = (xw[:, :, None] * position[:, 4:16].reshape(-1, 4, 3)).sum(1).cpu().numpy()
        xtrue = target["position"].cpu().numpy()
        xpred = xpred * x_std + x_mean
        xtrue = xtrue * x_std + x_mean
        x_err.extend(np.linalg.norm(xpred - xtrue, axis=1).tolist())

        direction = output["direction"]
        dw = direction[:, :4].softmax(-1)
        dmean = torch.nn.functional.normalize(direction[:, 4:16].reshape(-1, 4, 3), dim=-1)
        dpred = torch.nn.functional.normalize((dw[:, :, None] * dmean).sum(1), dim=-1)
        dtrue = torch.nn.functional.normalize(target_dev["direction"], dim=-1)
        angle = torch.rad2deg(torch.acos((dpred * dtrue).sum(-1).clamp(-1, 1)))
        theta_err.extend(angle.cpu().numpy().tolist())
    if not n:
        raise ValueError("test split is empty")
    mean_losses = {key: value / n for key, value in totals.items()}
    arrays = {"particle": np.asarray(parts), "event_index": np.asarray(event_indices, dtype=np.int64),
              "pid_true": np.asarray(pid_true),
              "pid_pred": np.asarray(pid_pred), "response_mask": np.asarray(masks),
              "energy_true": np.asarray(energies), "energy_error": np.asarray(e_err),
              "position_error": np.asarray(x_err), "direction_angle_error": np.asarray(theta_err),
              "event_nll": np.concatenate(event_nll)}

    def summarize(mask):
        count = int(mask.sum())
        if not count:
            return {"n": 0}
        confusion = np.zeros((2, 2), dtype=np.int64)
        np.add.at(confusion, (arrays["pid_true"][mask].astype(int), arrays["pid_pred"][mask].astype(int)), 1)
        energy_res = arrays["energy_error"][mask]
        position_res = arrays["position_error"][mask]
        angle_res = arrays["direction_angle_error"][mask]
        return {"n": count, "pid_confusion_true_rows": confusion.tolist(),
                "pid_accuracy": float(np.mean(arrays["pid_true"][mask] == arrays["pid_pred"][mask])),
                "energy_bias_mev": float(energy_res.mean()),
                "energy_abs_q68_mev": float(np.quantile(np.abs(energy_res), .68)),
                "position_3d_q68_mm": float(np.quantile(position_res, .68)),
                "direction_angle_q68_deg": float(np.quantile(angle_res, .68)),
                "direction_angle_q95_deg": float(np.quantile(angle_res, .95)),
                "total_nll": float(arrays["event_nll"][mask].mean())}

    groups = {"all": summarize(np.ones(n, dtype=bool))}
    for particle in sorted(set(parts)):
        groups[f"particle:{particle}"] = summarize(arrays["particle"] == particle)
    edges = [0., 5., 10., 20., 40., 60.0001]
    energy_bins = np.digitize(arrays["energy_true"], edges, right=False) - 1
    for i, label in enumerate(("0-5", "5-10", "10-20", "20-40", "40-60")):
        groups[f"energy_mev:{label}"] = summarize(energy_bins == i)
    combination_labels = {0: "none", 1: "TTC", 2: "SciFi", 3: "TTC+SciFi",
                          4: "ECAL", 5: "TTC+ECAL", 6: "SciFi+ECAL", 7: "TTC+SciFi+ECAL"}
    for bitmask, label in combination_labels.items():
        groups[f"hit_combination:{label}"] = summarize(arrays["response_mask"] == bitmask)

    metrics = {
        "test_events": n,
        "losses": mean_losses,
        "total_nll": sum(mean_losses.values()),
        "pid_accuracy": correct / n,
        "energy_bias_mev": float(np.mean(e_err)),
        "energy_abs_q68_mev": float(np.quantile(np.abs(e_err), .68)),
        "position_3d_q68_mm": float(np.quantile(x_err, .68)),
        "direction_angle_q68_deg": float(np.quantile(theta_err, .68)),
        "direction_angle_q95_deg": float(np.quantile(theta_err, .95)),
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint["best_epoch"],
        "data_manifest_sha256": dataset_sha,
        "geometry_yaml_sha256": geometry_meta["geometry_yaml_sha256"],
        "geometry_export_sha256": geometry_meta["geometry_export_sha256"],
        "grouped_metrics": groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_checkpoint is not None:
        from .evaluate import evaluate as evaluate_old
        baseline_checkpoint = Path(baseline_checkpoint).expanduser().resolve()
        old_metrics = evaluate_old(baseline_checkpoint, data_dir, batch_size=batch_size,
                                   device=device, output_dir=output_dir / "old_100k_baseline")
        old_manifest_sha = old_metrics["checkpoint"].get("data_manifest_sha256")
        if old_manifest_sha != dataset_sha:
            raise ValueError("old baseline and GNN evaluation data manifest hashes differ")
        metrics["old_100k_baseline"] = {"checkpoint": str(baseline_checkpoint),
                                         "best_epoch": old_metrics["checkpoint"].get("best_epoch"),
                                         "data_manifest_sha256": old_manifest_sha,
                                         "grouped_metrics": {key: value for key, value in
                                             old_metrics["grouped_metrics"].items()
                                             if key == "all" or key.startswith("particle:") or
                                             key.startswith("response_mask:")}}
    np.savez_compressed(output_dir / "test_predictions.npz", **arrays,
                        data_manifest_sha256=np.asarray(dataset_sha),
                        geometry_export_sha256=np.asarray(geometry_meta["geometry_export_sha256"]))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Phase-I GNN held-out evaluation", "", "Only the selected checkpoint is evaluated here; these values describe the SimHit simulation study.",
             "", "SciFi cross-layer edges are geometry-based candidate adjacencies derived from fiber local IDs, rotation, pitch, and radius over at most four layers. They approximate neighborhood structure and do not represent exact helix intersections at a reconstructed z; this may affect direction metrics.",
             "", f"Test events: {n}", "", "## Aggregate and grouped metrics", "",
             "| Group | N | PID accuracy | Energy bias (MeV) | Energy abs q68 (MeV) | Primary vertex x0 error q68 (mm) | Direction q68 (deg) | Total NLL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, row in groups.items():
        if row["n"]:
            lines.append(f"| {group} | {row['n']} | {row['pid_accuracy']:.4f} | {row['energy_bias_mev']:.4g} | {row['energy_abs_q68_mev']:.4g} | {row['position_3d_q68_mm']:.4g} | {row['direction_angle_q68_deg']:.4g} | {row['total_nll']:.5g} |")
    if "old_100k_baseline" in metrics:
        old = metrics["old_100k_baseline"]["grouped_metrics"].get("all", {})
        new = groups["all"]
        lines.extend(["", "## Same-data old 100k baseline", "",
                      f"Baseline checkpoint: `{baseline_checkpoint}` (best epoch {metrics['old_100k_baseline']['best_epoch']}).",
                      f"Both checkpoints use dataset manifest `{dataset_sha}` and the same held-out event split.",
                      "Cross-model NLL is omitted because the probabilistic head parameterizations and likelihood scales differ. `test_predictions.npz` stores each event's particle and particle-local event index; `data_manifest_sha256` binds those indices to the source dataset.", "",
                      "| Metric | GNN | Old baseline |", "|---|---:|---:|"])
        for key, label in (("pid_accuracy", "PID accuracy"), ("energy_bias_mev", "Energy bias (MeV)"),
                           ("energy_abs_q68_mev", "Energy abs q68 (MeV)"),
                           ("position_3d_q68_mm", "Primary-vertex x0 3D error q68 (mm)"),
                           ("direction_angle_q68_deg", "Direction q68 (deg)")):
            old_key = {"energy_abs_q68_mev": "energy_abs_residual_68_mev",
                       "position_3d_q68_mm": "x0_error_68_mm",
                       "direction_angle_q68_deg": "direction_angle_68_deg"}.get(key, key)
            if key in new and old_key in old:
                lines.append(f"| {label} | {new[key]:.5g} | {old[old_key]:.5g} |")
        lines.extend(["", "### PID confusion matrices", "",
                      "Rows are true gamma and e+, in that order; columns are predicted gamma and e+.", "",
                      "| Model | gamma→gamma | gamma→e+ | e+→gamma | e+→e+ |",
                      "|---|---:|---:|---:|---:|"])
        for label, row in (("GNN", new), ("Old 100k baseline", old)):
            matrix = row["pid_confusion_true_rows"]
            lines.append(f"| {label} | {matrix[0][0]} | {matrix[0][1]} | {matrix[1][0]} | {matrix[1][1]} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-test-events", type=int)
    parser.add_argument("--baseline-checkpoint", type=Path)
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.data, args.geometry, args.output_dir,
             geometry_manifest=args.geometry_manifest, batch_size=args.batch_size, device=args.device,
             max_test_events=args.max_test_events, baseline_checkpoint=args.baseline_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
