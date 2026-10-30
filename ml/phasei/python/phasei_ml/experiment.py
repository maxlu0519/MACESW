"""Run the Phase-I learning curve, full ablation ensemble, and final test reports."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .evaluate import evaluate
from .train import _code_snapshot_sha256, train


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        result=subprocess.check_output(["git","status","--porcelain"],text=True,stderr=subprocess.DEVNULL)
        return bool(result.strip())
    except Exception:
        return None


def _run_training(data: Path, run_dir: Path, *, seed: int, n_per_class: int | None,
                  use_experts: bool, device: str, config: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config={"seed": seed, "train_limit_per_class": n_per_class,
        "use_experts": use_experts, "device": device, **config}
    config_path=run_dir/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError(f"existing run {run_dir} has a different configuration")
    if not config_path.exists():
        config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    complete=run_dir/"training_result.json"
    if complete.exists():
        return json.loads(complete.read_text())
    training=config["training"]
    logfile=run_dir/"train.log"
    started=time.perf_counter()
    resume=run_dir/"last.pt"
    with logfile.open("a",encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
        result=train(data,run_dir/"best.pt",epochs=int(training["epochs"]),batch_size=int(training["batch_size"]),
                     learning_rate=float(training["learning_rate"]),patience=int(training["patience"]),
                     train_limit_per_class=n_per_class,seed=seed,use_experts=use_experts,device=device,
                     last_output=run_dir/"last.pt",resume=resume if resume.exists() else None,
                     loss_weights=training["loss_weights"])
    elapsed=time.perf_counter()-started
    result["wall_seconds"]=elapsed
    (run_dir/"training_result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def _aggregate(full_runs: list[dict], learning_runs: list[dict], output_dir: Path):
    summary={"runs":[],"groups":{}}
    fields=("pid_accuracy","energy_bias_mev","energy_abs_residual_68_mev","x0_error_68_mm","direction_angle_68_deg",
            "energy_nll","position_nll_mm","direction_nll_sr","energy_coverage_68","energy_coverage_95",
            "position_coverage_68","position_coverage_95","direction_coverage_68","direction_coverage_95")
    for run in full_runs:
        metrics=json.loads((run["run_dir"]/"metrics.json").read_text())
        summary["runs"].append({"name":run["name"],"model":run["model"],"seed":run["seed"],"best_epoch":run["training"]["best_epoch"],
                                "best_validation_loss":run["training"]["best_validation_loss"],"test_events":metrics["test_events"],
                                "metrics_file":str(run["run_dir"]/"metrics.json")})
        for group,row in metrics["grouped_metrics"].items():
            group_runs=summary["groups"].setdefault(group,{})
            model_runs=group_runs.setdefault(run["model"],{})
            model_runs.setdefault("per_seed",[]).append({"seed":run["seed"],**{key:row.get(key) for key in fields}})
    for group, models in summary["groups"].items():
        for model, info in models.items():
            for key in fields:
                vals=[r[key] for r in info["per_seed"] if r.get(key) is not None and np.isfinite(r[key])]
                info.setdefault("mean_std",{})[key]={"mean":float(np.mean(vals)) if vals else None,
                                                      "std":float(np.std(vals,ddof=1)) if len(vals)>1 else (0.0 if vals else None)}
    (output_dir/"ensemble_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    lines=["# Phase I training and baseline acceptance", "", "Existing γ/e⁺ single-particle SimHit sample; random stratified split. Results describe the simulated target-internal primary vertex, not detector-entry performance.", "", "## Full-size model runs", "", "| Model | Seed | Best epoch | Best validation loss | Test events |", "|---|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        lines.append(f"| {row['model']} | {row['seed']} | {row['best_epoch']} | {row['best_validation_loss']:.5g} | {row['test_events']} |")
    lines.extend(["", "## Aggregate performance by species and response mask", "", "Each entry is mean ± standard deviation over seeds. Full group metrics and bootstrap intervals are in the per-run JSON files.", "", "| Group | Model | PID acc | E bias (MeV) | E abs68 (MeV) | x0 error68 (mm) | angle68 (deg) |", "|---|---|---:|---:|---:|---:|---:|"])
    for group,models in summary["groups"].items():
        if group!="all" and not group.startswith("particle:") and not group.startswith("response_mask:"):
            continue
        for model,info in models.items():
            means=info["mean_std"]
            def fmt(key):
                m=means[key]["mean"]; s=means[key]["std"]
                return "n/a" if m is None else f"{m:.4g} ± {s:.2g}"
            lines.append(f"| {group} | {model} | {fmt('pid_accuracy')} | {fmt('energy_bias_mev')} | {fmt('energy_abs_residual_68_mev')} | {fmt('x0_error_68_mm')} | {fmt('direction_angle_68_deg')} |")
    lines.extend(["", "## Learning curves", "", "| Training events per species | Best epoch | Best validation loss | Events/s |", "|---:|---:|---:|---:|"])
    for run in learning_runs:
        res=run["training"]
        lines.append(f"| {run['n_per_class']} | {res['best_epoch']} | {res['best_validation_loss']:.5g} | {res['events_per_second']:.3g} |")
    baseline_file=full_runs[0]["run_dir"]/"metrics.json"
    if baseline_file.exists():
        bm=json.loads(baseline_file.read_text()).get("baselines",{})
        lines.extend(["", "## Baselines", "", "Training-fitted energy calibration and train-prior baselines are reported with the corresponding model run metrics.", "", "```json", json.dumps(bm,indent=2), "```"])
    lines.extend(["", "No physics-resolution threshold was assumed. Baseline and full grouped metrics are in each run's `metrics.json`; sparse groups with fewer than 100 events are descriptive only."])
    (output_dir/"report.md").write_text("\n".join(lines)+"\n")


def run_experiment(data_dir: str|Path, output_dir: str|Path, *, device="cuda:0", config_path: str|Path|None=None):
    data=Path(data_dir).resolve(); out=Path(output_dir).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("full Phase-I experiment requires CUDA; no CUDA device is visible")
    dev=torch.device(device)
    if dev.type!="cuda" or (dev.index is not None and dev.index>=torch.cuda.device_count()):
        raise RuntimeError(f"requested device {device} is not an available CUDA device")
    cfg_path=Path(config_path) if config_path else Path(__file__).resolve().parents[2]/"config"/"default.yaml"
    config=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config["training"].update({"epochs":30,"batch_size":256,"learning_rate":3e-4,"patience":5,
                               "loss_weights":{"pid":1.0,"energy":1.0,"position":1.0,"direction":1.0}})
    out.mkdir(parents=True,exist_ok=True)
    dataset_hash=hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest()
    metadata={"dataset":str(data),"dataset_manifest_sha256":dataset_hash,"config":config,"git_revision":_git_revision(),
              "git_dirty":_git_dirty(),"phasei_ml_source_sha256":_code_snapshot_sha256(),
              "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"device":str(dev),
              "gpu_name":torch.cuda.get_device_name(dev),"cuda_version":torch.version.cuda,"started_unix":time.time(),
              "evaluation_protocol":"No test split reads until all training runs complete."}
    metadata_path=out/"experiment.json"
    source_snapshots=[{"sha256":metadata["phasei_ml_source_sha256"],"description":"source used for current continuation"}]
    if metadata_path.exists():
        existing=json.loads(metadata_path.read_text())
        if (existing.get("dataset_manifest_sha256")!=dataset_hash or existing.get("device")!=str(dev) or
                existing.get("config")!=config):
            raise ValueError(f"existing experiment directory {out} belongs to another dataset, device, or config")
        metadata["started_unix"]=existing.get("started_unix",metadata["started_unix"])
        source_snapshots=existing.get("source_snapshots") or [{"sha256":existing.get("phasei_ml_source_sha256"),
                                                                  "description":"initial code snapshot before first training runs"}]
        current_hash=metadata["phasei_ml_source_sha256"]
        if not any(item.get("sha256")==current_hash for item in source_snapshots):
            source_snapshots.append({"sha256":current_hash,"description":"resume implementation with CUDA RNG state restored on CPU"})
    metadata["source_snapshots"]=source_snapshots
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    runs=[]
    learning_runs=[]
    # Learning curve: nested, deterministic per-class subsets, same validation set.
    for n in (5000,20000):
        name=f"learning_routed_n{n}_seed42"; run_dir=out/name
        result=_run_training(data,run_dir,seed=42,n_per_class=n,use_experts=True,device=str(dev),config=config)
        learning_runs.append({"name":name,"model":"routed","seed":42,"n_per_class":n,"training":result,"run_dir":run_dir,"learning_curve":True})
    # Identical seed-specific events and training budget for routed/no-expert comparisons.
    for model,use_experts in (("routed",True),("no_experts",False)):
        for seed in (42,43,44):
            name=f"{model}_seed{seed}"; run_dir=out/name
            result=_run_training(data,run_dir,seed=seed,n_per_class=None,use_experts=use_experts,device=str(dev),config=config)
            runs.append({"name":name,"model":model,"seed":seed,"training":result,"run_dir":run_dir,"learning_curve":False})
    # Only now open the held-out test data, after every training run is finished.
    full=[run for run in runs if not run["learning_curve"]]
    for run in full:
        if not all((run["run_dir"]/f).exists() for f in ("metrics.json","test_predictions.npz","baseline_predictions.npz")):
            evaluate(run["run_dir"]/"best.pt",data,device=str(dev),output_dir=run["run_dir"])
    _aggregate(full,learning_runs,out)
    metadata["completed_unix"]=time.time(); metadata["completed_runs"]=[r["name"] for r in runs]
    (out/"experiment.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return {"output_dir":str(out),"runs":[r["name"] for r in runs],"report":str(out/"report.md")}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("build/phasei_ml_training_v2"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--config",type=Path)
    args=parser.parse_args(argv)
    result=run_experiment(args.data,args.output_dir,device=args.device,config_path=args.config)
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
