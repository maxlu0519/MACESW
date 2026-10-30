"""Final held-out evaluation for Phase-I checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from functools import partial

from .model import PhaseIModel, losses
from .train import EventData, _collate, _load_manifest, _move, _select_indices


def _sample_energy(params: torch.Tensor, draws: int, generator: torch.Generator) -> torch.Tensor:
    k = params.shape[1] // 3
    logits, means, logstd = params[:, :k], params[:, k:2*k], params[:, 2*k:3*k].clamp(-7, 5)
    comp = torch.multinomial(logits.softmax(-1), draws, replacement=True, generator=generator)
    mu = means.gather(1, comp); sigma = logstd.exp().gather(1, comp)
    return mu + sigma * torch.randn(mu.shape, device=mu.device, generator=generator)


def _sample_position(params: torch.Tensor, draws: int, generator: torch.Generator) -> torch.Tensor:
    k = 4
    logits = params[:, :k]
    means = params[:, k:k+3*k].reshape(-1, k, 3)
    logstd = params[:, k+3*k:k+6*k].reshape(-1, k, 3).clamp(-7, 5)
    comp = torch.multinomial(logits.softmax(-1), draws, replacement=True, generator=generator)
    mu = means.gather(1, comp[..., None].expand(-1, -1, 3))
    sigma = logstd.exp().gather(1, comp[..., None].expand(-1, -1, 3))
    return mu + sigma * torch.randn(mu.shape, device=mu.device, generator=generator)


def _sample_direction(params: torch.Tensor, draws: int, generator: torch.Generator) -> torch.Tensor:
    k = 2
    logits = params[:, :k]
    means = torch.nn.functional.normalize(params[:, 2:8].reshape(-1, k, 3), dim=-1)
    concentration = torch.nn.functional.softplus(params[:, 8:10]).clamp(1e-5, 100.)
    comp = torch.multinomial(logits.softmax(-1), draws, replacement=True, generator=generator)
    mu = means.gather(1, comp[..., None].expand(-1, -1, 3))
    kap = concentration.gather(1, comp)
    u = torch.rand(kap.shape, device=kap.device, generator=generator).clamp(1e-7, 1-1e-7)
    # Inverse CDF for cos(theta) under a 3D von Mises-Fisher distribution.
    exp_term = torch.exp(-2 * kap)
    cos_theta = 1 + torch.log(u + (1-u) * exp_term) / kap.clamp_min(1e-5)
    phi = 2 * math.pi * torch.rand(kap.shape, device=kap.device, generator=generator)
    axis = torch.zeros_like(mu)
    axis[..., 0] = torch.where(mu[..., 0].abs() < .9, 1., 0.)
    axis[..., 1] = torch.where(mu[..., 0].abs() >= .9, 1., 0.)
    e1 = torch.nn.functional.normalize(torch.cross(mu, axis, dim=-1), dim=-1)
    e2 = torch.cross(mu, e1, dim=-1)
    sin_theta = (1-cos_theta.square()).clamp_min(0).sqrt()
    return torch.nn.functional.normalize(cos_theta[..., None]*mu + sin_theta[..., None]*(torch.cos(phi)[..., None]*e1 + torch.sin(phi)[..., None]*e2), dim=-1)


def _coverage_from_samples(samples: torch.Tensor, target: torch.Tensor, point: torch.Tensor, metric: str) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "energy":
        residual = (samples - point[:, None]).abs()
        truth = (target - point).abs()
    elif metric == "position":
        residual = torch.linalg.vector_norm(samples - point[:, None, :], dim=-1)
        truth = torch.linalg.vector_norm(target - point, dim=-1)
    else:
        means = torch.nn.functional.normalize(point, dim=-1)
        residual = torch.rad2deg(torch.acos((torch.nn.functional.normalize(samples, dim=-1) * means[:, None]).sum(-1).clamp(-1, 1)))
        truth = torch.rad2deg(torch.acos((torch.nn.functional.normalize(target, dim=-1) * means).sum(-1).clamp(-1, 1)))
    q68 = torch.quantile(residual, .68, dim=1)
    q95 = torch.quantile(residual, .95, dim=1)
    return truth <= q68, truth <= q95


@torch.no_grad()
def _predict_model(checkpoint_path: Path, data_dir: Path, *, batch_size=256, device="cuda:0", draws=256):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = _load_manifest(data_dir / "manifest.json")
    expected = __import__("hashlib").sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if expected != checkpoint["data_manifest_sha256"]:
        raise ValueError("checkpoint was trained from a different dataset manifest")
    target_scaler = checkpoint["target_scaler"]
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is unavailable")
    cfg = checkpoint["model_config"]
    detector_names = tuple(cfg.get("detector_names", ("ttc", "scifi", "ecal")))
    model = PhaseIModel(cfg["channel_counts"], cfg["use_experts"], detector_names)
    model.load_state_dict(checkpoint["model_state"]); model.to(dev).eval()
    rng = torch.Generator(device=dev).manual_seed(61471)
    output = {key: [] for key in ("particle", "evt_id", "pid", "pid_prob", "energy", "energy_nll", "energy_cov68", "energy_cov95", "position", "position_nll", "position_cov68", "position_cov95", "direction", "direction_nll", "direction_cov68", "direction_cov95", "response_mask", "t0", "Ek0", "x0", "p0", "theta", "phi")}
    for particle in manifest["particles"]:
        name = particle["particle"]
        directory = data_dir / name
        indices = _select_indices(directory, 2, None, 42)
        data = EventData(directory, indices, target_scaler, detector_names)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(_collate, target_scaler=target_scaler,
                                               detector_names=detector_names), pin_memory=dev.type == "cuda")
        evt_ids = np.load(directory / "evt_id.npy", mmap_mode="r")[indices]
        scalar = target_scaler["energy"]
        pscale = target_scaler["position"]
        cursor = 0
        for batch, target in loader:
            batch, target = _move(batch, dev), _move(target, dev)
            pred = model(batch)
            n = len(target["pid"])
            e_par = pred["energy"]
            e_logits = e_par[:, :5]
            e_mu = e_par[:, 5:10] * scalar["std"] + scalar["mean"]
            e_sig = e_par[:, 10:15].clamp(-7, 5).exp() * scalar["std"]
            e_point = (e_logits.softmax(-1) * e_mu).sum(-1)
            x_par = pred["position"]
            x_logits = x_par[:, :4]
            x_mu_std = x_par[:, 4:16].reshape(-1, 4, 3)
            x_sig_std = x_par[:, 16:28].reshape(-1, 4, 3).clamp(-7, 5).exp()
            x_mean = torch.as_tensor(pscale["mean"], device=dev, dtype=torch.float32)
            x_std = torch.as_tensor(pscale["std"], device=dev, dtype=torch.float32)
            x_mu = x_mu_std * x_std[None, None, :] + x_mean[None, None, :]
            x_sig = x_sig_std * x_std[None, None, :]
            x_point = (x_logits.softmax(-1)[..., None] * x_mu).sum(1)
            d_par = pred["direction"]
            d_mean = torch.nn.functional.normalize(d_par[:, 2:8].reshape(-1, 2, 3), dim=-1)
            d_point = torch.nn.functional.normalize((d_par[:, :2].softmax(-1)[..., None] * d_mean).sum(1), dim=-1)
            loss = losses(pred, target)
            e_nll = loss["energy"] + math.log(float(scalar["std"]))
            x_nll = loss["position"] + float(np.log(np.asarray(pscale["std"])).sum())
            d_nll = loss["direction"]
            e_samples = _sample_energy(e_par, draws, rng) * scalar["std"] + scalar["mean"]
            x_samples = _sample_position(x_par, draws, rng) * x_std[None, None, :] + x_mean[None, None, :]
            d_samples = _sample_direction(d_par, draws, rng)
            e_cov68, e_cov95 = _coverage_from_samples(e_samples, target["energy"]*scalar["std"]+scalar["mean"], e_point, "energy")
            x_truth = target["position"]*x_std+x_mean
            x_cov68, x_cov95 = _coverage_from_samples(x_samples, x_truth, x_point, "position")
            d_cov68, d_cov95 = _coverage_from_samples(d_samples, target["direction"], d_point, "direction")
            truth = {k: v.detach().cpu().numpy() for k, v in target.items() if k in ("pid", "energy", "position", "direction")}
            response = batch["response_mask"].detach().cpu().numpy().astype(np.uint8)
            direction = truth["direction"]
            theta = np.arccos(np.clip(direction[:, 2], -1, 1))
            phi = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2*math.pi)
            def append(key, value):
                output[key].append(np.asarray(value))
            append("particle", np.full(n, name))
            append("evt_id", evt_ids[cursor:cursor+n]); cursor += n
            append("pid", truth["pid"]); append("pid_prob", pred["pid"].softmax(-1).cpu().numpy())
            append("energy", np.column_stack([e_point.cpu().numpy(), e_mu.cpu().numpy(), e_sig.cpu().numpy(), e_logits.cpu().numpy()]))
            append("energy_nll", e_nll.cpu().numpy()); append("energy_cov68", e_cov68.cpu().numpy()); append("energy_cov95", e_cov95.cpu().numpy())
            append("position", np.column_stack([x_point.cpu().numpy(), x_mu.reshape(n, -1).cpu().numpy(), x_sig.reshape(n, -1).cpu().numpy(), x_logits.cpu().numpy()]))
            append("position_nll", x_nll.cpu().numpy()); append("position_cov68", x_cov68.cpu().numpy()); append("position_cov95", x_cov95.cpu().numpy())
            append("direction", np.column_stack([d_point.cpu().numpy(), d_par[:, :].cpu().numpy()]))
            append("direction_nll", d_nll.cpu().numpy()); append("direction_cov68", d_cov68.cpu().numpy()); append("direction_cov95", d_cov95.cpu().numpy())
            append("response_mask", response)
            for key in ("t0", "Ek0", "x0", "p0"):
                arr = np.load(directory / f"{key}.npy", mmap_mode="r")[indices[cursor-n:cursor]]
                append(key, arr)
            append("theta", theta); append("phi", phi)
    return {key: np.concatenate(vals, axis=0) if vals else np.empty(0) for key, vals in output.items()}, checkpoint


def _bootstrap_ci(values: np.ndarray, metric, *, seed=42, reps=300):
    values = np.asarray(values)
    if len(values) < 100:
        return [None, None]
    rng = np.random.default_rng(seed)
    stats = np.empty(reps, dtype=np.float64)
    chunk = max(1, min(reps, 8_000_000 // len(values)))
    start = 0
    while start < reps:
        k = min(chunk, reps-start)
        ids = rng.integers(0, len(values), size=(k, len(values)))
        stats[start:start+k] = metric(values[ids], axis=1)
        start += k
    lo, hi = np.quantile(stats[np.isfinite(stats)], [.025, .975])
    return [float(lo), float(hi)]


def _metric_table(pred: dict, output_dir: Path, *, seed=42):
    labels = pred["pid"].astype(int)
    names = pred["particle"].astype(str)
    energy_truth = pred["Ek0"].astype(float)
    energy_hat = pred["energy"][:, 0]
    energy_res = energy_hat - energy_truth
    energy_abs = np.abs(energy_res)
    position_hat = pred["position"][:, :3]
    position_err = np.linalg.norm(position_hat - pred["x0"], axis=1)
    direction_hat = pred["direction"][:, :3]
    direction_hat = direction_hat / np.maximum(np.linalg.norm(direction_hat, axis=1, keepdims=True), 1e-12)
    direction_truth = pred["p0"] / np.maximum(np.linalg.norm(pred["p0"], axis=1, keepdims=True), 1e-12)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(direction_hat*direction_truth, axis=1), -1, 1)))
    probabilities = pred["pid_prob"]
    predicted = probabilities.argmax(1)
    def groups():
        yield "all", np.ones(len(labels), dtype=bool)
        for particle in sorted(set(names)):
            yield f"particle:{particle}", names == particle
        for i in range(8):
            yield f"response_mask:{i}", pred["response_mask"] == i
        e_bins = [0., 1., 5., 10., 20., 40., 60.0001]
        for i, (a,b) in enumerate(zip(e_bins[:-1],e_bins[1:])):
            yield f"energy_mev:{a:g}-{b:g}", (energy_truth >= a) & (energy_truth < b)
        theta = pred["theta"]
        t_edges = np.linspace(0, math.pi + 1e-8, 5)
        for i,(a,b) in enumerate(zip(t_edges[:-1],t_edges[1:])):
            yield f"theta_rad:{a:.3f}-{b:.3f}", (theta >= a) & (theta < b)
        phi_edges = np.linspace(0,2*math.pi+1e-8,9)
        for i,(a,b) in enumerate(zip(phi_edges[:-1],phi_edges[1:])):
            yield f"phi_rad:{a:.3f}-{b:.3f}", (pred["phi"] >= a) & (pred["phi"] < b)
        radius=np.linalg.norm(pred["x0"][:,:2],axis=1)
        r_edges=(0.,10.,20.,30.0001)
        for i,(a,b) in enumerate(zip(r_edges[:-1],r_edges[1:])):
            yield f"x0_radius_mm:{a:g}-{b:g}",(radius>=a)&(radius<b)
        z=pred["x0"][:,2]
        z_edges=(-30.,-10.,10.,30.0001)
        for a,b in zip(z_edges[:-1],z_edges[1:]):
            yield f"x0_z_mm:{a:g}-{b:g}",(z>=a)&(z<b)
        for particle in sorted(set(names)):
            for mask in range(8):
                yield f"particle:{particle}|mask:{mask}", (names == particle) & (pred["response_mask"] == mask)
    table = {}
    for group, select in groups():
        n = int(select.sum())
        if not n:
            continue
        y, phat = labels[select], predicted[select]
        probs = probabilities[select]
        correct = (y == phat).astype(float)
        confusion = np.zeros((2,2),dtype=int)
        np.add.at(confusion,(y,phat),1)
        onehot = np.eye(2)[y]
        brier = np.sum((probs-onehot)**2,axis=1)
        ece = 0.
        conf, cls = probs.max(1), phat
        for lo in np.linspace(0,1,11)[:-1]:
            bucket = (conf >= lo) & (conf < lo+.1 if lo < .9 else conf <= 1)
            if bucket.any(): ece += bucket.mean()*abs(correct[bucket].mean()-conf[bucket].mean())
        row = {"n": n, "statistically_small": n < 100,
               "pid_accuracy": float(correct.mean()), "pid_accuracy_ci95": _bootstrap_ci(correct, np.mean, seed=seed),
               "pid_brier_ci95": _bootstrap_ci(brier,np.mean,seed=seed+1),
               "pid_brier": float(brier.mean()), "pid_ece_10bin": float(ece), "pid_confusion_true_rows": confusion.tolist(),
               "energy_bias_mev": float(energy_res[select].mean()),
               "energy_bias_ci95": _bootstrap_ci(energy_res[select], np.mean, seed=seed),
               "energy_abs_residual_68_mev": float(np.quantile(energy_abs[select],.68)),
               "energy_abs68_ci95": _bootstrap_ci(energy_abs[select], lambda a,axis: np.quantile(a,.68,axis=axis), seed=seed),
               "energy_nll": float(pred["energy_nll"][select].mean()),
               "energy_nll_ci95": _bootstrap_ci(pred["energy_nll"][select],np.mean,seed=seed+2),
               "energy_coverage_68": float(pred["energy_cov68"][select].mean()), "energy_coverage_95": float(pred["energy_cov95"][select].mean()),
               "energy_coverage_68_ci95": _bootstrap_ci(pred["energy_cov68"][select],np.mean,seed=seed+3),
               "energy_coverage_95_ci95": _bootstrap_ci(pred["energy_cov95"][select],np.mean,seed=seed+4),
               "x0_error_median_mm": float(np.median(position_err[select])),
               "x0_error_68_mm": float(np.quantile(position_err[select],.68)),
               "x0_error_68_ci95": _bootstrap_ci(position_err[select],lambda a,axis:np.quantile(a,.68,axis=axis),seed=seed+5),
               "x0_rmse_mm": float(np.sqrt(np.mean(position_err[select]**2))),
               "x0_rmse_ci95": _bootstrap_ci(position_err[select],lambda a,axis:np.sqrt(np.mean(a*a,axis=axis)),seed=seed+6),
               "position_nll_mm": float(pred["position_nll"][select].mean()),
               "position_nll_ci95": _bootstrap_ci(pred["position_nll"][select],np.mean,seed=seed+7),
               "position_coverage_68": float(pred["position_cov68"][select].mean()), "position_coverage_95": float(pred["position_cov95"][select].mean()),
               "position_coverage_68_ci95": _bootstrap_ci(pred["position_cov68"][select],np.mean,seed=seed+8),
               "position_coverage_95_ci95": _bootstrap_ci(pred["position_cov95"][select],np.mean,seed=seed+9),
               "direction_angle_median_deg": float(np.median(angle[select])),
               "direction_angle_68_deg": float(np.quantile(angle[select],.68)),
               "direction_angle_68_ci95": _bootstrap_ci(angle[select],lambda a,axis:np.quantile(a,.68,axis=axis),seed=seed+10),
               "direction_nll_sr": float(pred["direction_nll"][select].mean()),
               "direction_nll_ci95": _bootstrap_ci(pred["direction_nll"][select],np.mean,seed=seed+11),
               "direction_coverage_68": float(pred["direction_cov68"][select].mean()), "direction_coverage_95": float(pred["direction_cov95"][select].mean())}
        row["direction_coverage_68_ci95"]=_bootstrap_ci(pred["direction_cov68"][select],np.mean,seed=seed+12)
        row["direction_coverage_95_ci95"]=_bootstrap_ci(pred["direction_cov95"][select],np.mean,seed=seed+13)
        if n < 100:
            for key in ("pid_accuracy_ci95", "energy_bias_ci95", "energy_abs68_ci95"):
                row[key] = None
            for key in ("pid_brier_ci95","energy_nll_ci95","energy_coverage_68_ci95","energy_coverage_95_ci95",
                        "x0_error_68_ci95","x0_rmse_ci95","position_nll_ci95","position_coverage_68_ci95",
                        "position_coverage_95_ci95","direction_angle_68_ci95","direction_nll_ci95",
                        "direction_coverage_68_ci95","direction_coverage_95_ci95"):
                row[key]=None
        table[group] = row
    efficiency = {}
    for detector, bit in (("ttc",1),("scifi",2),("ecal",4)):
        for particle in sorted(set(names)):
            m = names == particle
            hit=((pred["response_mask"] & bit) != 0)[m].astype(float)
            efficiency[f"{particle}:{detector}"] = {"generated_events": int(m.sum()), "events_with_hit": int(hit.sum()),
                                                      "efficiency": float(hit.mean()),"efficiency_ci95":_bootstrap_ci(hit,np.mean,seed=bit+42)}
    zero=(pred["response_mask"]==0).astype(float); any_hit=(pred["response_mask"]!=0).astype(float)
    efficiency["all_zero"] = {"events": int(zero.sum()), "fraction": float(zero.mean()),"fraction_ci95":_bootstrap_ci(zero,np.mean)}
    efficiency["any_hit"] = {"events": int(any_hit.sum()), "fraction": float(any_hit.mean()),"fraction_ci95":_bootstrap_ci(any_hit,np.mean)}
    return {"test_events": len(labels), "grouped_metrics": table, "generated_event_efficiency": efficiency,
            "position_target": "SimPrimaryVertex.x0 (target-internal primary vertex), not detector entry"}


def fit_baselines(data_dir: Path, output_dir: Path):
    """Fit all baseline parameters from train split only, then score fixed test events."""
    manifest = _load_manifest(data_dir / "manifest.json")
    train_rows = []; test_rows=[]
    for p in manifest["particles"]:
        name=p["particle"]; d=data_dir/name
        for split_id, store in ((0,train_rows),(2,test_rows)):
            idx=_select_indices(d,split_id,None,42)
            response=np.load(d/"response_mask.npy",mmap_mode="r")[idx]
            pid=np.load(d/"pid.npy",mmap_mode="r")[idx]
            energy=np.load(d/"Ek0.npy",mmap_mode="r")[idx]
            pos=np.load(d/"x0.npy",mmap_mode="r")[idx]
            p0=np.load(d/"p0.npy",mmap_mode="r")[idx]
            offsets=np.load(d/"ecal_offsets.npy",mmap_mode="r"); edep=np.load(d/"ecal_Edep.npy",mmap_mode="r")
            sums=np.asarray([edep[offsets[i]:offsets[i+1]].sum() for i in idx],dtype=np.float64)
            evt=np.load(d/"evt_id.npy",mmap_mode="r")[idx]
            store.append({"particle":np.full(len(idx),name),"evt_id":evt,"pid":pid,"response_mask":response,"energy":energy,"position":pos,"p0":p0,"ecal_sum":sums})
    train={k:np.concatenate([row[k] for row in train_rows]) for k in train_rows[0]}
    test={k:np.concatenate([row[k] for row in test_rows]) for k in test_rows[0]}
    pid_prob=np.tile(np.bincount(train["pid"].astype(int),minlength=2)/len(train["pid"]),(len(test["pid"]),1))
    mask_prob=np.zeros((8,2),dtype=float)
    for mask in range(8):
        rows=train["response_mask"]==mask
        mask_prob[mask]=(np.bincount(train["pid"][rows].astype(int),minlength=2)+1)/(rows.sum()+2)
    energy_fit=np.polyfit(train["ecal_sum"],train["energy"],1)
    energy_mean=float(train["energy"].mean()); energy_std=max(float(train["energy"].std()),1e-8)
    pos_mean=train["position"].mean(axis=0)
    direction=train["p0"]/np.maximum(np.linalg.norm(train["p0"],axis=1,keepdims=True),1e-12)
    dir_mean=direction.mean(axis=0); dir_mean=dir_mean/max(np.linalg.norm(dir_mean),1e-12)
    prediction={"pid_prior":np.broadcast_to(pid_prob,(len(pid_prob),2)).copy(),
                "pid_mask_lookup":mask_prob[test["response_mask"].astype(int)],
                "ecal_sum_raw":test["ecal_sum"],
                "ecal_sum_affine":energy_fit[0]*test["ecal_sum"]+energy_fit[1],
                "energy_prior_mean":np.full(len(test["energy"]),energy_mean),
                "energy_prior_std":np.full(len(test["energy"]),energy_std),
                "x0_train_mean":np.broadcast_to(pos_mean,test["position"].shape).copy(),
                "direction_train_mean":np.broadcast_to(dir_mean,test["p0"].shape).copy()}
    # Include both keys in returned arrays for reproducible, species-balanced scoring.
    prediction.update({"particle":test["particle"],"evt_id":test["evt_id"],"pid":test["pid"],"response_mask":test["response_mask"],"energy_truth":test["energy"],"position_truth":test["position"],"p0_truth":test["p0"]})
    return prediction


def evaluate(checkpoint_path: str | Path, data_dir: str | Path, *, batch_size=256, device="cuda:0", output_dir: str | Path | None=None):
    checkpoint_path=Path(checkpoint_path).resolve(); data_dir=Path(data_dir).resolve()
    pred, checkpoint = _predict_model(checkpoint_path,data_dir,batch_size=batch_size,device=device)
    outdir=Path(output_dir).resolve() if output_dir else checkpoint_path.parent
    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outdir/"test_predictions.npz",**pred)
    metrics=_metric_table(pred,outdir)
    baseline=fit_baselines(data_dir,outdir)
    np.savez_compressed(outdir/"baseline_predictions.npz",**baseline)
    # Point metrics on the same test events; calibration parameters came from train only.
    baseline_rows={}
    for key in ("ecal_sum_raw","ecal_sum_affine"):
        residual=baseline[key]-baseline["energy_truth"]
        absolute=np.abs(residual)
        baseline_rows[key]={"energy_bias_mev":float(residual.mean()),"energy_bias_ci95":_bootstrap_ci(residual,np.mean),
                            "energy_abs68_mev":float(np.quantile(absolute,.68)),"energy_abs68_ci95":_bootstrap_ci(absolute,lambda a,axis:np.quantile(a,.68,axis=axis)),
                            "energy_rmse_mev":float(np.sqrt(np.mean(residual**2))),"energy_rmse_ci95":_bootstrap_ci(residual,lambda a,axis:np.sqrt(np.mean(a*a,axis=axis)))}
    residual=baseline["energy_prior_mean"]-baseline["energy_truth"]
    absolute=np.abs(residual)
    baseline_rows["training_energy_prior"]={"energy_bias_mev":float(residual.mean()),"energy_bias_ci95":_bootstrap_ci(residual,np.mean),
        "energy_abs68_mev":float(np.quantile(absolute,.68)),"energy_abs68_ci95":_bootstrap_ci(absolute,lambda a,axis:np.quantile(a,.68,axis=axis)),
        "energy_rmse_mev":float(np.sqrt(np.mean(residual**2))),"energy_rmse_ci95":_bootstrap_ci(residual,lambda a,axis:np.sqrt(np.mean(a*a,axis=axis))),
        "nll_gaussian_mev":float(np.mean(.5*((residual/baseline["energy_prior_std"])**2+np.log(2*np.pi*baseline["energy_prior_std"]**2))))}
    for key in ("pid_prior","pid_mask_lookup"):
        probs=baseline[key]; y=baseline["pid"].astype(int); guess=probs.argmax(1)
        correct=(guess==y).astype(float); brier=np.sum((probs-np.eye(2)[y])**2,axis=1)
        baseline_rows[key]={"pid_accuracy":float(np.mean(correct)),"pid_accuracy_ci95":_bootstrap_ci(correct,np.mean),
                            "pid_brier":float(np.mean(brier)),"pid_brier_ci95":_bootstrap_ci(brier,np.mean)}
    pos_err=np.linalg.norm(baseline["x0_train_mean"]-baseline["position_truth"],axis=1)
    dir_true=baseline["p0_truth"]/np.maximum(np.linalg.norm(baseline["p0_truth"],axis=1,keepdims=True),1e-12)
    dir_pred=baseline["direction_train_mean"][0]/max(np.linalg.norm(baseline["direction_train_mean"][0]),1e-12)
    angles=np.rad2deg(np.arccos(np.clip(dir_true@dir_pred,-1,1)))
    baseline_rows["train_priors"]={"x0_error_68_mm":float(np.quantile(pos_err,.68)),
        "x0_error_68_ci95":_bootstrap_ci(pos_err,lambda a,axis:np.quantile(a,.68,axis=axis)),
        "direction_angle_68_deg":float(np.quantile(angles,.68)),
        "direction_angle_68_ci95":_bootstrap_ci(angles,lambda a,axis:np.quantile(a,.68,axis=axis))}
    metrics["baselines"]=baseline_rows
    metrics["checkpoint"]={"path":str(checkpoint_path),"best_epoch":checkpoint.get("best_epoch"),"training_config":checkpoint.get("training_config"),"data_manifest_sha256":checkpoint["data_manifest_sha256"]}
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    lines=["# Phase I held-out evaluation", "", "Idealized SimHit study; position target is the target-internal primary vertex.", "", f"Test events: {metrics['test_events']}", "", "## Aggregate groups", "", "| Group | N | PID acc | E bias (MeV) | E abs68 (MeV) | x0 error68 (mm) | angle68 (deg) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group,row in metrics["grouped_metrics"].items():
        if group == "all" or group.startswith("particle:") or group.startswith("response_mask:"):
            lines.append(f"| {group} | {row['n']} | {row['pid_accuracy']:.4f} | {row['energy_bias_mev']:.4g} | {row['energy_abs_residual_68_mev']:.4g} | {row['x0_error_68_mm']:.4g} | {row['direction_angle_68_deg']:.4g} |")
    lines.extend(["", "## Baselines", "", "```json", json.dumps(baseline_rows,indent=2), "```", "", "Small groups (N < 100) are descriptive only; their intervals are not used for model selection."])
    (outdir/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"metrics":str(outdir/"metrics.json"),"report":str(outdir/"report.md"),"predictions":str(outdir/"test_predictions.npz"),"test_events":metrics["test_events"]},sort_keys=True))
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path); parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--batch-size",type=int,default=256); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    evaluate(args.checkpoint,args.data,batch_size=args.batch_size,device=args.device,output_dir=args.output_dir)
    return 0


if __name__=="__main__": raise SystemExit(main())
