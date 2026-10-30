"""Compact detector-wise hit encoder and probabilistic event reconstruction heads."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DetectorEncoder(nn.Module):
    def __init__(self, n_channels: int, embedding_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.channel = nn.Embedding(max(1, n_channels), embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.pool = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.empty = nn.Parameter(torch.zeros(hidden))

    def forward(self, hit: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        channel = hit["channel"].clamp(min=0, max=self.channel.num_embeddings - 1)
        x = torch.cat([self.channel(channel), hit["t"].unsqueeze(-1), hit["log_edep"].unsqueeze(-1)], -1)
        z = self.mlp(x)
        mask = hit["mask"].unsqueeze(-1)
        count = mask.sum(1).clamp_min(1.0)
        mean = (z * mask).sum(1) / count
        maximum = z.masked_fill(~mask.bool(), -1e4).max(1).values
        energy = hit["weight"].unsqueeze(-1) * mask
        weighted = (z * energy).sum(1) / energy.sum(1).clamp_min(1e-8)
        pooled = self.pool(torch.cat([mean, maximum, weighted], -1))
        has = hit["mask"].any(1)
        pooled = torch.where(has.unsqueeze(-1), pooled, self.empty.unsqueeze(0))
        return pooled, has


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128),
                                 nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))

    def forward(self, x):
        return self.net(x)


class PhaseIModel(nn.Module):
    def __init__(self, channel_counts: dict[str, int], use_experts: bool = True,
                 detector_names: tuple[str, ...] = ("ttc", "scifi", "ecal")):
        super().__init__()
        if not detector_names or any(name not in ("ttc", "scifi", "ecal") for name in detector_names):
            raise ValueError("detector_names must select one or more Phase-I detectors")
        self.detector_names = tuple(detector_names)
        self.detectors = nn.ModuleDict({name: DetectorEncoder(channel_counts[name]) for name in self.detector_names})
        fusion_width = 67 * len(self.detector_names) + 2 ** len(self.detector_names)
        self.fusion = nn.Sequential(nn.Linear(fusion_width, 128), nn.SiLU(), nn.LayerNorm(128),
                                    nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))
        self.zero_prior = nn.Parameter(torch.zeros(128))
        self.use_experts = use_experts
        if use_experts:
            self.router = nn.Linear(128, 3)
            self.experts = nn.ModuleList([Expert() for _ in range(3)])
        self.pid = nn.Linear(128, 2)
        self.energy = nn.Linear(128, 15)  # 5 logits, means, log-scales
        self.position = nn.Linear(128, 28)  # 4 logits, means(3), log-scales(3)
        self.direction = nn.Linear(128, 10)  # 2 logits, unit-vector parameters(3), kappa

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        vectors = []
        flags = []
        stats = []
        for name in self.detector_names:
            hit = batch[name]
            pooled, has = self.detectors[name](hit)
            vectors.append(pooled)
            flags.append(has)
            n = hit["mask"].sum(1).float()
            total_energy = (hit["weight"] * hit["mask"]).sum(1)
            time_mean = (hit["t"] * hit["mask"]).sum(1) / n.clamp_min(1.0)
            stats.extend([torch.log1p(n).unsqueeze(1), torch.log1p(total_energy).unsqueeze(1), time_mean.unsqueeze(1)])
        mask_id = sum(flag.long() * (1 << i) for i, flag in enumerate(flags))
        onehot = F.one_hot(mask_id, 2 ** len(self.detector_names)).float()
        latent = self.fusion(torch.cat(vectors + stats + [onehot], dim=1))
        is_zero = ~torch.stack(flags, 1).any(1)
        if self.use_experts:
            gate = self.router(latent).softmax(-1)
            expert_states = torch.stack([expert(latent) for expert in self.experts], 1)
            mixed = (expert_states * gate.unsqueeze(-1)).sum(1)
            latent = torch.where(is_zero[:, None], self.zero_prior[None, :], mixed)
        else:
            latent = torch.where(is_zero[:, None], self.zero_prior[None, :], latent)
        return {"pid": self.pid(latent), "energy": self.energy(latent),
                "position": self.position(latent), "direction": self.direction(latent),
                "is_zero": is_zero}


def _mixture_nll(params: torch.Tensor, target: torch.Tensor, components: int, dims: int) -> torch.Tensor:
    logits = params[:, :components]
    mu_end = components + components * dims
    means = params[:, components:mu_end].reshape(-1, components, dims)
    logstd = params[:, mu_end:mu_end + components * dims].reshape(-1, components, dims).clamp(-7, 5)
    diff = (target[:, None, :] - means) * torch.exp(-logstd)
    logp = -.5 * (diff.square() + 2 * logstd + math.log(2 * math.pi)).sum(-1)
    return -torch.logsumexp(F.log_softmax(logits, -1) + logp, -1)


def _vmf_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = params[:, :2]
    means = F.normalize(params[:, 2:8].reshape(-1, 2, 3), dim=-1, eps=1e-8)
    kappa = F.softplus(params[:, 8:10]).clamp(1e-5, 100.0)
    dot = (means * target[:, None, :]).sum(-1).clamp(-1, 1)
    # 3D vMF: C(k)=k/(4*pi*sinh(k)); stable small- and large-k evaluation.
    log_sinh = torch.where(kappa < 1e-3,
                           torch.log(kappa.clamp_min(1e-30)) + torch.log1p(kappa.square() / 6),
                           kappa - math.log(2.0) + torch.log1p(-torch.exp(-2 * kappa)))
    log_c = torch.log(kappa) - math.log(4 * math.pi) - log_sinh
    component = log_c + kappa * dot
    return -torch.logsumexp(F.log_softmax(logits, -1) + component, -1)


def losses(output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"pid": F.cross_entropy(output["pid"], target["pid"], reduction="none"),
            "energy": _mixture_nll(output["energy"], target["energy"].reshape(-1, 1), 5, 1),
            "position": _mixture_nll(output["position"], target["position"], 4, 3),
            "direction": _vmf_nll(output["direction"], target["direction"])}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
