"""Detector-specific gated graph networks and probabilistic event heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SPEC = {
    "scifi": {"channels": 2360, "embedding": 16, "width": 160, "blocks": 4, "features": 9},
    "ttc": {"channels": 108, "embedding": 8, "width": 96, "blocks": 2, "features": 8},
    "ecal": {"channels": 488, "embedding": 8, "width": 160, "blocks": 3, "features": 9},
}


class GatedEdgeBlock(nn.Module):
    def __init__(self, width: int, edge_features: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width + edge_features, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.gate = nn.Linear(2 * width + edge_features, width)
        self.update = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                    nn.Linear(2 * width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        pair = torch.cat((h[src], h[dst], edge_attr), dim=-1)
        message = self.message(pair) * torch.sigmoid(self.gate(pair))
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, dst, message)
        degree = torch.zeros((len(h), 1), dtype=h.dtype, device=h.device)
        degree.index_add_(0, dst, torch.ones((len(dst), 1), dtype=h.dtype, device=h.device))
        aggregate = aggregate / degree.clamp_min(1).sqrt()
        return self.norm(h + self.update(aggregate))


class DetectorGraphEncoder(nn.Module):
    def __init__(self, detector: str):
        super().__init__()
        spec = SPEC[detector]
        self.detector = detector
        self.embedding = nn.Embedding(spec["channels"], spec["embedding"])
        self.input = nn.Sequential(nn.Linear(spec["features"] + spec["embedding"], spec["width"]),
                                   nn.SiLU(), nn.LayerNorm(spec["width"]))
        self.blocks = nn.ModuleList(GatedEdgeBlock(spec["width"], spec["features"])
                                    for _ in range(spec["blocks"]))
        self.key = nn.Linear(spec["width"], spec["width"], bias=False)
        self.queries = nn.Parameter(torch.randn(2, spec["width"]) / math.sqrt(spec["width"]))
        self.empty = nn.Parameter(torch.zeros(2, spec["width"]))
        self.project = nn.Sequential(nn.Linear(spec["width"], 256), nn.LayerNorm(256))

    def forward(self, graph: dict[str, torch.Tensor], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        channels = graph["channel"].long()
        h = self.input(torch.cat((graph["x"], self.embedding(channels)), dim=-1))
        for block in self.blocks:
            h = block(h, graph["edge_index"], graph["edge_attr"])
        batch = graph["batch"].long()
        score = self.key(h) @ self.queries.T / math.sqrt(h.shape[-1])
        expanded_batch = batch[:, None].expand(-1, 2)
        maxima = torch.full((batch_size, 2), -torch.inf, dtype=h.dtype, device=h.device)
        if len(score):
            maxima.scatter_reduce_(0, expanded_batch, score, reduce="amax", include_self=True)
            weight = torch.exp(score - maxima[batch])
            denom = torch.zeros((batch_size, 2), dtype=h.dtype, device=h.device)
            denom.index_add_(0, batch, weight)
            pooled = torch.zeros((batch_size, 2, h.shape[-1]), dtype=h.dtype, device=h.device)
            pooled.index_add_(0, batch, weight.unsqueeze(-1) * h.unsqueeze(1))
            pooled = pooled / denom.clamp_min(1e-12).unsqueeze(-1)
            has = denom[:, 0] > 0
            pooled = torch.where(has[:, None, None], pooled, self.empty[None, :, :])
        else:
            has = torch.zeros(batch_size, dtype=torch.bool, device=h.device)
            pooled = self.empty[None, :, :].expand(batch_size, -1, -1)
        return self.project(pooled), has


class PhaseIGNN(nn.Module):
    """Sparse three-detector GNN with a seven-token transformer fusion stage."""

    def __init__(self):
        super().__init__()
        self.detectors = nn.ModuleDict({name: DetectorGraphEncoder(name) for name in ("scifi", "ttc", "ecal")})
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.token_type = nn.Parameter(torch.randn(1, 7, 256) * 0.02)
        self.zero_prior = nn.Parameter(torch.zeros(256))
        self.pid = nn.Linear(256, 2)
        self.energy = nn.Linear(256, 15)
        self.position = nn.Linear(256, 40)
        self.direction = nn.Linear(256, 20)

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch["scifi"]["counts"])
        pooled = []
        has = []
        for name in ("scifi", "ttc", "ecal"):
            tokens, present = self.detectors[name](batch[name], batch_size)
            pooled.append(tokens)
            has.append(present)
        tokens = torch.cat([self.class_token.expand(batch_size, -1, -1), *pooled], dim=1)
        latent = self.fusion(tokens + self.token_type)[:, 0]
        zero_hit = ~torch.stack(has, dim=1).any(1)
        latent = torch.where(zero_hit[:, None], self.zero_prior[None], latent)
        return {"pid": self.pid(latent), "energy": self.energy(latent),
                "position": self.position(latent), "direction": self.direction(latent),
                "is_zero": zero_hit}


def losses(output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"pid": F.cross_entropy(output["pid"], target["pid"], reduction="none"),
            "energy": _energy_nll(output["energy"], target["energy"]),
            "position": _position_nll(output["position"], target["position"]),
            "direction": _direction_nll(output["direction"], target["direction"])}


def _energy_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, mean, logstd = params.split(5, dim=-1)
    logstd = logstd.clamp(-7, 5)
    diff = (target[:, None] - mean) * torch.exp(-logstd)
    component = -0.5 * (diff.square() + 2 * logstd + math.log(2 * math.pi))
    return -torch.logsumexp(F.log_softmax(logits, -1) + component, dim=-1)


def _position_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch = len(params)
    logits = params[:, :4]
    means = params[:, 4:16].reshape(batch, 4, 3)
    raw = params[:, 16:].reshape(batch, 4, 6)
    lower = torch.zeros((batch, 4, 3, 3), dtype=params.dtype, device=params.device)
    lower[..., 0, 0] = F.softplus(raw[..., 0]) + 0.03
    lower[..., 1, 0] = raw[..., 1]
    lower[..., 1, 1] = F.softplus(raw[..., 2]) + 0.03
    lower[..., 2, 0] = raw[..., 3]
    lower[..., 2, 1] = raw[..., 4]
    lower[..., 2, 2] = F.softplus(raw[..., 5]) + 0.03
    residual = target[:, None, :] - means
    whitened = torch.linalg.solve_triangular(lower, residual.unsqueeze(-1), upper=False).squeeze(-1)
    logdet = torch.log(torch.diagonal(lower, dim1=-2, dim2=-1)).sum(-1)
    component = -0.5 * whitened.square().sum(-1) - logdet - 1.5 * math.log(2 * math.pi)
    return -torch.logsumexp(F.log_softmax(logits, -1) + component, dim=-1)


def _direction_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch = len(params)
    logits = params[:, :4]
    mean = F.normalize(params[:, 4:16].reshape(batch, 4, 3), dim=-1, eps=1e-8)
    kappa = F.softplus(params[:, 16:20]).clamp(1e-5, 100.)
    dot = (mean * target[:, None]).sum(-1).clamp(-1, 1)
    log_sinh = torch.where(kappa < 1e-3,
                           torch.log(kappa.clamp_min(1e-30)) + torch.log1p(kappa.square() / 6),
                           kappa - math.log(2.) + torch.log1p(-torch.exp(-2 * kappa)))
    log_c = torch.log(kappa) - math.log(4 * math.pi) - log_sinh
    return -torch.logsumexp(F.log_softmax(logits, -1) + log_c + kappa * dot, dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
