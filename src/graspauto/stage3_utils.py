"""Stage 3 metric-retrieval helpers.

Minimal landing for the 2026-04-09 pipeline reset:
- projection heads for query/code metric space
- multi-positive contrastive loss over oracle / soft positives
- lightweight latent-space sampler used only to refine top-k retrieval
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MultiPositiveContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        query_embed: torch.Tensor,
        code_embed: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
        scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        logits = torch.matmul(F.normalize(query_embed, dim=-1), F.normalize(code_embed, dim=-1).t())
        if scale is not None:
            logits = logits * scale
        else:
            logits = logits / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)

        if positive_weights is None:
            positive_weights = torch.ones_like(positive_ids, dtype=log_probs.dtype, device=log_probs.device)
        else:
            positive_weights = positive_weights.to(log_probs.dtype)
        positive_weights = positive_weights / positive_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)

        gathered = log_probs.gather(dim=1, index=positive_ids.long())
        return -(gathered * positive_weights).sum(dim=-1).mean()


def build_assembly_aware_positive_weights(
    positive_errors_mm: torch.Tensor,
    base_weights: torch.Tensor | None = None,
    temp_mm: float = 10.0,
    min_weight: float = 1e-4,
) -> torch.Tensor:
    """Turn post-assembly quality into soft positive weights."""

    err = positive_errors_mm.float()
    quality = torch.softmax(-(err - err.amin(dim=-1, keepdim=True)) / max(float(temp_mm), 1e-4), dim=-1)
    if base_weights is not None:
        weights = quality * base_weights.float().clamp_min(0.0)
    else:
        weights = quality
    weights = weights.clamp_min(float(min_weight))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)


class AssemblyAwareListMLELoss(nn.Module):
    """Lightweight listwise ranking loss over oracle positives only."""

    def forward(
        self,
        logits: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        gathered = logits.gather(dim=1, index=positive_ids.long())
        weights = positive_weights.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
        order = torch.argsort(weights, dim=-1, descending=True)
        ordered_logits = gathered.gather(dim=1, index=order)
        ordered_weights = weights.gather(dim=1, index=order)

        listmle = torch.zeros(ordered_logits.shape[0], device=ordered_logits.device, dtype=ordered_logits.dtype)
        for i in range(ordered_logits.shape[1]):
            tail = ordered_logits[:, i:]
            listmle = listmle + ordered_weights[:, i] * (torch.logsumexp(tail, dim=-1) - ordered_logits[:, i])
        return listmle.mean()


class LightweightDiffusionSampler(nn.Module):
    """Deterministic latent refinement over top-k candidates.

    This is intentionally lightweight, not a full DDPM implementation. It behaves
    like a short denoise-to-prior step in latent space so we can start testing the
    report's retrieval idea without rewriting the whole stack.
    """

    def __init__(self, num_steps: int = 4, temperature: float = 0.5, prior_mix: float = 0.35) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.temperature = float(temperature)
        self.prior_mix = float(prior_mix)

    def forward(self, query_embed: torch.Tensor, candidate_embed: torch.Tensor) -> torch.Tensor:
        if self.num_steps <= 0:
            return F.normalize(query_embed, dim=-1)

        z = F.normalize(query_embed, dim=-1)
        cand = F.normalize(candidate_embed, dim=-1)
        for step in range(self.num_steps):
            sim = torch.einsum("bd,bkd->bk", z, cand) / max(self.temperature, 1e-4)
            weight = torch.softmax(sim, dim=-1)
            prior = torch.einsum("bk,bkd->bd", weight, cand)
            alpha = self.prior_mix * float(step + 1) / float(self.num_steps)
            z = F.normalize((1.0 - alpha) * z + alpha * prior, dim=-1)
        return z


__all__ = [
    "ProjectionHead",
    "MultiPositiveContrastiveLoss",
    "build_assembly_aware_positive_weights",
    "AssemblyAwareListMLELoss",
    "LightweightDiffusionSampler",
]
