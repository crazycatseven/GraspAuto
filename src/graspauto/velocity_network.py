"""Velocity network for the graspauto conditional flow matching decoder.

Implements the velocity field

    vθ(xt, t | c)  →  Tensor (B, 54)

where:
- `xt` is the current 54-D MANO parameter vector at flow time t
   (layout: 6D rot + 3D translation + 45D hand pose; see `mano_decoder.py`).
- `t`  is the per-sample flow time, scalar in [0, 1].
- `c`  is a `ConditioningBundle` from `graspauto.conditioning`, containing
   (B, 71, H) conditioning tokens (object patches + finger + palm + unified).

Architecture (per the paper.3):

1. **Sinusoidal time embedding** of `t` to 128 dims, MLP-projected to the
   transformer hidden dim. Used as the AdaLN modulation signal.
2. **xt encoder**: a small MLP mapping the 54-D state to a hidden vector,
   summed with the time embedding, treated as a single query token.
3. **N cross-attention blocks**, each with:
     - AdaLN-modulated self-attention (query attends to itself)
     - AdaLN-modulated cross-attention (query attends to the 71 conditioning tokens)
     - AdaLN-modulated FFN
   The same time embedding modulates every block via AdaLN scale + shift.
4. **Output projection** from hidden_dim back to 54.

Total parameters at default config (hidden=256, n_layers=6, n_heads=4):
~10 M, which matches the budget in the paper.

Design choices:
- **Single query token (xt)**: keeps the network compact and avoids the
  capacity waste of N learned query tokens for an output that only needs
  54 numbers.
- **AdaLN modulation everywhere**: standard DiT pattern, gives stable
  time conditioning without adding many extra parameters.
- **Cross-attention is the only conditioning path**: no concat, no FiLM
  on the conditioning side. Cross-attention preserves the spatial
  structure of the object patches and the per-finger token semantics.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graspauto.conditioning import ConditioningBundle


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Embeds a scalar time t ∈ [0, 1] as a sinusoidal feature vector.

    Same construction as DDPM / DiT papers: log-spaced frequencies, sin/cos
    interleaved. Output dim must be even.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) → (B, dim)"""
        if t.dim() != 1:
            raise ValueError(f"t must be 1-D (B,), got {t.dim()}-D")
        half = self.dim // 2
        # log-spaced frequencies
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        # (B, half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb.to(t.dtype)


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (AdaLN)
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """Layer norm whose scale + shift are produced from a time embedding.

    Standard DiT pattern: `norm(x) * (1 + scale(t_emb)) + shift(t_emb)`.
    """

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(time_dim, 2 * dim)
        # Init to zero so initially the modulation is identity (scale=1, shift=0).
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """x: (B, N, D), t_emb: (B, time_dim) → (B, N, D)"""
        scale, shift = self.modulation(t_emb).chunk(2, dim=-1)  # each (B, D)
        normed = self.norm(x)
        return normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Standard transformer block: self-attn → cross-attn → FFN, with AdaLN.

    The query input is shape (B, Q, D) where Q is the number of query tokens
    (=1 in graspauto for the xt query). The conditioning is shape (B, K, D)
    where K is the number of conditioning tokens (=71).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        time_dim: int,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        # Stored so forward() can reshape attn_bias to (B*heads, Q, K).
        self._num_heads = self.n_heads

        # Self-attention on the query (small, since query is usually 1 token)
        self.self_norm = AdaLayerNorm(self.dim, time_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Cross-attention from query to conditioning
        self.cross_norm = AdaLayerNorm(self.dim, time_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Feed-forward
        self.ffn_norm = AdaLayerNorm(self.dim, time_dim)
        ffn_hidden = self.dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, self.dim),
        )

    def forward(
        self,
        query: Tensor,
        condition: Tensor,
        t_emb: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query:     (B, Q, D)
            condition: (B, K, D)
            t_emb:     (B, time_dim)
            attn_bias: Optional (B, K) additive bias added to the cross-
                       attention logits (shared across all query tokens and
                       heads). Used for distance-based part-aware gating.
        Returns:
            (B, Q, D)
        """
        B, Q, _ = query.shape
        K = condition.shape[1]

        # Self-attention on the query
        h = self.self_norm(query, t_emb)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        query = query + attn_out

        # Cross-attention into the conditioning tokens.
        # PyTorch MultiheadAttention accepts attn_mask of shape (B*heads, Q, K)
        # when used with float. We broadcast (B, K) → (B*heads, Q, K) so every
        # query / head sees the same per-key additive bias.
        h = self.cross_norm(query, t_emb)
        mask = None
        if attn_bias is not None:
            # (B, K) → (B, 1, K) → expand to (B*heads, Q, K)
            mask = attn_bias.unsqueeze(1).expand(B, Q, K).to(h.dtype)
            mask = mask.unsqueeze(1).expand(B, self.n_heads, Q, K).reshape(B * self.n_heads, Q, K)
        attn_out, _ = self.cross_attn(h, condition, condition, need_weights=False, attn_mask=mask)
        query = query + attn_out

        # Feed-forward
        h = self.ffn_norm(query, t_emb)
        query = query + self.ffn(h)

        return query


# ---------------------------------------------------------------------------
# Velocity network
# ---------------------------------------------------------------------------

class VelocityNetwork(nn.Module):
    """Cross-attention transformer implementing vθ(xt, t | c) for graspauto.

    Defaults:
        input_dim     = 54   (MANO param vector layout)
        hidden_dim    = 256
        n_heads       = 4
        n_layers      = 6
        time_embed_dim = 128
        ffn_mult      = 4

    Parameter count at defaults: ~10 M.
    """

    def __init__(
        self,
        input_dim: int = 54,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        time_embed_dim: int = 128,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        out_proj_init_std: float = 0.02,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_proj_init_std = float(out_proj_init_std)

        # Time embedding pipeline
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # xt encoder: 54 → hidden, then sum with time embedding
        self.xt_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stack of cross-attention blocks
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim,
                    n_heads=n_heads,
                    time_dim=hidden_dim,
                    ffn_mult=ffn_mult,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Final output projection.
        # `out_proj_init_std` controls the initial weight scale:
        #   * 0.0  → exact DiT-style zero init (initial v_pred = 0 everywhere,
        #            maximally stable but the projection takes ~80 epochs to
        #            grow to full scale on this task — confirmed empirically
        #            by the r005-r007 runs on 2026-04-11).
        #   * >0.0 → small Gaussian (GPT-style small-init). Starts with
        #            near-zero output magnitude (since fan_in * std^2 is small)
        #            but has nonzero gradients on the first step, so the
        #            projection starts growing immediately. Default 0.02 is
        #            the standard transformer small-init scale. This reclaims
        #            the ~80 wasted warmup epochs without destabilizing early
        #            training.
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        if self.out_proj_init_std == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.out_proj_init_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        xt: Tensor,
        t: Tensor,
        condition: Optional[Any] = None,
    ) -> Tensor:
        """Predict the velocity at (xt, t) given the conditioning bundle.

        Args:
            xt:        (B, 54) — current MANO parameter state.
            t:         (B,)    — flow time in [0, 1].
            condition: a `ConditioningBundle` from `graspauto.conditioning`,
                       OR a raw (B, K, hidden_dim) tensor (for tests),
                       OR None for unconditional sampling (in which case
                       cross-attention attends over a single zero token).

        Returns:
            (B, 54) — predicted velocity.
        """
        if xt.dim() != 2 or xt.shape[-1] != self.input_dim:
            raise ValueError(
                f"xt must be (B, {self.input_dim}), got {tuple(xt.shape)}"
            )
        if t.shape != (xt.shape[0],):
            raise ValueError(
                f"t must be ({xt.shape[0]},), got {tuple(t.shape)}"
            )

        # Build the conditioning tensor (B, K, hidden_dim) plus optional attention bias.
        attention_bias: Optional[Tensor] = None
        if condition is None:
            cond_tokens = torch.zeros(
                xt.shape[0], 1, self.hidden_dim, device=xt.device, dtype=xt.dtype
            )
        elif isinstance(condition, ConditioningBundle):
            cond_tokens = condition.tokens
            attention_bias = condition.attention_bias
        elif isinstance(condition, Tensor):
            if condition.dim() != 3:
                raise ValueError(
                    f"raw condition tensor must be 3-D (B, K, D), got {condition.dim()}-D"
                )
            cond_tokens = condition
        else:
            raise TypeError(
                f"condition must be ConditioningBundle, Tensor, or None — got {type(condition).__name__}"
            )

        if cond_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"conditioning hidden_dim {cond_tokens.shape[-1]} != network hidden_dim {self.hidden_dim}"
            )

        # Time embedding
        t_sin = self.time_embed(t)        # (B, time_embed_dim)
        t_emb = self.time_mlp(t_sin)      # (B, hidden_dim)

        # xt → (B, 1, hidden_dim) query token, with time injected
        xt_h = self.xt_encoder(xt)        # (B, hidden_dim)
        query = (xt_h + t_emb).unsqueeze(1)  # (B, 1, hidden_dim)

        # Stack of cross-attention blocks
        for block in self.blocks:
            query = block(query, cond_tokens, t_emb, attn_bias=attention_bias)

        # Output projection
        out = self.out_norm(query.squeeze(1))   # (B, hidden_dim)
        velocity = self.out_proj(out)            # (B, input_dim)

        return velocity

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "SinusoidalTimeEmbedding",
    "AdaLayerNorm",
    "CrossAttentionBlock",
    "VelocityNetwork",
]
