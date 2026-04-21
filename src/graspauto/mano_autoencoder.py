"""Lightweight autoencoder for 54-D MANO parameter compression.

Compresses the raw 54-D MANO parameter vector (6D rot + 3D trans + 45D
hand pose) into a lower-dimensional latent space (default: 16-D). Used
as the first stage of the latent flow approach (Grok #2):

  1. Train this AE on all MANO grasps to learn a faithful 54→16→54 mapping.
  2. Cache the latent z = encode(x) for each training sample.
  3. Train the VelocityNetwork to do flow matching in the 16-D latent space.
  4. At inference: flow generates latent z → decode(z) → MANO mesh.

The bottleneck reduces the flow's output dimensionality from 54 to 16,
making it dramatically easier to hit precise targets in 10 Euler steps
(identified as the ceiling of the raw 54-D flow in Grok report #3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MANOAutoEncoder(nn.Module):
    """MLP autoencoder: 54-D MANO → latent_dim → 54-D MANO.

    Architecture: simple symmetric MLP with GELU activations and optional
    layer normalization. No VQ bottleneck initially — just continuous
    compression. VQ can be added later if discretization benefits materialize.

    Default config (latent_dim=16, hidden=[256, 128]):
      Encoder: 54 → 256 → 128 → 16
      Decoder: 16 → 128 → 256 → 54
      Total params: ~100K (negligible vs velocity network)
    """

    def __init__(
        self,
        input_dim: int = 54,
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 128),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input → hidden layers → latent
        enc_layers: list[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h_d))
            if use_layer_norm:
                enc_layers.append(nn.LayerNorm(h_d))
            enc_layers.append(nn.GELU())
            in_d = h_d
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent → hidden layers (reversed) → output
        dec_layers: list[nn.Module] = []
        in_d = latent_dim
        for h_d in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_d, h_d))
            if use_layer_norm:
                dec_layers.append(nn.LayerNorm(h_d))
            dec_layers.append(nn.GELU())
            in_d = h_d
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        """(B, 54) → (B, latent_dim)"""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """(B, latent_dim) → (B, 54)"""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (reconstruction, latent).

        Args:
            x: (B, 54) raw MANO parameters
        Returns:
            x_recon: (B, 54) reconstructed MANO parameters
            z: (B, latent_dim) latent representation
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Residual block for the ResidualMANOAutoEncoder
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + MLP(LayerNorm(x))."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        # Zero-init the last linear so the residual starts as identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ResidualMANOAutoEncoder(nn.Module):
    """Residual autoencoder for 54-D MANO parameter compression.

    Upgrades over plain MANOAutoEncoder:
    - Skip connections within encoder and decoder (ResidualBlock) for
      better gradient flow and fine-grained corrections.
    - Deeper: default 4 ResBlocks per side (vs 2-3 plain layers).
    - Same API (encode / decode / forward) so it's a drop-in replacement.

    Default config (latent_dim=32, hidden=256, n_blocks=4):
      Encoder: 54 → project(256) → 4×ResBlock(256) → compress(32)
      Decoder: 32 → expand(256) → 4×ResBlock(256) → project(54)
      Total params: ~1.1M
    """

    def __init__(
        self,
        input_dim: int = 54,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        expansion: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_proj = nn.Linear(input_dim, hidden_dim)
        self.enc_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expansion) for _ in range(n_blocks)]
        )
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_from_latent = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expansion) for _ in range(n_blocks)]
        )
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor) -> Tensor:
        h = self.enc_proj(x)
        h = self.enc_blocks(h)
        h = self.enc_norm(h)
        return self.enc_to_latent(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.dec_from_latent(z)
        h = self.dec_blocks(h)
        h = self.dec_norm(h)
        return self.dec_proj(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# VQ-VAE with Residual encoder/decoder
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Straight-through VQ bottleneck with EMA codebook update.

    Quantizes continuous latent vectors to nearest codebook entries.
    The quantization distance serves as a naturalness score: low distance
    means the input is close to a known pattern in the training data.
    """

    def __init__(self, num_codes: int = 512, code_dim: int = 32, ema_decay: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay

        # Codebook
        self.register_buffer("codebook", torch.randn(num_codes, code_dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_weight", self.codebook.clone())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize z to nearest codebook entry.

        Args:
            z: (B, code_dim) continuous latent vectors
        Returns:
            z_q: (B, code_dim) quantized vectors (straight-through)
            indices: (B,) codebook indices
            commit_loss: scalar commitment loss
        """
        # Distances to all codebook entries
        dists = torch.cdist(z.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)  # (B, num_codes)
        indices = dists.argmin(dim=-1)  # (B,)
        z_q = self.codebook[indices]  # (B, code_dim)

        # Commitment loss: encourage encoder to produce vectors close to codebook
        commit_loss = (z - z_q.detach()).pow(2).mean()

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(z.shape[0], self.num_codes, device=z.device)
                one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
                count = one_hot.sum(0)
                self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
                weight_sum = one_hot.T @ z  # (num_codes, code_dim)
                self.ema_weight.mul_(self.ema_decay).add_(weight_sum, alpha=1 - self.ema_decay)
                # Laplace smoothing
                n = self.ema_count.sum()
                count_smooth = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.codebook.copy_(self.ema_weight / count_smooth.unsqueeze(1))

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()  # forward: z_q, backward: z

        return z_q_st, indices, commit_loss

    def quantize_distance(self, z: Tensor) -> Tensor:
        """Compute distance to nearest codebook entry (naturalness score).

        Args:
            z: (B, code_dim) continuous latent vectors
        Returns:
            (B,) distance to nearest codebook entry
        """
        dists = torch.cdist(z.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        return dists.min(dim=-1).values


class ResidualVQVAE(nn.Module):
    """VQ-VAE with Residual encoder/decoder for MANO parameters.

    Same encoder/decoder as ResidualMANOAutoEncoder, but with a VQ
    bottleneck between them. The quantization distance provides a
    naturalness score for candidate filtering.
    """

    def __init__(
        self,
        input_dim: int = 54,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        num_codes: int = 512,
        expansion: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder (same as ResidualMANOAutoEncoder)
        self.enc_proj = nn.Linear(input_dim, hidden_dim)
        self.enc_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expansion) for _ in range(n_blocks)]
        )
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # VQ bottleneck
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)

        # Decoder (same as ResidualMANOAutoEncoder)
        self.dec_from_latent = nn.Linear(latent_dim, hidden_dim)
        self.dec_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expansion) for _ in range(n_blocks)]
        )
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor) -> Tensor:
        """Encode to continuous latent (pre-quantization)."""
        h = self.enc_proj(x)
        h = self.enc_blocks(h)
        h = self.enc_norm(h)
        return self.enc_to_latent(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent (quantized or continuous)."""
        h = self.dec_from_latent(z)
        h = self.dec_blocks(h)
        h = self.dec_norm(h)
        return self.dec_proj(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward: encode → quantize → decode.

        Returns:
            x_recon: (B, input_dim) reconstruction
            z_q: (B, latent_dim) quantized latent
            indices: (B,) codebook indices
            commit_loss: scalar commitment loss
        """
        z = self.encode(x)
        z_q, indices, commit_loss = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, z_q, indices, commit_loss

    def quantize_distance(self, x_normed: Tensor) -> Tensor:
        """Compute naturalness score: encode → distance to nearest code."""
        z = self.encode(x_normed)
        return self.vq.quantize_distance(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["MANOAutoEncoder", "ResidualMANOAutoEncoder", "ResidualVQVAE", "VectorQuantizer"]
