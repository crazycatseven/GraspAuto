"""MANO decoder wrapper with 6D continuous rotation parameterization.

The graspauto flow decoder produces a 54-dimensional vector encoding a MANO
hand pose:

    mano_params[:,  0: 6]  →  6D continuous rotation (Zhou et al. 2019)
    mano_params[:,  6: 9]  →  global translation (xyz, meters)
    mano_params[:,  9:54]  →  hand pose, 45 axis-angle joint rotations
                              (MANO's native pose representation)

The 6D rotation parameterization (Zhou et al. 2019, "On the Continuity of
Rotation Representations in Neural Networks", CVPR 2019) uses the first
two columns of the rotation matrix, then recovers the third via
Gram-Schmidt orthonormalization. This avoids the discontinuities that
plague axis-angle (gimbal lock at π rotations) and quaternion (antipodal
ambiguity) under continuous learning. It's the standard choice for any
modern pose-regression / pose-generation network.

The decoder class wraps `manotorch.manolayer.ManoLayer` (the same
library the rest of the project uses; verified installed in `.venv` and
imported by various training and evaluation modules already). We
pass *zero* global orientation to the MANO layer and apply the 6D rotation
ourselves *after* the MANO forward pass, on the output vertices and joints.
This is intentional: it sidesteps the need to convert a rotation matrix
back to axis-angle (which is non-injective and would break gradient flow
near the discontinuities).

References:
- Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the
  Continuity of Rotation Representations in Neural Networks. CVPR 2019.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Default MANO assets root, matching graspauto.vertex_groups.MANO_MODEL_PATH structure.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANO_ASSETS_ROOT = PROJECT_ROOT / "assets" / "mano_v1_2"

# Layout constants for the 54-D graspauto MANO parameter vector.
ROT6D_DIM = 6
TRANSLATION_DIM = 3
HAND_POSE_DIM = 45
MANO_PARAM_DIM = ROT6D_DIM + TRANSLATION_DIM + HAND_POSE_DIM  # = 54

# Slices for unpacking a (B, 54) parameter tensor.
ROT6D_SLICE = slice(0, ROT6D_DIM)
TRANSLATION_SLICE = slice(ROT6D_DIM, ROT6D_DIM + TRANSLATION_DIM)
POSE_SLICE = slice(ROT6D_DIM + TRANSLATION_DIM, MANO_PARAM_DIM)


# ---------------------------------------------------------------------------
# 6D continuous rotation ↔ 3×3 rotation matrix
# ---------------------------------------------------------------------------

def rot6d_to_rotation_matrix(d6: Tensor) -> Tensor:
    """Convert a 6D continuous rotation representation to a 3×3 rotation matrix.

    Implementation follows Zhou et al. 2019, equation 2: take the first two
    columns of the input as the first two columns of the matrix, run
    Gram-Schmidt to make them orthonormal, then take the cross product to
    get the third column.

    Args:
        d6: Tensor of shape (..., 6).

    Returns:
        Tensor of shape (..., 3, 3) representing rotation matrices in SO(3).
        Guaranteed orthonormal up to floating-point precision; determinant +1.
    """
    if d6.shape[-1] != 6:
        raise ValueError(f"input last dim must be 6, got {d6.shape}")

    # First two raw column vectors
    a1 = d6[..., 0:3]  # (..., 3)
    a2 = d6[..., 3:6]  # (..., 3)

    # Gram-Schmidt orthonormalization
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    # Subtract the projection of a2 onto b1
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj, dim=-1, eps=1e-8)
    # Third column is the cross product of the first two (right-handed frame)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns: shape (..., 3, 3)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_matrix_to_rot6d(R: Tensor) -> Tensor:
    """Convert a 3×3 rotation matrix to its 6D continuous representation.

    Inverse of `rot6d_to_rotation_matrix` up to the redundant degrees of
    freedom (the 6D representation is one-to-one with SO(3) rotations after
    Gram-Schmidt projection, but there are infinitely many 6-vectors that
    map to the same matrix). We pick the canonical representative: the
    first two columns of the matrix.

    Args:
        R: Tensor of shape (..., 3, 3) representing rotation matrices.

    Returns:
        Tensor of shape (..., 6).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"input last two dims must be (3, 3), got {R.shape}")
    # Concatenate the first two columns
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


# ---------------------------------------------------------------------------
# MangoMANODecoder
# ---------------------------------------------------------------------------

class MangoMANODecoder(nn.Module):
    """Wraps a `manotorch.ManoLayer` to decode graspauto's 54-D parameter vector.

    The decoder is a thin nn.Module so that it integrates into the training
    pipeline with `.to(device)` semantics matching the rest of the project.
    Internally:

    1. Parse `mano_params: (B, 54)` into rot6d, translation, hand_pose.
    2. Convert rot6d → 3×3 rotation matrix via Gram-Schmidt.
    3. Build a (B, 48) full pose with **zero global orientation** + the 45-D
       hand pose, and call the underlying `ManoLayer`.
    4. Apply the rotation matrix and translation to the resulting vertices
       and joints, in that order.

    The MANO `betas` are fixed at zero (the user's project uses pose-only
    targets — graspauto's `mano_stats.py` and the surrounding pipeline assume
    `betas=0`). Adding `betas` as an additional input is a future extension
    and would mean a `(B, 64)` input vector instead of `(B, 54)`.
    """

    def __init__(
        self,
        mano_assets_root: str | Path | None = None,
        side: str = "right",
        center_idx: int | None = None,
        flat_hand_mean: bool = True,
    ):
        super().__init__()
        if mano_assets_root is None:
            mano_assets_root = DEFAULT_MANO_ASSETS_ROOT
        # Lazy import so the rest of graspauto doesn't fail to import if
        # manotorch is missing for some reason — only the decoder will fail.
        from manotorch.manolayer import ManoLayer  # noqa: PLC0415

        # IMPORTANT: center_idx default is None, NOT 0. The graspauto dataset's
        # gt_local_verts and gt_world_verts were computed with center_idx=None
        # (verified empirically 2026-04-11 by exact 0.0000 mm match against
        # the rest hand). Using center_idx=0 introduces a constant ~96 mm
        # offset because it subtracts the wrist position from all vertices.
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=center_idx,
            mano_assets_root=str(mano_assets_root),
            use_pca=False,
            flat_hand_mean=flat_hand_mean,
        )
        # Expose face indices as a buffer so .to(device) propagates them.
        self.register_buffer("faces", self.mano_layer.th_faces.long(), persistent=False)

    def forward(self, mano_params: Tensor, betas: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Decode a 54-D parameter vector to a posed MANO hand mesh.

        Args:
            mano_params: Tensor of shape (B, 54). Layout described in module docstring.
            betas:       Optional Tensor of shape (B, 10) for shape parameters.
                         Defaults to zeros (the project's standard setting).

        Returns:
            Dictionary with keys:
                "vertices": (B, 778, 3) — posed and translated hand vertices.
                "joints":   (B, 21, 3)  — posed and translated MANO joints.
                "rotation_matrix": (B, 3, 3) — recovered rotation matrix.
                "translation":     (B, 3)    — translation vector.
                "hand_pose":       (B, 45)   — pass-through of the joint angles.
        """
        if mano_params.dim() != 2:
            raise ValueError(f"mano_params must be 2-D (B, 54), got {mano_params.dim()}-D")
        if mano_params.shape[-1] != MANO_PARAM_DIM:
            raise ValueError(
                f"mano_params last dim must be {MANO_PARAM_DIM}, got {mano_params.shape[-1]}"
            )

        batch_size = mano_params.shape[0]
        device = mano_params.device
        dtype = mano_params.dtype

        rot6d = mano_params[:, ROT6D_SLICE]
        translation = mano_params[:, TRANSLATION_SLICE]
        hand_pose = mano_params[:, POSE_SLICE]  # (B, 45) axis-angle

        rotation_matrix = rot6d_to_rotation_matrix(rot6d)  # (B, 3, 3)

        # Build (B, 48) full pose with zero global orient
        zero_global = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        full_pose = torch.cat([zero_global, hand_pose], dim=-1)  # (B, 48)

        if betas is None:
            betas = torch.zeros(batch_size, 10, device=device, dtype=dtype)
        elif betas.shape != (batch_size, 10):
            raise ValueError(f"betas must have shape ({batch_size}, 10), got {tuple(betas.shape)}")

        mano_out = self.mano_layer(full_pose, betas)
        verts_rest = mano_out.verts  # (B, 778, 3) — pose-only, no global rot/trans
        joints_rest = mano_out.joints  # (B, 21, 3)

        # Apply our rotation. Vertices are stored as row vectors in (B, V, 3),
        # so the rotation is verts @ R^T  (i.e., each row v becomes v @ R^T = R @ v).
        verts_rotated = torch.einsum("bvc,bcd->bvd", verts_rest, rotation_matrix.transpose(-1, -2))
        joints_rotated = torch.einsum("bjc,bcd->bjd", joints_rest, rotation_matrix.transpose(-1, -2))

        # Apply translation
        verts_final = verts_rotated + translation.unsqueeze(1)  # broadcast over V
        joints_final = joints_rotated + translation.unsqueeze(1)

        return {
            "vertices": verts_final,
            "joints": joints_final,
            "rotation_matrix": rotation_matrix,
            "translation": translation,
            "hand_pose": hand_pose,
        }

    @property
    def num_vertices(self) -> int:
        return 778

    @property
    def num_joints(self) -> int:
        return 21


# Convenience exports
__all__ = [
    "MANO_PARAM_DIM",
    "ROT6D_DIM",
    "TRANSLATION_DIM",
    "HAND_POSE_DIM",
    "rot6d_to_rotation_matrix",
    "rotation_matrix_to_rot6d",
    "MangoMANODecoder",
]
