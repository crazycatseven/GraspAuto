"""Full-parameter Test-Time Optimization (TTO) for graspauto inference.

After the conditional flow matching decoder produces an initial 54-D MANO
parameter vector (and after best-of-N selection picks the top candidate),
we run gradient descent directly on all 54 parameters to minimize a
physics-aware loss that the generator alone cannot achieve:

    L = w_contact * L_contact_align
      + w_pen     * L_penetration
      + w_joint   * L_joint_limit
      + w_rot     * L_rot_orth
      + w_prior   * ||params - init_params||^2

This is the analog of graspauto's post-processing TTO step, adapted to
graspauto's parameter layout. The key difference vs `rigid_micro_refine`
is that we optimize the hand pose (joints) as well, allowing the hand
to articulate into a better grasp, not just translate/rotate.

called for a "light rigid micro-refine"; this module is the heavier
"full TTO" that follows it (or replaces it) when sub-30mm accuracy is
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from graspauto.losses import DEFAULT_MANO_FINGER_TIP_JOINTS
from graspauto.mano_decoder import (
    MANO_PARAM_DIM,
    POSE_SLICE,
    ROT6D_SLICE,
    TRANSLATION_SLICE,
    MangoMANODecoder,
)


@dataclass
class TTOResult:
    """Output of `full_params_tto`."""
    refined_params: Tensor   # (B, 54) post-refine MANO params
    initial_loss: Tensor     # scalar: mean total loss at step 0
    final_loss: Tensor       # scalar: mean total loss at last step
    num_steps: int
    history: Optional[list[float]] = None  # per-step mean loss (if store_history=True)


def nearest_point_penetration_loss(
    hand_verts: Tensor,          # (B, V, 3)
    object_points: Tensor,        # (B, N, 3)
    object_normals: Tensor,       # (B, N, 3)
) -> Tensor:
    """Mean per-sample hand-inside-object penetration depth.

    For each hand vertex, find the nearest object surface point and check
    its signed distance against the outward object normal. Negative signed
    distance = inside the object = penetration to penalize.

    Returns (B,) tensor of penetration depths in meters (the same unit the
    mesh uses internally). Differentiable end-to-end through the pairwise
    distance and gather operations.
    """
    hv_sq = hand_verts.pow(2).sum(dim=-1, keepdim=True)
    op_sq = object_points.pow(2).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(hand_verts, object_points.transpose(-1, -2))
    dist_sq = (hv_sq + op_sq - 2.0 * cross).clamp_min(1e-12)
    nearest_idx = dist_sq.argmin(dim=-1)
    idx_exp = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
    nearest_pts = object_points.gather(1, idx_exp)
    nearest_nor = object_normals.gather(1, idx_exp)
    signed = ((hand_verts - nearest_pts) * nearest_nor).sum(dim=-1)
    # Negative signed means the vertex is on the inside of the surface.
    # Penetration depth = max(0, -signed).
    return (-signed).clamp_min(0.0).mean(dim=-1)


def contact_align_loss_per_sample(
    decoded_joints: Tensor,              # (B, 21, 3)
    target_finger_centroids: Tensor,     # (B, 5, 3)
    finger_joint_indices: tuple[int, ...] = tuple(DEFAULT_MANO_FINGER_TIP_JOINTS),
) -> Tensor:
    """Per-sample (B,) mean squared distance from fingertips to centroids."""
    idx = torch.tensor(list(finger_joint_indices), dtype=torch.long, device=decoded_joints.device)
    tips = decoded_joints.index_select(dim=1, index=idx)  # (B, 5, 3)
    diff = tips - target_finger_centroids
    return diff.pow(2).sum(dim=-1).mean(dim=-1)


def fingertip_to_surface_loss(
    decoded_joints: Tensor,              # (B, 21, 3)
    object_points: Tensor,               # (B, N, 3)
    finger_joint_indices: tuple[int, ...] = tuple(DEFAULT_MANO_FINGER_TIP_JOINTS),
) -> Tensor:
    """Per-sample (B,) mean squared distance from each fingertip to the nearest
    object surface point.

    Unlike `contact_align_loss_per_sample`, this does NOT rely on the frozen
    contact head's (noisy) predicted centroids — it directly targets "land
    each fingertip on the object surface", which is an unambiguous physical
    objective. Ambiguity about *where* on the object the fingertip should
    land is left to the generator; TTO only makes sure the fingertip touches
    the object cleanly.
    """
    idx = torch.tensor(list(finger_joint_indices), dtype=torch.long, device=decoded_joints.device)
    tips = decoded_joints.index_select(dim=1, index=idx)  # (B, 5, 3)
    # Pairwise distance (B, 5, N)
    hv_sq = tips.pow(2).sum(dim=-1, keepdim=True)
    op_sq = object_points.pow(2).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(tips, object_points.transpose(-1, -2))
    dist_sq = (hv_sq + op_sq - 2.0 * cross).clamp_min(1e-12)
    # Per-fingertip minimum distance
    per_tip = dist_sq.min(dim=-1).values  # (B, 5), squared meters
    return per_tip.mean(dim=-1)


def full_params_tto(
    init_mano_params: Tensor,
    *,
    target_finger_centroids: Optional[Tensor],
    object_points: Tensor,
    object_normals: Tensor,
    decoder: MangoMANODecoder,
    num_steps: int = 200,
    lr: float = 5e-3,
    w_contact: float = 1.0,       # weight on fingertip→predicted-centroid (0 disables)
    w_surface: float = 0.0,       # weight on fingertip→nearest-object-surface
    w_pen: float = 5.0,
    w_joint: float = 0.01,
    w_rot: float = 0.01,
    w_prior: float = 0.01,
    joint_lower: float = -1.5,
    joint_upper: float = 1.5,
    store_history: bool = False,
) -> TTOResult:
    """Gradient-descent TTO on the full 54-D MANO parameter vector.

    Args:
        init_mano_params: (B, 54) initial MANO parameters from the flow decoder.
        target_finger_centroids: (B, 5, 3) predicted centroids from the frozen head.
        object_points: (B, N, 3) object point cloud in the same frame.
        object_normals: (B, N, 3) outward object normals.
        decoder: MangoMANODecoder instance on the right device.
        num_steps: Adam iterations (default 200).
        lr: Adam lr (default 5e-3 — finger-scale movements per step).
        w_*: Loss-term weights. w_pen is the dominant term (5.0) because
            penetration is the main physics failure mode; w_contact=1.0
            keeps the hand aligned to the predicted contact target; the
            regularizers are light.
        w_prior: Pulls params toward their init to prevent drifting off-
            manifold — this is the analog of "diffusion prior" guidance
            in guided sampling but applied after sampling.
        joint_lower/upper: Hinge range for pose angles.
        store_history: If True, returns per-step mean-loss history.

    Returns:
        TTOResult with refined_params, initial_loss, final_loss, num_steps.
    """
    if init_mano_params.dim() != 2 or init_mano_params.shape[-1] != MANO_PARAM_DIM:
        raise ValueError(f"init_mano_params must be (B, {MANO_PARAM_DIM}), got {tuple(init_mano_params.shape)}")

    init_detached = init_mano_params.detach()
    params = init_detached.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)

    def compute_loss() -> tuple[Tensor, dict[str, Tensor]]:
        out = decoder(params)
        verts = out["vertices"]
        joints = out["joints"]

        if w_contact > 0 and target_finger_centroids is not None:
            contact = contact_align_loss_per_sample(joints, target_finger_centroids)
        else:
            contact = torch.zeros(params.shape[0], device=params.device)

        if w_surface > 0:
            surface = fingertip_to_surface_loss(joints, object_points)
        else:
            surface = torch.zeros(params.shape[0], device=params.device)

        pen = nearest_point_penetration_loss(verts, object_points, object_normals)  # (B,)

        pose = params[:, POSE_SLICE]
        over = (pose - joint_upper).clamp_min(0.0)
        under = (joint_lower - pose).clamp_min(0.0)
        joint_hinge = (over.pow(2) + under.pow(2)).mean(dim=-1)

        rot6d = params[:, ROT6D_SLICE]
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]
        dot = (a1 * a2).sum(dim=-1)
        n1 = a1.norm(dim=-1)
        n2 = a2.norm(dim=-1)
        rot_orth = dot.pow(2) + (n1 - 1.0).pow(2) + (n2 - 1.0).pow(2)

        prior = (params - init_detached).pow(2).mean(dim=-1)

        total_per_sample = (
            w_contact * contact
            + w_surface * surface
            + w_pen * pen
            + w_joint * joint_hinge
            + w_rot * rot_orth
            + w_prior * prior
        )
        total = total_per_sample.mean()
        return total, {
            "contact": contact.mean().detach(),
            "surface": surface.mean().detach(),
            "penetration": pen.mean().detach(),
            "joint": joint_hinge.mean().detach(),
            "rot_orth": rot_orth.mean().detach(),
            "prior": prior.mean().detach(),
        }

    history: list[float] = []
    with torch.enable_grad():
        initial_loss, _ = compute_loss()
        initial_loss = initial_loss.detach()
        final_loss = initial_loss

        for step in range(num_steps):
            optimizer.zero_grad()
            loss, _ = compute_loss()
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()
            if store_history:
                history.append(float(final_loss.item()))

    return TTOResult(
        refined_params=params.detach(),
        initial_loss=initial_loss,
        final_loss=final_loss,
        num_steps=int(num_steps),
        history=history if store_history else None,
    )


__all__ = [
    "TTOResult",
    "nearest_point_penetration_loss",
    "contact_align_loss_per_sample",
    "full_params_tto",
]
