"""Stage 3 rigid analytic assembly utilities.

This module contains the reusable rigid alignment pieces for the 2026-04-05
contact-graph Stage 3 path:
- weighted Kabsch / Procrustes
- rigid application helpers
- short rigid-only refinement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from graspauto.rotation_utils import matrix_to_rot6d, rot6d_to_matrix

EPS = 1e-8


@dataclass
class RigidPose:
    rot: torch.Tensor
    trans: torch.Tensor


@dataclass
class RigidRefineResult:
    rot: torch.Tensor
    trans: torch.Tensor
    loss: torch.Tensor
    steps: int


def _ensure_batch(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if points.dim() == 2:
        return points.unsqueeze(0), True
    if points.dim() != 3:
        raise ValueError(f"Expected (N,3) or (B,N,3), got {tuple(points.shape)}")
    return points, False


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


def weighted_kabsch(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
) -> RigidPose:
    """Solve the weighted rigid transform from source -> target.

    Args:
        source_points: (N,3) or (B,N,3)
        target_points: (N,3) or (B,N,3)
        weights: (N,) or (B,N)
    """
    source_points, squeeze = _ensure_batch(source_points)
    target_points, _ = _ensure_batch(target_points)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)

    weights = weights.float().clamp_min(0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)

    src_center = (source_points * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
    tgt_center = (target_points * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
    src_centered = source_points - src_center
    tgt_centered = target_points - tgt_center

    cov = src_centered.transpose(1, 2) @ (tgt_centered * weights.unsqueeze(-1))
    u, _, vh = torch.linalg.svd(cov)
    rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)

    det = torch.det(rot)
    reflection = det < 0
    if reflection.any():
        vh = vh.clone()
        vh[reflection, -1, :] *= -1.0
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)

    trans = tgt_center.squeeze(1) - torch.bmm(rot, src_center.squeeze(1).unsqueeze(-1)).squeeze(-1)
    if squeeze:
        return RigidPose(rot=rot.squeeze(0), trans=trans.squeeze(0))
    return RigidPose(rot=rot, trans=trans)


def apply_rigid(points: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    points, squeeze = _ensure_batch(points)
    rot, rot_squeeze = _ensure_batch(rot.transpose(-1, -2)) if rot.dim() == 2 else (rot, False)
    if trans.dim() == 1:
        trans = trans.unsqueeze(0)
    transformed = torch.bmm(points, rot.transpose(1, 2)) + trans.unsqueeze(1)
    if squeeze:
        return transformed.squeeze(0)
    return transformed


def weighted_anchor_residual(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
    rot: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    source_points, squeeze = _ensure_batch(source_points)
    target_points, _ = _ensure_batch(target_points)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    aligned = apply_rigid(source_points, rot, trans)
    residual = (aligned - target_points).norm(dim=-1)
    loss = (residual * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(EPS)
    return loss.squeeze(0) if squeeze else loss


def refine_rigid_pose(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
    init_rot: torch.Tensor,
    init_trans: torch.Tensor,
    steps: int = 10,
    lr: float = 0.05,
    source_normals: Optional[torch.Tensor] = None,
    target_normals: Optional[torch.Tensor] = None,
    normal_weight: float = 0.1,
) -> RigidRefineResult:
    """Short rigid-only refinement around an initial alignment.

    This is intentionally lightweight: rotation + translation only.
    """
    source_points, squeeze = _ensure_batch(source_points)
    target_points, _ = _ensure_batch(target_points)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    if init_rot.dim() == 2:
        init_rot = init_rot.unsqueeze(0)
    if init_trans.dim() == 1:
        init_trans = init_trans.unsqueeze(0)
    if source_normals is not None:
        source_normals, _ = _ensure_batch(source_normals)
    if target_normals is not None:
        target_normals, _ = _ensure_batch(target_normals)

    rot6d = matrix_to_rot6d(init_rot).detach().clone().requires_grad_(True)
    trans = init_trans.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([rot6d, trans], lr=lr)

    final_loss = None
    with torch.enable_grad():
        for _ in range(int(steps)):
            optimizer.zero_grad()
            rot = rot6d_to_matrix(rot6d)
            aligned = apply_rigid(source_points, rot, trans)
            point_residual = (aligned - target_points).norm(dim=-1)
            loss = (point_residual * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(EPS)
            loss = loss.mean()

            if source_normals is not None and target_normals is not None:
                aligned_normals = torch.bmm(source_normals, rot.transpose(1, 2))
                cos = (F.normalize(aligned_normals, dim=-1) * F.normalize(target_normals, dim=-1)).sum(dim=-1)
                normal_loss = (1.0 - cos) * weights
                normal_loss = normal_loss.sum(dim=-1) / weights.sum(dim=-1).clamp_min(EPS)
                loss = loss + float(normal_weight) * normal_loss.mean()

            loss.backward()
            optimizer.step()
            final_loss = loss.detach()

    final_rot = rot6d_to_matrix(rot6d.detach())
    final_trans = trans.detach()
    if squeeze:
        return RigidRefineResult(
            rot=final_rot.squeeze(0),
            trans=final_trans.squeeze(0),
            loss=final_loss.squeeze(0) if final_loss.dim() > 0 else final_loss,
            steps=int(steps),
        )
    return RigidRefineResult(rot=final_rot, trans=final_trans, loss=final_loss, steps=int(steps))


__all__ = [
    "RigidPose",
    "RigidRefineResult",
    "weighted_kabsch",
    "apply_rigid",
    "weighted_anchor_residual",
    "refine_rigid_pose",
]
