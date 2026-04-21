"""Rotation utilities: 6D representation, geodesic loss, conversions.

References:
- Zhou et al. 2019 "On the Continuity of Rotation Representations in Neural Networks"
- StoryMR tokenizer architecture (~/storyboard_mr/docs/tokenizer_architecture.md)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (N, 3) to rotation matrix (N, 3, 3) via Rodrigues."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    R = eye + sin * K + (1 - cos) * (K @ K)
    return R


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (N, 3, 3) to axis-angle (N, 3)."""
    # Use the logarithmic map
    cos_angle = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)  # (N,)
    
    # Skew-symmetric part
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    
    norm = torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis / norm
    return axis * angle.unsqueeze(-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (*, 3, 3) to 6D representation (*, 6).
    
    Takes the first two columns of the rotation matrix.
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation (*, 6) to rotation matrix (*, 3, 3).
    
    Gram-Schmidt orthogonalization of the two input vectors.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (..., 3) to 6D rotation (..., 6)."""
    shape = aa.shape[:-1]
    R = axis_angle_to_matrix(aa.reshape(-1, 3))
    rot6d = matrix_to_rot6d(R)
    return rot6d.reshape(*shape, 6)


def rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation (..., 6) to axis-angle (..., 3)."""
    shape = rot6d.shape[:-1]
    R = rot6d_to_matrix(rot6d.reshape(-1, 6))
    aa = matrix_to_axis_angle(R)
    return aa.reshape(*shape, 3)


def hand_pose_aa_to_rot6d(hand_pose_aa: torch.Tensor) -> torch.Tensor:
    """Convert hand_pose from axis-angle (*, 45) to 6D rotation (*, 90).
    
    15 joints × 3D axis-angle → 15 joints × 6D rotation.
    """
    shape = hand_pose_aa.shape[:-1]
    joints = hand_pose_aa.reshape(*shape, 15, 3)
    rot6d = axis_angle_to_rot6d(joints)
    return rot6d.reshape(*shape, 90)


def hand_pose_rot6d_to_aa(hand_pose_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert hand_pose from 6D rotation (*, 90) to axis-angle (*, 45).
    
    15 joints × 6D rotation → 15 joints × 3D axis-angle.
    """
    shape = hand_pose_rot6d.shape[:-1]
    joints_6d = hand_pose_rot6d.reshape(*shape, 15, 6)
    aa = rot6d_to_axis_angle(joints_6d)
    return aa.reshape(*shape, 45)


def geodesic_loss(pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
    """Geodesic distance loss between two sets of 6D rotations.
    
    Args:
        pred_rot6d: (B, N*6) predicted 6D rotations (N joints)
        target_rot6d: (B, N*6) target 6D rotations
    
    Returns:
        Scalar mean geodesic distance in radians.
    """
    B = pred_rot6d.shape[0]
    pred_R = rot6d_to_matrix(pred_rot6d.reshape(-1, 6))     # (B*N, 3, 3)
    target_R = rot6d_to_matrix(target_rot6d.reshape(-1, 6))  # (B*N, 3, 3)
    
    # R_diff = R_pred^T @ R_target
    R_diff = pred_R.transpose(-1, -2) @ target_R
    
    # Geodesic distance = arccos((tr(R_diff) - 1) / 2)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)  # radians
    
    return angle.mean()
