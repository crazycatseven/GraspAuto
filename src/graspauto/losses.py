"""Training losses for the graspauto conditional flow matching decoder.

Per the paper.4, the training loss is a weighted
sum of four terms:

| term            | default weight  | source                                                       |
|-----------------|-----------------|--------------------------------------------------------------|
| L_flow          | 1.0             | flow matching velocity loss (already in flow_matching.py)    |
| L_contact_align | 0.1 → 0.5 ramp  | distance from MANO finger tips to predicted contact centroids|
| L_joint_limit   | 0.01            | hinge on MANO joint angles outside anatomical range          |
| L_rotation_orth | 0.01            | encourage input 6D rotation to already be orthonormal        |

The contact alignment weight follows a curriculum: zero during the first
20 epochs (let the decoder learn the marginal MANO distribution from flow
loss alone), then linear ramp from 0.1 → 0.5 over epochs 20–60, then
constant at 0.5. This is the schedule documented in the paper

`MangoLossBundle.forward(...)` returns a dict with:
  - per-component losses (already weighted)
  - "total" — the sum

Curriculum support is provided by `LossSchedule(epoch)` which returns a
dict[str, float] of per-epoch weights. The trainer is expected to call
`schedule(epoch)` once per epoch and pass the resulting weights into the
loss bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


# Default MANO finger-tip joint indices for the standard 21-joint MANO model.
# Order: [wrist=0, thumb (1-4), index (5-8), middle (9-12), ring (13-16), little (17-20)].
# The fingertip is the 4th joint of each finger.
DEFAULT_MANO_FINGER_TIP_JOINTS: tuple[int, ...] = (4, 8, 12, 16, 20)

# Default anatomical joint limit. ±1.5 rad ≈ ±86° per axis-angle component.
# This is intentionally loose; tightening it slightly is a future tuning option.
DEFAULT_JOINT_LOWER = -1.5
DEFAULT_JOINT_UPPER = 1.5


# ---------------------------------------------------------------------------
# Loss schedule (curriculum)
# ---------------------------------------------------------------------------

@dataclass
class LossSchedule:
    """Per-epoch loss weight schedule.

    Implements the curriculum from the paper.4:
      - Epochs [0, ramp_start): contact alignment weight is `warmup_value` (default 0.0).
        Train from flow loss alone to learn the marginal MANO distribution.
      - Epochs [ramp_start, ramp_end): linearly interpolate contact alignment
        weight from `contact_initial` to `contact_final`.
      - Epochs [ramp_end, ∞): contact alignment weight is `contact_final`.

    All other weights are constant across epochs.
    """

    flow_weight: float = 1.0
    joint_limit_weight: float = 0.01
    rotation_orth_weight: float = 0.01
    contact_initial: float = 0.1
    contact_final: float = 0.5
    contact_warmup_value: float = 0.0
    contact_ramp_start: int = 20
    contact_ramp_end: int = 60

    def __post_init__(self) -> None:
        if self.contact_ramp_end <= self.contact_ramp_start:
            raise ValueError(
                f"contact_ramp_end ({self.contact_ramp_end}) must be > "
                f"contact_ramp_start ({self.contact_ramp_start})"
            )

    def __call__(self, epoch: int) -> Dict[str, float]:
        if epoch < self.contact_ramp_start:
            contact_w = self.contact_warmup_value
        elif epoch < self.contact_ramp_end:
            ramp_t = (epoch - self.contact_ramp_start) / (
                self.contact_ramp_end - self.contact_ramp_start
            )
            contact_w = self.contact_initial + ramp_t * (
                self.contact_final - self.contact_initial
            )
        else:
            contact_w = self.contact_final
        return {
            "flow": self.flow_weight,
            "contact_align": float(contact_w),
            "joint_limit": self.joint_limit_weight,
            "rotation_orth": self.rotation_orth_weight,
        }


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def joint_limit_hinge(
    joint_angles: Tensor,
    lower: float = DEFAULT_JOINT_LOWER,
    upper: float = DEFAULT_JOINT_UPPER,
) -> Tensor:
    """Quadratic hinge loss penalizing axis-angle components outside [lower, upper].

    Args:
        joint_angles: Tensor of shape (B, 45) — MANO hand pose in axis-angle.
        lower:        Per-component lower bound (radians).
        upper:        Per-component upper bound (radians).

    Returns:
        Scalar tensor — mean squared excess across batch and dimensions.
    """
    if joint_angles.dim() != 2:
        raise ValueError(f"joint_angles must be 2-D, got {joint_angles.dim()}-D")
    over = (joint_angles - upper).clamp_min(0.0)
    under = (lower - joint_angles).clamp_min(0.0)
    return (over.pow(2) + under.pow(2)).mean()


def rotation_orthogonality_regularizer(rot6d: Tensor) -> Tensor:
    """Encourage the 6D vector's first two columns to be orthonormal.

    The Gram-Schmidt step inside `rot6d_to_rotation_matrix` always produces
    an orthonormal output regardless of the input, but training is more
    stable if the input itself is already close to orthonormal. This
    regularizer pulls `‖a1‖ → 1`, `‖a2‖ → 1`, and `a1·a2 → 0`.

    Args:
        rot6d: Tensor of shape (B, 6).

    Returns:
        Scalar tensor.
    """
    if rot6d.dim() != 2 or rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must be (B, 6), got {tuple(rot6d.shape)}")
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    dot = (a1 * a2).sum(dim=-1)
    norm1 = a1.norm(dim=-1)
    norm2 = a2.norm(dim=-1)
    return (dot.pow(2) + (norm1 - 1.0).pow(2) + (norm2 - 1.0).pow(2)).mean()


def contact_alignment_loss(
    decoded_joints: Tensor,
    target_finger_centroids: Tensor,
    finger_joint_indices: Sequence[int] = DEFAULT_MANO_FINGER_TIP_JOINTS,
) -> Tensor:
    """Mean squared distance from MANO finger tips to target contact centroids.

    The graspauto contact graph head predicts per-finger contact centroids
    (`graph["finger_centroid"]`, shape (B, 5, 3)) from the object geometry.
    This loss encourages the generated MANO hand's finger tips to land at
    those centroids, providing a soft alignment supervision in addition to
    the flow matching loss.

    Args:
        decoded_joints:          (B, 21, 3) — MANO joints from `MangoMANODecoder`.
        target_finger_centroids: (B, 5, 3)  — per-finger centroids from the
                                 graspauto contact graph head.
        finger_joint_indices:    Which 5 joint indices in the 21-joint MANO
                                 model are the finger tips. Default
                                 (4, 8, 12, 16, 20) for (thumb, index, middle,
                                 ring, little).

    Returns:
        Scalar tensor.
    """
    if decoded_joints.dim() != 3 or decoded_joints.shape[1:] != (21, 3):
        raise ValueError(
            f"decoded_joints must be (B, 21, 3), got {tuple(decoded_joints.shape)}"
        )
    if target_finger_centroids.dim() != 3 or target_finger_centroids.shape[1:] != (5, 3):
        raise ValueError(
            f"target_finger_centroids must be (B, 5, 3), got "
            f"{tuple(target_finger_centroids.shape)}"
        )
    if len(finger_joint_indices) != 5:
        raise ValueError(f"finger_joint_indices must have 5 entries, got {len(finger_joint_indices)}")

    indices = torch.tensor(
        list(finger_joint_indices), dtype=torch.long, device=decoded_joints.device
    )
    finger_tips = decoded_joints.index_select(dim=1, index=indices)  # (B, 5, 3)
    diffs = finger_tips - target_finger_centroids
    return diffs.pow(2).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Loss bundle
# ---------------------------------------------------------------------------

class MangoLossBundle:
    """Compute the full weighted training loss for one batch.

    The bundle is a callable that takes the per-component raw losses (or
    enough information to compute them) and returns a dict of weighted
    components plus the total. It is stateless beyond the schedule.
    """

    def __init__(self, schedule: Optional[LossSchedule] = None):
        self.schedule = schedule if schedule is not None else LossSchedule()

    def __call__(
        self,
        *,
        epoch: int,
        flow_loss: Tensor,
        rot6d: Tensor,
        joint_angles: Tensor,
        decoded_joints: Optional[Tensor] = None,
        target_finger_centroids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Combine the four loss terms into a weighted total.

        Args:
            epoch:                   Training epoch, used by the curriculum.
            flow_loss:               Pre-computed flow matching loss (scalar).
                                     Caller computes this via
                                     `ConditionalFlowMatching.loss(...)`.
            rot6d:                   (B, 6) — first 6 dims of the predicted MANO
                                     parameters, used for the orthogonality regularizer.
            joint_angles:            (B, 45) — last 45 dims of the predicted MANO
                                     parameters, used for the joint limit hinge.
            decoded_joints:          Optional (B, 21, 3) MANO joints from the decoder.
                                     If None, contact alignment loss is skipped.
            target_finger_centroids: Optional (B, 5, 3) target centroids from
                                     the graspauto contact graph head. If None,
                                     contact alignment loss is skipped.

        Returns:
            dict[str, Tensor] with keys "flow", "contact_align", "joint_limit",
            "rotation_orth", "total". Components are already weighted; "total"
            is their sum.
        """
        weights = self.schedule(epoch)
        device = flow_loss.device

        weighted: Dict[str, Tensor] = {
            "flow": weights["flow"] * flow_loss,
            "joint_limit": weights["joint_limit"] * joint_limit_hinge(joint_angles),
            "rotation_orth": weights["rotation_orth"] * rotation_orthogonality_regularizer(rot6d),
        }

        if (
            decoded_joints is not None
            and target_finger_centroids is not None
            and weights["contact_align"] > 0.0
        ):
            weighted["contact_align"] = weights["contact_align"] * contact_alignment_loss(
                decoded_joints, target_finger_centroids
            )
        else:
            weighted["contact_align"] = torch.zeros((), device=device)

        weighted["total"] = sum(weighted.values())  # type: ignore[assignment]
        return weighted


__all__ = [
    "DEFAULT_MANO_FINGER_TIP_JOINTS",
    "DEFAULT_JOINT_LOWER",
    "DEFAULT_JOINT_UPPER",
    "LossSchedule",
    "joint_limit_hinge",
    "rotation_orthogonality_regularizer",
    "contact_alignment_loss",
    "MangoLossBundle",
]
