"""Light rigid micro-refine for the graspauto inference path.

After the conditional flow matching decoder produces a 54-D MANO parameter
vector, we run a short gradient descent on the *rigid* slots only —
the 6D rotation (slice [0:6]) and the global translation (slice [6:9]) —
keeping the 45D hand pose (slice [9:54]) frozen. The objective is to
minimize the distance from MANO finger tips to the predicted contact
centroids (and optionally SDF-based penetration), tightening the alignment
without disturbing the learned hand pose.

This is the analog of graspauto's `refine_rigid_pose` but adapted to the
graspauto parameter layout. graspauto's version operated on point clouds
directly (weighted ICP with init_rot, init_trans, source_points,
target_points); graspauto's version operates on the full MANO parameter
vector and re-decodes the hand mesh on every gradient step.

Per the paper.5 step 5: "Light rigid
micro-refine (~50 gradient steps) on the rigid (translation + rotation)
parameters only, minimizing penetration + contact alignment to target.
Joint angles are frozen during this step."

Future extensions (not in this stub):
- Penetration loss via object SDF: add a `penetration_fn(vertices, sdf)`
  callable to the loss.
- Force-closure regularizer.
- Per-finger weighting based on the active_finger_prob from the contact
  graph head (currently all 5 fingers are weighted equally).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from graspauto.losses import (
    DEFAULT_MANO_FINGER_TIP_JOINTS,
    contact_alignment_loss,
)
from graspauto.mano_decoder import (
    MANO_PARAM_DIM,
    POSE_SLICE,
    ROT6D_SLICE,
    TRANSLATION_SLICE,
    MangoMANODecoder,
)


@dataclass
class RigidRefineResult:
    """Output of `rigid_micro_refine`.

    Attributes:
        refined_params: (B, 54) — the post-refine MANO parameter vector.
                         Same shape as the input. Joint-angle slots are
                         identical to the input; only rot6d + translation
                         slots have been updated.
        initial_loss:   Scalar Tensor — alignment loss before refinement.
        final_loss:     Scalar Tensor — alignment loss after refinement.
        steps:          Number of gradient steps actually taken.
    """

    refined_params: Tensor
    initial_loss: Tensor
    final_loss: Tensor
    steps: int


def rigid_micro_refine(
    mano_params: Tensor,
    target_finger_centroids: Tensor,
    decoder: MangoMANODecoder,
    *,
    num_steps: int = 50,
    lr: float = 1e-2,
    finger_joint_indices=DEFAULT_MANO_FINGER_TIP_JOINTS,
) -> RigidRefineResult:
    """Refine only the rigid slots of a MANO parameter vector.

    Algorithm:
    1. Split the input vector into (rot6d, translation, hand_pose).
    2. Mark rot6d and translation as `requires_grad=True`. Keep hand_pose
       as a constant (no grad).
    3. Run `num_steps` of Adam updates minimizing
       `contact_alignment_loss(decoder(reassembled_params).joints, target_finger_centroids)`.
    4. Return the refined parameter vector and before/after loss.

    Args:
        mano_params:             (B, 54) — initial MANO parameters from the flow decoder.
        target_finger_centroids: (B, 5, 3) — target centroids from the
                                 graspauto contact graph head.
        decoder:                 A `MangoMANODecoder` instance (must be on the
                                 same device as `mano_params`).
        num_steps:               Number of Adam steps. Default 50.
        lr:                      Adam learning rate. Default 1e-2.
        finger_joint_indices:    Which 5 MANO joints are the fingertips.

    Returns:
        `RigidRefineResult`.
    """
    if mano_params.dim() != 2 or mano_params.shape[-1] != MANO_PARAM_DIM:
        raise ValueError(
            f"mano_params must be (B, {MANO_PARAM_DIM}), got {tuple(mano_params.shape)}"
        )
    if target_finger_centroids.dim() != 3 or target_finger_centroids.shape[1:] != (5, 3):
        raise ValueError(
            f"target_finger_centroids must be (B, 5, 3), got "
            f"{tuple(target_finger_centroids.shape)}"
        )
    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}")

    device = mano_params.device
    dtype = mano_params.dtype

    # Split into mutable rigid slots and frozen pose slot
    rot6d = mano_params[:, ROT6D_SLICE].detach().clone().requires_grad_(True)
    trans = mano_params[:, TRANSLATION_SLICE].detach().clone().requires_grad_(True)
    hand_pose = mano_params[:, POSE_SLICE].detach().clone()  # frozen, no grad

    target = target_finger_centroids.detach()

    optimizer = torch.optim.Adam([rot6d, trans], lr=lr)

    def reassemble() -> Tensor:
        """Reassemble (rot6d, trans, frozen pose) into a (B, 54) vector."""
        return torch.cat([rot6d, trans, hand_pose], dim=-1)

    def compute_loss() -> Tensor:
        params = reassemble()
        out = decoder(params)
        return contact_alignment_loss(
            out["joints"], target, finger_joint_indices=finger_joint_indices
        )

    with torch.enable_grad():
        initial_loss = compute_loss().detach()
        final_loss = initial_loss

        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()

    refined_params = torch.cat(
        [rot6d.detach(), trans.detach(), hand_pose.detach()], dim=-1
    ).to(device=device, dtype=dtype)

    return RigidRefineResult(
        refined_params=refined_params,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=int(num_steps),
    )


__all__ = [
    "RigidRefineResult",
    "rigid_micro_refine",
]
