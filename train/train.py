#!/usr/bin/env python3
"""graspauto training entrypoint — Step 8 smoke test version.

Wires together the six graspauto modules (flow_matching, mano_decoder,
conditioning, velocity_network, losses, refine) plus the existing
graspauto contact graph head, and runs a conditional flow matching training
loop on the existing graspauto dataset (`Stage3ContactGraphDataset`).

Smoke test usage:

    .venv/bin/python train/train.py \\
        --tag mango_smoke_2026-04-11 \\
        --epochs 1 \\
        --train-limit 8 --val-limit 4 \\
        --batch-size 2 --num-workers 0 \\
        --device auto

This is the Step 8 deliverable per the paper
prove that the new flow-matching pipeline can do a forward + backward + eval
+ checkpoint-save round trip on real data with no crashes. Real training
runs (Phase 2 / Phase 3 from the paper) will subsequently use this
script with full `--epochs` / `--batch-size` / proper warm-start from a
graspauto contact-head checkpoint.

Architecture being exercised:

  Stage3ContactGraphDataset (graspauto data)
    ↓
  PointM2AEContactGraphModel (graspauto contact graph head, FROZEN)
    ↓ produces: (graph dict, active_finger_prob, m2ae_local)
  ContactGraphConditioningAdapter (graspauto, trainable)
    ↓ produces: ConditioningBundle (B, 71, hidden_dim)
  VelocityNetwork (graspauto, trainable)
    ↓ outputs: velocity prediction (B, 54)
  ConditionalFlowMatching kernel
    ↓
  MangoLossBundle
    ↓
  backward → optimizer.step

For Step 8 (smoke test) the contact head is initialized from random
(no warm-start). Phase 2 of the paper will add warm-start from
`outputs/stage3_metric_reset_short1/best.pt`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

# graspauto imports — for the data + frozen contact graph head
from graspauto.stage3_contact_graph import (  # noqa: E402
    DEFAULT_CODEBANK_METADATA,
    DEFAULT_OBJECT_M2AE_CACHE,
    DEFAULT_STAGE3_CONTACT_GRAPH_ROOT,
    PointM2AEContactGraphModel,
    Stage3ContactGraphDataset,
)
from graspauto.utils import DEFAULT_GEOMETRY_CACHE, ensure_dir, resolve_device, set_seed, write_json  # noqa: E402

# graspauto imports — the new modules implemented in steps 1-7
from graspauto.conditioning import ContactGraphConditioningAdapter, NUM_OBJECT_PATCHES  # noqa: E402
from graspauto.flow_matching import ConditionalFlowMatching  # noqa: E402
from graspauto.losses import LossSchedule, MangoLossBundle  # noqa: E402
from graspauto.mano_decoder import (  # noqa: E402
    MANO_PARAM_DIM,
    POSE_SLICE,
    ROT6D_SLICE,
    TRANSLATION_SLICE,
    MangoMANODecoder,
    rotation_matrix_to_rot6d,
)
from graspauto.velocity_network import VelocityNetwork  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="graspauto_smoke")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reinit-palm-proj", action="store_true",
                        help="Re-initialise adapter.palm_proj after warm-start (use with geometric palm tokens).")
    parser.add_argument("--palm-proj-lr-mult", type=float, default=1.0,
                        help="Multiplier on --lr for adapter.palm_proj parameters. 1.0 disables layered LR.")
    parser.add_argument("--adapter-lr-mult", type=float, default=1.0,
                        help="Multiplier on --lr for ALL adapter parameters (palm_proj + rest). "
                             "Stacks with palm_proj_lr_mult for palm_proj only.")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--num-flow-steps", type=int, default=10)

    # Dataset paths (default to existing graspauto cache locations)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_STAGE3_CONTACT_GRAPH_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_oracle.pt"))
    parser.add_argument("--val-split", type=Path, default=Path("val_oracle.pt"))
    parser.add_argument("--geometry-path", type=Path, default=DEFAULT_GEOMETRY_CACHE)
    parser.add_argument("--object-cache", type=Path, default=DEFAULT_OBJECT_M2AE_CACHE)
    parser.add_argument("--codebank-metadata", type=Path, default=DEFAULT_CODEBANK_METADATA)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    # Optional warm-start for the graspauto contact head. If set, loads
    # `model_state` from the given checkpoint with strict=False (so the
    # 12 retrieval-only keys in the graspauto checkpoint are silently
    # dropped). Used for Phase 2 of the paper
    parser.add_argument("--warm-start-pear-v1", type=Path, default=None,
                        help="Path to a graspauto contact-graph checkpoint (e.g., outputs/stage3_metric_reset_short1/best.pt). "
                             "If set, loads model_state into the contact head before training. Phase 2+.")
    parser.add_argument("--teacher-forcing", action="store_true",
                        help="Use GROUND-TRUTH contact graph (finger/palm/unified centroids, normals, etc.) "
                             "from the dataset batch as the conditioning graph, instead of the frozen "
                             "contact head's predicted graph. Breaks the information floor where object-only "
                             "conditioning is ambiguous about which of many valid grasps per object to produce. "
                             "Applies to BOTH training and validation (so val_flow reflects the teacher-forced "
                             "floor). Eval still uses the predicted graph (the inference-time path). Phase 2c+.")
    parser.add_argument("--cfg-dropout", type=float, default=0.0,
                        help="Classifier-free guidance dropout probability. During training, with this "
                             "probability the conditioning bundle's tokens are zeroed out for each batch "
                             "element, teaching the velocity network both conditional and unconditional "
                             "modes. At inference, pass --cfg-scale >1 to eval_graspauto.py to sharpen "
                             "generations toward the conditional mode. Standard values: 0.1–0.2. "
                             "0.0 disables CFG (default).")
    parser.add_argument("--physics-loss-weight", type=float, default=0.0,
                        help="Weight for physics-aware energy loss (Grok #1). Computes differentiable "
                             "penetration + surface distance on the PREDICTED x1 (decoded through MANO) "
                             "and adds it to the total loss. Sharpens the flow distribution toward "
                             "physically valid grasps. 0.0 disables. Recommended: 0.1-1.0.")
    parser.add_argument("--naturalness-codebook", type=Path, default=None,
                        help="Path to a latent codebook (.pt with 'centers' key). When set, adds a "
                             "commitment-like loss that pulls the flow's predicted latent toward the "
                             "nearest codebook center. This teaches the flow to generate latents that "
                             "correspond to known natural grasp patterns. Only works with --latent-ae-ckpt.")
    parser.add_argument("--naturalness-weight", type=float, default=0.1,
                        help="Weight for the naturalness commitment loss. 0.1 = gentle pull.")
    parser.add_argument("--min-of-k", type=int, default=0,
                        help="Min-of-K training (Grok #2). Sample K different noise vectors per "
                             "condition, compute flow loss for each, back-propagate only the best. "
                             "Directly optimizes single-shot quality. 0 = disabled. Recommended: 4. "
                             "Training is ~K× slower per step.")
    parser.add_argument("--holdout-objects", type=str, default="",
                        help="Comma-separated list of object_ids to EXCLUDE from training "
                             "(and from the validation pool used during training). Used for "
                             "held-out-object generalization experiments. At eval time, set "
                             "--only-objects on eval_graspauto.py to the same list to test on "
                             "the excluded objects only.")
    parser.add_argument("--residual-modulation", action="store_true",
                        help="Replace scalar multiply gate with residual modulation (GPT Pro #1): "
                             "token = token × (1+α×w) + β×w×tap_embed. Adds z_near/z_far/z_global "
                             "summary tokens. Preserves patch magnitude info. Requires --part-aware-gating.")
    parser.add_argument("--part-aware-gating", action="store_true",
                        help="Enable part-aware spatial gating on object patch tokens. "
                             "Computes a per-patch contact weight by aggregating the per-point "
                             "contact target (unified_contact_target or stage1_contact_input) "
                             "via nearest-neighbor, then multiplies object tokens by this weight "
                             "so patches far from the grasp region are silenced.")
    parser.add_argument("--topk-gate", type=int, default=0,
                        help="r023+: if >0, replace the soft part-aware Gaussian with HARD top-K "
                             "mask (keep only closest-K patches to tap, zero rest). Forces "
                             "locality — tests 'handle-invariance': axe/hammer/wrench share the "
                             "same handle geometry so should produce the same grasp.")
    parser.add_argument("--intent-token", action="store_true",
                        help="Add a user-intent token (use/handoff) to the conditioning bundle. "
                             "Trained from the dataset's intent labels; at inference the user "
                             "provides the intent explicitly via VR UI.")
    parser.add_argument("--palm-only-intent", action="store_true",
                        help="Simplified intent mode for VR tap-to-grasp: condition the model on "
                             "the object point cloud + a SINGLE 3D palm target point (no finger "
                             "centroids, no unified contact, no intent category). At training the "
                             "palm target is the GT palm centroid; at inference the user taps a "
                             "point on the object. Automatically disables --intent-token since "
                             "it is dataset-specific.")
    parser.add_argument("--advanced-gating", action="store_true",
                        help="Use multi-scale log-domain attention bias + sinusoidal palm-offset "
                             "positional encoding on object patch tokens. Replaces the scalar "
                             "multiply gate of --part-aware-gating with a biases-only scheme that "
                             "lets cross-attention decide which patches to attend to at multiple "
                             "distance scales (3/6/12 cm). Requires --palm-only-intent or at least "
                             "a palm target signal in the batch.")
    parser.add_argument("--palm-jitter-std", type=float, default=0.0,
                        help="Gaussian noise stddev (meters) added to `palm_centroid` during "
                             "TRAINING only. Used with --palm-only-intent to both (a) simulate "
                             "user tap inaccuracy at inference and (b) regularize the generator "
                             "against the mild overfit seen in r020 (train_flow=0.19 vs val=0.24 "
                             "at ep400 on 22 train objects). 0.0 disables. Typical: 0.005 (5 mm).")
    parser.add_argument("--use-unified-intent", action="store_true",
                        help="(sphere) rebind `palm_centroid` to `unified_centroid` "
                             "(palm + finger joint-weighted contact centroid). Makes the 'intent' "
                             "more general — user taps where hand as a whole should make contact, "
                             "not specifically palm. Other palm_* features (normal, spread, entropy, "
                             "mass) are kept as palm-region context. Applied in both train and val.")
    parser.add_argument("--use-approach-direction", action="store_true",
                        help="[DEPRECATED r002] Rebinds palm_normal with world-space hand forward axis. "
                             "FAILED — destroyed surface-normal signal, regressed all metrics. "
                             "Prefer --use-intent-direction (r003+).")
    parser.add_argument("--direction-mask-prob", type=float, default=0.5,
                        help="Probability of masking direction token per sample during TRAINING. "
                             "Eval always uses direction. Applies to both r002 approach-direction "
                             "and r003 intent-direction paths.")
    parser.add_argument("--use-intent-direction", action="store_true",
                        help="(sphere) add intent_direction as a NEW conditioning token "
                             "(does NOT overwrite palm_normal). Semantic: direction = "
                             "normalize(unified_centroid - hTm_trans), i.e., 'from where the hand "
                             "base is toward where the user tapped'. Goes through direction_proj "
                             "in the adapter; masked samples use a learned direction_absent "
                             "embedding (not zero — preserves token magnitude).")
    parser.add_argument("--mix-oakink", action="store_true",
                        help="Interleave OakInk batches during ContactPose training. "
                             "Uses data/oakink/v4_cache/{train,val}.pt + m2ae_cache.pt. "
                             "OakInk is ContactPose-quality (1.14 mm mean contact) with 100 objects, "
                             "so mix-ratio can be higher than GraspXL's 10%%.")
    parser.add_argument("--oakink-mix-ratio", type=float, default=0.30,
                        help="Fraction of batches that are OakInk (default 0.30).")
    parser.add_argument("--oakink-cache", type=Path, default=Path("data/oakink/v4_cache/train.pt"))
    parser.add_argument("--oakink-m2ae-cache", type=Path, default=Path("data/oakink/v4_cache/m2ae_cache.pt"))
    parser.add_argument("--oakink-palm-centroid-path", type=Path, default=None,
                        help="Override OakInk's palm_centroids_{side}.pt path (defaults to cache-adjacent).")
    parser.add_argument("--oakink-palm-features-path", type=Path, default=None,
                        help="Override OakInk's palm_features*.pt path (defaults to v2/v1 auto-discovery).")
    parser.add_argument("--oakink-quality-mm", type=float, default=None,
                        help="Filter OakInk to grasps with min_dist_mm < this value (meters).")
    parser.add_argument("--oakink-max-samples", type=int, default=None,
                        help="Cap OakInk sample count (for testing).")
    parser.add_argument("--use-sphere-intent", action="store_true",
                        help="(sphere) 'hand as a sphere on the object' conditioning. "
                             "Extends --use-unified-intent by ALSO swapping palm_spread with a "
                             "scalar radius (= ||unified_spread||) broadcast into 3 slots, and "
                             "palm_entropy with unified_entropy. Architecture unchanged (still "
                             "palm-only intent, 65 tokens); only the semantic of the palm token "
                             "changes from 'palm-specific contact' to 'whole-hand sphere patch'.")
    parser.add_argument("--tf-dropout", type=float, default=0.0,
                        help="Teacher-forcing dropout probability. During TRAINING, with this "
                             "probability each batch element uses the frozen contact head's "
                             "PREDICTED graph instead of the GT graph. This closes the "
                             "conditioning mismatch: the model learns to handle both GT-quality "
                             "and predicted-quality contact features, so at inference (where "
                             "only predicted features are available) the distribution is matched. "
                             "Requires --teacher-forcing to be set (so the default path is GT). "
                             "0.0 = always GT (original behavior). 0.5 = 50/50 mix. 1.0 = always "
                             "predicted (equivalent to disabling teacher-forcing entirely).")
    parser.add_argument("--latent-ae-ckpt", type=Path, default=None,
                        help="Path to a trained MANOAutoEncoder checkpoint (from train_mano_ae.py). "
                             "When set, the flow operates in the AE's latent space instead of raw "
                             "54-D MANO: targets are ae.encode(normalize(mano_params)), the "
                             "VelocityNetwork input_dim is set to the AE's latent_dim, and noise "
                             "is sampled in latent space. At inference, flow output is decoded via "
                             "ae.decode + denormalize + MangoMANODecoder. Grok #2 approach.")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Two-stage hierarchical flow: Stage 1 generates wrist pose (rot6D + "
                             "translation = 9-D), Stage 2 generates finger joints (45-D) conditioned "
                             "on the Stage 1 wrist output. Each stage uses its own VelocityNetwork. "
                             "Stage 2 receives the wrist as an extra conditioning token. Incompatible "
                             "with --latent-ae-ckpt (operates in raw 54-D space, split into 9+45).")
    parser.add_argument("--warm-start-from", type=Path, default=None,
                        help="Path to a previous graspauto checkpoint to warm-start the adapter "
                             "and velocity network from. Used for GraspXL pretrain → ContactPose "
                             "fine-tune transfer. Loads adapter_state and velocity_net_state with "
                             "strict=False (tolerates minor shape mismatches).")
    parser.add_argument("--graspxl", action="store_true",
                        help="Use GraspXL dataset instead of ContactPose Stage3. Loads processed "
                             "shards from data/graspxl_mango/ + M2AE cache. For pretraining on "
                             "large-scale synthetic data before ContactPose fine-tuning.")
    parser.add_argument("--graspxl-shard-dir", type=Path, default=Path("data/graspxl_mango"))
    parser.add_argument("--graspxl-m2ae-cache", type=Path, default=Path("data/graspxl_mango/m2ae_cache.pt"))
    parser.add_argument("--graspxl-shards", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14",
                        help="Comma-separated shard IDs to use. Default: all 15.")
    parser.add_argument("--graspxl-max-samples", type=int, default=None,
                        help="Limit total GraspXL samples (for testing)")
    parser.add_argument("--mix-graspxl", action="store_true",
                        help="Mixed training: each epoch alternates ContactPose and GraspXL batches. "
                             "Epochs are counted on ContactPose; GraspXL batches are sampled alongside. "
                             "Requires GraspXL data at --graspxl-shard-dir.")
    parser.add_argument("--mix-ratio", type=float, default=0.25,
                        help="Fraction of batches that are GraspXL (default 0.25 = 1 GraspXL per 3 CP)")
    return parser.parse_args()


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices.

    Uses `torch.linalg.matrix_exp` of the skew-symmetric matrix form, which
    is exactly the closed-form Rodrigues result without manual trig.
    """
    if aa.shape[-1] != 3:
        raise ValueError(f"axis-angle last dim must be 3, got {aa.shape}")
    skew = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    skew[..., 0, 1] = -aa[..., 2]
    skew[..., 0, 2] =  aa[..., 1]
    skew[..., 1, 0] =  aa[..., 2]
    skew[..., 1, 2] = -aa[..., 0]
    skew[..., 2, 0] = -aa[..., 1]
    skew[..., 2, 1] =  aa[..., 0]
    return torch.linalg.matrix_exp(skew)


# MANO right-hand rest wrist position with center_idx=None, flat_hand_mean=True,
# zero pose, zero betas. Computed empirically 2026-04-11 against the project's
# `assets/mano_v1_2/MANO_RIGHT.pkl`. Used by build_target_mano_params to
# compensate the wrist-pivot offset that arises when MANO's global orient is
# applied at the wrist joint INSIDE MANO but graspauto applies its rotation
# externally about the world origin.
MANO_RIGHT_REST_WRIST = torch.tensor(
    [0.0956699401140213, 0.0063834283500909805, 0.006186304613947868],
    dtype=torch.float32,
)


def build_target_mano_params(batch: dict) -> torch.Tensor:
    """Convert the dataset's GT MANO into graspauto's 54-D target vector.

    Layout: [rot6d (6), translation (3), hand_pose (45)] per `mano_decoder.py`.

    Derivation (verified empirically to give 0.009 mm GT-replay error):

    The dataset stores `gt_world_verts` as

        gt_world_verts = MANO(pose_48, betas, center_idx=None) @ hTm_rot.T + hTm_trans

    where MANO applies the global orient (`pose_48[:, 0:3]`) at the wrist
    joint INSIDE the kinematic chain (i.e., as a rotation about a non-zero
    pivot point `w` = the rest wrist position).

    graspauto's MangoMANODecoder, however, takes (rot6d, translation,
    hand_pose) and computes:

        decoder_out = MANO(zero_global + hand_pose, betas, center_idx=None)
                          @ rot6d_to_matrix(rot6d).T + translation

    The decoder's rotation is applied externally to the vertices about the
    world origin, NOT at the wrist. To make `decoder_out == gt_world_verts`,
    the target's (rot6d, translation) must compensate for the pivot mismatch:

        R_g     = axis_angle_to_matrix(pose_48[:, 0:3])
        rot6d   = rotation_matrix_to_rot6d(hTm_rot @ R_g)
        trans   = (I - R_g) @ w @ hTm_rot.T + hTm_trans
        hand_pose = pose_48[:, 3:48]

    where `w` is the constant rest wrist position MANO_RIGHT_REST_WRIST.

    This formula was derived analytically and verified to produce
    GT-replay error of 0.009 mm (vs 96 mm with the naive identity formula
    that just used hTm_rot directly). See commit log around the Phase 1
    debug session 2026-04-11.
    """
    pose_48 = batch["pose_48"]
    hTm_rot = batch["hTm_rot"]                                              # (B, 3, 3)
    hTm_trans = batch["hTm_trans"]                                          # (B, 3)

    R_g = axis_angle_to_matrix(pose_48[:, 0:3])                             # (B, 3, 3)
    total_rot = torch.bmm(hTm_rot, R_g)                                     # (B, 3, 3)
    rot6d = rotation_matrix_to_rot6d(total_rot)                             # (B, 6)

    # Pivot compensation: (I - R_g) @ w @ hTm_rot.T
    # w is a (3,) constant; broadcast across the batch.
    w = MANO_RIGHT_REST_WRIST.to(device=hTm_rot.device, dtype=hTm_rot.dtype)
    eye = torch.eye(3, device=hTm_rot.device, dtype=hTm_rot.dtype)          # (3, 3)
    eye_minus_Rg = eye.unsqueeze(0) - R_g                                   # (B, 3, 3)
    # (B, 3) = (B, 3, 3) @ (3,) — broadcast w across batch
    w_b = w.unsqueeze(0).expand(R_g.shape[0], -1)                           # (B, 3)
    pivot_term = torch.bmm(
        torch.bmm(eye_minus_Rg, w_b.unsqueeze(-1)).squeeze(-1).unsqueeze(1),  # (B, 1, 3)
        hTm_rot.transpose(-1, -2),                                            # (B, 3, 3)
    ).squeeze(1)                                                              # (B, 3)
    trans = pivot_term + hTm_trans                                          # (B, 3)

    hand_pose = pose_48[:, 3:48]                                            # (B, 45)
    return torch.cat([rot6d, trans, hand_pose], dim=-1)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_teacher_graph(batch: dict) -> tuple[dict, torch.Tensor]:
    """Build a conditioning graph dict + active_finger_prob from ground-truth batch fields.

    The Stage3ContactGraphDataset stores per-grasp GT contact features that the
    graspauto head is normally trained to predict. Using these directly as the
    conditioning during training lets the velocity network condition on
    *within-object* grasp information (not just object identity), which is
    necessary to break the information-theoretic floor at ~1.072 per dim seen
    with object-only conditioning.

    Returns a (graph_dict, active_finger_prob) tuple in exactly the shape
    `ContactGraphConditioningAdapter.forward(...)` expects.
    """
    graph = {
        # Per-finger: (B, 5, *)
        "finger_centroid": batch["finger_centroid"],
        "finger_normal":   batch["finger_normal"],
        "finger_spread":   batch["finger_spread"],
        "finger_entropy":  batch["finger_entropy"],
        "finger_mass":     batch["finger_mass"],
        # Palm: (B, 3) / (B,)
        "palm_centroid": batch["palm_centroid"],
        "palm_normal":   batch["palm_normal"],
        "palm_spread":   batch["palm_spread"],
        "palm_entropy":  batch["palm_entropy"],
        "palm_mass":     batch["palm_mass"],
        # Unified: (B, 3) / (B,)
        "unified_centroid": batch["unified_centroid"],
        "unified_normal":   batch["unified_normal"],
        "unified_spread":   batch["unified_spread"],
        "unified_entropy":  batch["unified_entropy"],
    }
    # `active_finger_score` is the continuous (0..1) probability target the graspauto
    # active-finger head is trained against; use it as the teacher-forced version
    # of the head's predicted active_finger_prob.
    active_finger_prob = batch["active_finger_score"]
    return graph, active_finger_prob


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # ---- Dataset ----
    if args.graspxl:
        from graspauto.graspxl_dataset import GraspXLDataset  # noqa: PLC0415
        shard_ids = [int(x) for x in args.graspxl_shards.split(",")]
        # Use 90% of shards for train, last shard for val
        train_shard_ids = shard_ids[:-1]
        val_shard_ids = shard_ids[-1:]
        train_ds = GraspXLDataset(
            shard_dir=args.graspxl_shard_dir,
            m2ae_cache_path=args.graspxl_m2ae_cache,
            shard_ids=train_shard_ids,
            max_samples=args.graspxl_max_samples,
        )
        val_ds = GraspXLDataset(
            shard_dir=args.graspxl_shard_dir,
            m2ae_cache_path=args.graspxl_m2ae_cache,
            shard_ids=val_shard_ids,
            max_samples=args.val_limit or 5000,
        )
    else:
        train_split = args.preprocess_root / args.train_split
        val_split = args.preprocess_root / args.val_split
        train_ds = Stage3ContactGraphDataset(
            split_path=train_split,
            geometry_path=args.geometry_path,
            object_m2ae_cache_path=args.object_cache,
            limit=args.train_limit,
        )
        val_ds = Stage3ContactGraphDataset(
            split_path=val_split,
            geometry_path=args.geometry_path,
            object_m2ae_cache_path=args.object_cache,
            limit=args.val_limit,
        )

    # Held-out object filtering for generalization experiments.
    if args.holdout_objects:
        holdout_ids = {int(x) for x in args.holdout_objects.split(",") if x.strip()}
        train_keep = [i for i in range(len(train_ds))
                      if int(train_ds.object_id[i].item()) not in holdout_ids]
        val_keep = [i for i in range(len(val_ds))
                    if int(val_ds.object_id[i].item()) not in holdout_ids]
        from torch.utils.data import Subset  # noqa: PLC0415
        train_ds = Subset(train_ds, train_keep)
        val_ds = Subset(val_ds, val_keep)
        # When we use Subset, the trainer needs num_codes to come from the
        # underlying dataset, not the subset view. Stash a reference.
        train_ds.num_codes = train_ds.dataset.num_codes  # type: ignore[attr-defined]
        print(
            f"[holdout] excluding objects {sorted(holdout_ids)}: "
            f"train {len(train_keep)}/{len(train_ds.dataset)}, val {len(val_keep)}/{len(val_ds.dataset)}",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Mixed training: OakInk ((sphere))
    oakink_loader = None
    if args.mix_oakink:
        from graspauto.oakink_dataset import OakInkDataset  # noqa: PLC0415
        oakink_ds = OakInkDataset(
            cache_path=args.oakink_cache,
            m2ae_cache_path=args.oakink_m2ae_cache,
            max_samples=args.oakink_max_samples,
            quality_filter_mm=args.oakink_quality_mm,
            palm_centroid_path=args.oakink_palm_centroid_path,
            palm_features_path=args.oakink_palm_features_path,
        )
        oakink_loader = DataLoader(
            oakink_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        print(f"[mix] OakInk loaded: {len(oakink_ds)} samples, ratio={args.oakink_mix_ratio}", flush=True)

    # Mixed training: load GraspXL alongside ContactPose
    graspxl_loader = None
    if args.mix_graspxl:
        from graspauto.graspxl_dataset import GraspXLDataset  # noqa: PLC0415
        gx_shard_ids = [int(x) for x in args.graspxl_shards.split(",")]
        gx_ds = GraspXLDataset(
            shard_dir=args.graspxl_shard_dir,
            m2ae_cache_path=args.graspxl_m2ae_cache,
            shard_ids=gx_shard_ids[:3],  # use 3 shards (~150K) to limit memory
            max_samples=args.graspxl_max_samples or 50000,
        )
        graspxl_loader = DataLoader(
            gx_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        print(f"[mix] GraspXL loaded: {len(gx_ds)} samples, ratio={args.mix_ratio}", flush=True)

    num_codes = int(train_ds.num_codes)

    # ---- Models ----
    # graspauto contact graph head — used as a feature provider, frozen.
    # If --warm-start-pear-v1 is provided, load the contact head weights from
    # an existing graspauto checkpoint (Phase 2+); otherwise random-init (Phase 1).
    contact_head = PointM2AEContactGraphModel(num_codes=num_codes).to(device).eval()
    if args.warm_start_graspauto is not None:
        ckpt_path = Path(args.warm_start_graspauto)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"warm-start checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state" not in ckpt:
            raise KeyError(f"checkpoint at {ckpt_path} has no 'model_state' key (got {list(ckpt.keys())[:10]})")
        missing, unexpected = contact_head.load_state_dict(ckpt["model_state"], strict=False)
        print(
            f"[warm-start] loaded graspauto contact head from {ckpt_path}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected (retrieval-only keys, OK to drop)",
            flush=True,
        )
    for p in contact_head.parameters():
        p.requires_grad_(False)

    # ---- Latent flow setup (optional) ----
    latent_ae = None
    latent_mean = None
    latent_std = None
    flow_input_dim = MANO_PARAM_DIM  # default: raw 54-D
    if args.latent_ae_ckpt is not None:
        ae_ckpt = torch.load(args.latent_ae_ckpt, map_location=device, weights_only=False)
        ae_latent_dim = ae_ckpt["latent_dim"]
        ae_hidden_dims = ae_ckpt["hidden_dims"]
        if ae_ckpt.get("residual", False):
            from graspauto.mano_autoencoder import ResidualMANOAutoEncoder  # noqa: PLC0415
            latent_ae = ResidualMANOAutoEncoder(
                input_dim=54, latent_dim=ae_latent_dim,
                hidden_dim=ae_ckpt["res_hidden_dim"], n_blocks=ae_ckpt["res_n_blocks"],
            ).to(device)
        else:
            from graspauto.mano_autoencoder import MANOAutoEncoder  # noqa: PLC0415
            latent_ae = MANOAutoEncoder(
                input_dim=54, latent_dim=ae_latent_dim, hidden_dims=ae_hidden_dims,
            ).to(device)
        latent_ae.load_state_dict(ae_ckpt["model_state"])
        latent_ae.eval()
        for p in latent_ae.parameters():
            p.requires_grad_(False)
        latent_mean = ae_ckpt["train_mean"].to(device)
        latent_std = ae_ckpt["train_std"].to(device)
        flow_input_dim = ae_latent_dim
        print(f"[latent] loaded AE from {args.latent_ae_ckpt}: latent_dim={ae_latent_dim}, "
              f"hidden={ae_hidden_dims}, flow operates in {ae_latent_dim}-D latent space", flush=True)

    # graspauto trainable modules
    adapter = ContactGraphConditioningAdapter(
        hidden_dim=args.hidden_dim,
        use_intent_token=args.intent_token,
        part_aware_gating=args.part_aware_gating,
        palm_only_intent=args.palm_only_intent,
        advanced_gating=args.advanced_gating,
        residual_modulation=getattr(args, 'residual_modulation', False),
        use_intent_direction=getattr(args, 'use_intent_direction', False),
    ).to(device)
    # Hierarchical flow: two velocity networks (wrist 9-D, fingers 45-D)
    velocity_net_finger = None
    wrist_proj = None
    WRIST_DIM = 9   # rot6D(6) + translation(3)
    FINGER_DIM = 45  # hand_pose
    if args.hierarchical:
        velocity_net = VelocityNetwork(
            input_dim=WRIST_DIM,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
        velocity_net_finger = VelocityNetwork(
            input_dim=FINGER_DIM,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
        # Project 9-D wrist into hidden_dim to use as an extra conditioning token
        wrist_proj = nn.Linear(WRIST_DIM, args.hidden_dim).to(device)
    else:
        velocity_net = VelocityNetwork(
            input_dim=flow_input_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)

    # MANO decoder — instantiated lazily so the smoke test still runs even if MANO assets are
    # missing in the smoke environment. We catch the ImportError / FileNotFoundError and
    # fall back to a None decoder, which disables the contact alignment loss for that run.
    try:
        mano_decoder = MangoMANODecoder().to(device)
        mano_decoder.eval()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] MangoMANODecoder unavailable, contact alignment loss disabled: {e}", flush=True)
        mano_decoder = None

    flow = ConditionalFlowMatching()

    # Load naturalness codebook if specified
    _nat_codebook = None
    if args.naturalness_codebook is not None:
        cb_data = torch.load(args.naturalness_codebook, map_location=device, weights_only=False)
        _nat_codebook = cb_data["centers"].to(device)
        print(f"[naturalness] loaded codebook: {_nat_codebook.shape[0]} centers, weight={args.naturalness_weight}", flush=True)

    schedule = LossSchedule()
    loss_bundle = MangoLossBundle(schedule=schedule)

    # Warm-start from a prior graspauto checkpoint (for pretrain→finetune transfer).
    if args.warm_start_from is not None:
        ws_ckpt = torch.load(args.warm_start_from, map_location=device, weights_only=False)
        missing_a, unexpected_a = adapter.load_state_dict(ws_ckpt["adapter_state"], strict=False)
        missing_v, unexpected_v = velocity_net.load_state_dict(ws_ckpt["velocity_net_state"], strict=False)
        print(f"[warm-start-from] {args.warm_start_from}: "
              f"adapter {len(missing_a)} missing {len(unexpected_a)} unexpected, "
              f"velocity_net {len(missing_v)} missing {len(unexpected_v)} unexpected", flush=True)

    # Geometric palm-token O2 mode: reinit the 11-D palm projection because the
    # new palm features have completely different statistics than the CP heatmap
    # features the warm-started weights were learned against. Keeping the old
    # palm_proj would just guarantee a large first-batch loss and slow learning.
    if getattr(args, "reinit_palm_proj", False):
        import torch.nn.init as _init  # noqa: PLC0415
        with torch.no_grad():
            _init.xavier_uniform_(adapter.palm_proj.weight)
            adapter.palm_proj.bias.zero_()
        print(f"[reinit] adapter.palm_proj: xavier_uniform + zero bias "
              f"({adapter.palm_proj.weight.shape})", flush=True)

    # Layered learning rate: the palm projection (if reinited) learns the new
    # geometric-feature language from scratch, while the rest of the adapter
    # and the velocity network are fine-tuned gently. Falls back to a single
    # LR when --palm-proj-lr-mult == 1.0 (the default).
    palm_mult = float(getattr(args, "palm_proj_lr_mult", 1.0))
    adapter_mult = float(getattr(args, "adapter_lr_mult", 1.0))
    if palm_mult != 1.0 or adapter_mult != 1.0:
        palm_proj_params = list(adapter.palm_proj.parameters())
        other_adapter_params = [
            p for name, p in adapter.named_parameters()
            if not name.startswith("palm_proj.")
        ]
        param_groups = [
            {"params": palm_proj_params, "lr": args.lr * palm_mult * adapter_mult, "name": "palm_proj"},
            {"params": other_adapter_params, "lr": args.lr * adapter_mult, "name": "adapter_other"},
            {"params": list(velocity_net.parameters()), "lr": args.lr, "name": "velocity_net"},
        ]
        if velocity_net_finger is not None:
            param_groups.append(
                {"params": list(velocity_net_finger.parameters()) + list(wrist_proj.parameters()),
                 "lr": args.lr, "name": "finger_net"}
            )
        # Flat list of tensor params for grad_clip and param counts; optimiser
        # takes the grouped form for layered LR.
        trainable_params = [p for g in param_groups for p in g["params"]]
        n_trainable = sum(p.numel() for p in trainable_params)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"[lr] layered: palm_proj={args.lr * palm_mult * adapter_mult:.2e} "
              f"(×{palm_mult*adapter_mult:.1f}), "
              f"adapter={args.lr * adapter_mult:.2e} (×{adapter_mult:.1f}), "
              f"velocity={args.lr:.2e}", flush=True)
    else:
        trainable_params = list(adapter.parameters()) + list(velocity_net.parameters())
        if velocity_net_finger is not None:
            trainable_params += list(velocity_net_finger.parameters()) + list(wrist_proj.parameters())
        n_trainable = sum(p.numel() for p in trainable_params)
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} codes={num_codes} "
        f"trainable={n_trainable / 1e6:.2f}M "
        f"contact_head=frozen mano_decoder={'available' if mano_decoder is not None else 'unavailable'}",
        flush=True,
    )

    # ---- Output dir ----
    out_dir = ensure_dir(PROJECT_ROOT / "outputs" / args.tag)
    metrics_log: list[dict] = []

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        adapter.train()
        velocity_net.train()
        train_sums = {"flow": 0.0, "joint_limit": 0.0, "rotation_orth": 0.0, "contact_align": 0.0, "total": 0.0}
        train_samples = 0

        # Build batch iterators: interleave GraspXL / OakInk if mix mode is on
        if graspxl_loader is not None:
            _gx_iter = iter(graspxl_loader)
        if oakink_loader is not None:
            _oi_iter = iter(oakink_loader)

        for raw_batch in train_loader:
            # Mixed training: after each CP batch, with some probability replace
            # with a GraspXL or OakInk batch. Three-way mix: ContactPose /
            # OakInk / GraspXL in ratios (1 - oakink - graspxl) / oakink / graspxl.
            rv = torch.rand(1).item()
            if oakink_loader is not None and rv < args.oakink_mix_ratio:
                try:
                    raw_batch = next(_oi_iter)
                except StopIteration:
                    _oi_iter = iter(oakink_loader)
                    raw_batch = next(_oi_iter)
            elif graspxl_loader is not None and rv < args.oakink_mix_ratio + args.mix_ratio:
                try:
                    raw_batch = next(_gx_iter)
                except StopIteration:
                    _gx_iter = iter(graspxl_loader)
                    raw_batch = next(_gx_iter)
            batch = move_batch_to_device(raw_batch, device)

            # (sphere) unified intent swap. Rebind palm_centroid to
            # unified_centroid BEFORE any downstream use (jitter, part-aware gate,
            # build_teacher_graph). Keeps palm_normal/spread/entropy/mass as-is.
            if args.use_unified_intent:
                batch["palm_centroid"] = batch["unified_centroid"].clone()

            # (sphere) sphere-intent adds radius + entropy swap.
            # palm_spread = ||unified_spread|| broadcast into 3 slots (scalar
            # radius in all three axes = isotropic sphere). palm_entropy =
            # unified_entropy (overall contact density, not palm-only).
            # Requires --use-unified-intent (for centroid swap).
            if args.use_sphere_intent:
                radius = batch["unified_spread"].norm(dim=-1, keepdim=True)  # (B, 1)
                batch["palm_spread"] = radius.expand(-1, 3).contiguous()
                batch["palm_entropy"] = batch["unified_entropy"].clone()

            # (sphere) approach-direction token. Compute world-space
            # hand forward axis per sample (hTm_rot @ [0,0,1]), rebind into
            # palm_normal channel. 50% of samples (by default) have direction
            # masked to zero so the model handles both direction-supplied and
            # direction-absent cases at inference.
            if args.use_approach_direction:
                hTm_rot = batch["hTm_rot"]  # (B, 3, 3)
                fwd_local = torch.tensor([0.0, 0.0, 1.0], device=hTm_rot.device, dtype=hTm_rot.dtype)
                approach_dir = hTm_rot @ fwd_local  # (B, 3), world-space hand forward
                if args.direction_mask_prob > 0.0:
                    mask = torch.rand(hTm_rot.shape[0], device=hTm_rot.device) < args.direction_mask_prob
                    approach_dir = approach_dir.clone()
                    approach_dir[mask] = 0.0
                batch["palm_normal"] = approach_dir

            # Palm-target jitter (training only). Adds isotropic Gaussian noise
            # to the GT palm centroid *before* it is consumed by either the
            # part-aware gate or the advanced_gating path. This simultaneously
            # regularizes the generator and matches the inference-time signal
            # (the user's tap point is not the exact GT centroid). Palm-only
            # intent must be on — otherwise the palm centroid flows into the
            # dense contact-graph conditioning where jitter would corrupt
            # supervision.
            if args.palm_jitter_std > 0.0 and args.palm_only_intent:
                noise = torch.randn_like(batch["palm_centroid"]) * args.palm_jitter_std
                batch["palm_centroid"] = batch["palm_centroid"] + noise

            if args.teacher_forcing:
                # Teacher-forcing dropout: with probability tf_dropout, replace
                # the GT graph with the frozen head's predicted graph for each
                # batch element. This teaches the velocity network to handle
                # both GT-quality and predicted-quality conditioning, closing
                # the train/infer mismatch that causes the ~12mm gap on SEEN
                # objects (identified by Grok external review 2026-04-12).
                use_predicted = False
                if args.tf_dropout > 0.0:
                    # Per-element coin flip: decide GT vs predicted for each sample
                    tf_mask = torch.rand(batch["object_points"].shape[0], device=device) < args.tf_dropout
                    if tf_mask.all():
                        use_predicted = True
                    elif tf_mask.any():
                        # Mixed batch: need both GT and predicted, then blend.
                        # For simplicity, run both paths and select per-element.
                        gt_graph, gt_afp = build_teacher_graph(batch)
                        with torch.no_grad():
                            head_out = contact_head(
                                object_points=batch["object_points"],
                                object_normals=batch["object_normals"],
                                stage1_contact_input=batch["stage1_contact_input"],
                                m2ae_global=batch["m2ae_global"],
                                m2ae_local=batch["m2ae_local"],
                                patch_centers=batch["patch_centers"],
                            )
                        pred_graph = head_out["graph"]
                        pred_afp = head_out["active_finger_prob"]
                        # Blend: where tf_mask is True, use predicted; else GT
                        graph = {}
                        for k in gt_graph:
                            g, p = gt_graph[k], pred_graph[k]
                            mask = tf_mask
                            # Broadcast mask to match tensor dims
                            while mask.dim() < g.dim():
                                mask = mask.unsqueeze(-1)
                            graph[k] = torch.where(mask.expand_as(g), p, g)
                        afp_mask = tf_mask.unsqueeze(-1) if gt_afp.dim() > 1 else tf_mask
                        active_finger_prob = torch.where(afp_mask.expand_as(gt_afp), pred_afp, gt_afp)
                    # else: tf_mask is all False → fall through to pure GT below

                if use_predicted:
                    with torch.no_grad():
                        head_out = contact_head(
                            object_points=batch["object_points"],
                            object_normals=batch["object_normals"],
                            stage1_contact_input=batch["stage1_contact_input"],
                            m2ae_global=batch["m2ae_global"],
                            m2ae_local=batch["m2ae_local"],
                            patch_centers=batch["patch_centers"],
                        )
                    graph = head_out["graph"]
                    active_finger_prob = head_out["active_finger_prob"]
                elif args.tf_dropout == 0.0 or not tf_mask.any():
                    # Pure GT path (original behavior)
                    graph, active_finger_prob = build_teacher_graph(batch)
            else:
                with torch.no_grad():
                    head_out = contact_head(
                        object_points=batch["object_points"],
                        object_normals=batch["object_normals"],
                        stage1_contact_input=batch["stage1_contact_input"],
                        m2ae_global=batch["m2ae_global"],
                        m2ae_local=batch["m2ae_local"],
                        patch_centers=batch["patch_centers"],
                    )
                graph = head_out["graph"]
                active_finger_prob = head_out["active_finger_prob"]

            # Part-aware gating weights (optional)
            patch_contact_weight = None
            if args.part_aware_gating:
                if args.palm_only_intent:
                    from graspauto.conditioning import compute_patch_weight_from_point  # noqa: PLC0415
                    # Gate patches by Gaussian distance to the palm target. The
                    # user's "tap" (or the GT palm centroid at training time)
                    # defines a spatial neighborhood for attention.
                    patch_contact_weight = compute_patch_weight_from_point(
                        patch_centers=batch["patch_centers"],
                        target_point=batch["palm_centroid"],
                    )
                else:
                    from graspauto.conditioning import compute_patch_contact_weight  # noqa: PLC0415
                    # When tf_dropout is active and this element got predicted graph,
                    # the contact_mask should also come from the predicted signal.
                    # For simplicity, use unified_contact_target when TF is on
                    # (it's the GT), and stage1_contact_input otherwise. In mixed
                    # batches the gating signal won't be per-element-matched, but
                    # the part-aware gate is a soft multiplier so this is fine.
                    patch_contact_weight = compute_patch_contact_weight(
                        patch_centers=batch["patch_centers"],
                        object_points=batch["object_points"],
                        contact_mask=batch["unified_contact_target"] if args.teacher_forcing else batch["stage1_contact_input"],
                    )
                # Part-aware locality: hard top-K mask forces model to attend
                # ONLY to the closest-K patches to the tap. Tests "handle-like"
                # part-invariance: axe/hammer/wrench should produce same grasp
                # because the local handle geometry is what matters.
                if args.topk_gate > 0 and patch_contact_weight is not None:
                    _vals, _idx = patch_contact_weight.topk(args.topk_gate, dim=-1)
                    _mask = torch.zeros_like(patch_contact_weight)
                    _mask.scatter_(-1, _idx, 1.0)
                    patch_contact_weight = _mask

            intent_ids = batch["intent_id"] if args.intent_token else None

            adv_patch_centers = batch["patch_centers"] if args.advanced_gating else None
            adv_palm_target = batch["palm_centroid"] if args.advanced_gating else None

            # (sphere) compute intent_direction + 50% mask (training only).
            train_intent_direction = None
            train_direction_mask = None
            if args.use_intent_direction:
                # approach_vec = normalize(unified_centroid - hTm_trans)
                # Semantic: "from user's hand base toward their tap on the object".
                _anchor = batch["hTm_trans"]  # (B, 3)
                _tap = batch["unified_centroid"]  # (B, 3)
                _dir = _tap - _anchor
                _dir = _dir / _dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                train_intent_direction = _dir
                if args.direction_mask_prob > 0.0:
                    train_direction_mask = torch.rand(_dir.shape[0], device=_dir.device) < args.direction_mask_prob

            bundle = adapter(
                m2ae_local=batch["m2ae_local"],
                graph=graph,
                active_finger_prob=active_finger_prob,
                patch_contact_weight=patch_contact_weight,
                intent_ids=intent_ids,
                patch_centers=adv_patch_centers,
                palm_target=adv_palm_target,
                intent_direction=train_intent_direction,
                direction_mask=train_direction_mask,
            )

            # Classifier-free guidance dropout: with probability cfg_dropout,
            # zero out the conditioning for a batch element. The velocity
            # network then sees both conditional and unconditional training
            # examples, so at inference it can extrapolate "conditional minus
            # unconditional" to sharpen generations.
            if args.cfg_dropout > 0.0:
                drop_mask = (torch.rand(bundle.tokens.shape[0], device=device) < args.cfg_dropout)
                if drop_mask.any():
                    new_tokens = bundle.tokens.clone()
                    new_tokens[drop_mask] = 0.0
                    bundle = bundle.__class__(
                        tokens=new_tokens,
                        object_tokens=new_tokens[:, :64, :],
                        contact_tokens=new_tokens[:, 64:, :],
                        token_types=bundle.token_types,
                    )

            mano_x1 = build_target_mano_params(batch)
            if args.hierarchical:
                # Hierarchical flow: Stage 1 (wrist 9-D) + Stage 2 (fingers 45-D)
                wrist_x1 = mano_x1[:, :WRIST_DIM]       # (B, 9)
                finger_x1 = mano_x1[:, WRIST_DIM:]      # (B, 45)

                # Stage 1: flow on wrist
                wrist_x0 = torch.randn_like(wrist_x1)
                wrist_fb = flow.prepare_batch(wrist_x0, wrist_x1)
                wrist_pred_v = velocity_net(wrist_fb.xt, wrist_fb.t, condition=bundle)
                wrist_loss = flow.loss(wrist_pred_v, wrist_fb.target_velocity)

                # Stage 2: flow on fingers, conditioned on GENERATED wrist
                # (not GT) to eliminate train/infer distribution mismatch.
                # Run Stage 1 flow sampling (detached) to get a realistic wrist.
                with torch.no_grad():
                    wrist_sampled = flow.sample(
                        velocity_net, torch.randn_like(wrist_x1),
                        condition=bundle, num_steps=10, method="euler",
                    )
                wrist_token = wrist_proj(wrist_sampled).unsqueeze(1)  # (B, 1, H)
                stage2_tokens = torch.cat([bundle.tokens, wrist_token], dim=1)
                from graspauto.conditioning import ConditioningBundle as _CB  # noqa: PLC0415
                stage2_bundle = _CB(
                    tokens=stage2_tokens,
                    object_tokens=bundle.object_tokens,
                    contact_tokens=bundle.contact_tokens,
                    token_types=torch.cat([
                        bundle.token_types,
                        torch.full((1,), 5, dtype=torch.long, device=device),  # type 5 = wrist token
                    ]),
                )

                finger_x0 = torch.randn_like(finger_x1)
                finger_fb = flow.prepare_batch(finger_x0, finger_x1)
                finger_pred_v = velocity_net_finger(finger_fb.xt, finger_fb.t, condition=stage2_bundle)
                finger_loss = flow.loss(finger_pred_v, finger_fb.target_velocity)

                flow_loss = wrist_loss + finger_loss
                # For auxiliary losses, use the full x1
                x1 = mano_x1
            elif latent_ae is not None:
                # Encode into latent space: normalize → encode → flow target
                with torch.no_grad():
                    normed = (mano_x1 - latent_mean) / latent_std
                    x1 = latent_ae.encode(normed)

                if args.min_of_k > 1:
                    # Min-of-K: sample K noise vectors, pick the one with lowest flow loss
                    best_flow_loss = None
                    best_pred_velocity = None
                    best_fb = None
                    for _k in range(args.min_of_k):
                        x0_k = torch.randn_like(x1)
                        fb_k = flow.prepare_batch(x0_k, x1)
                        pv_k = velocity_net(fb_k.xt, fb_k.t, condition=bundle)
                        fl_k = flow.loss(pv_k, fb_k.target_velocity)
                        if best_flow_loss is None or fl_k.item() < best_flow_loss:
                            best_flow_loss = fl_k.item()
                            best_pred_velocity = pv_k
                            best_fb = fb_k
                            flow_loss = fl_k
                    pred_velocity = best_pred_velocity
                    fb = best_fb
                else:
                    x0 = torch.randn_like(x1)
                    fb = flow.prepare_batch(x0, x1)
                    pred_velocity = velocity_net(fb.xt, fb.t, condition=bundle)
                    flow_loss = flow.loss(pred_velocity, fb.target_velocity)
            else:
                x1 = mano_x1
                if args.min_of_k > 1:
                    best_flow_loss = None
                    for _k in range(args.min_of_k):
                        x0_k = torch.randn_like(x1)
                        fb_k = flow.prepare_batch(x0_k, x1)
                        pv_k = velocity_net(fb_k.xt, fb_k.t, condition=bundle)
                        fl_k = flow.loss(pv_k, fb_k.target_velocity)
                        if best_flow_loss is None or fl_k.item() < best_flow_loss:
                            best_flow_loss = fl_k.item()
                            pred_velocity = pv_k
                            fb = fb_k
                            flow_loss = fl_k
                else:
                    x0 = torch.randn_like(x1)
                    fb = flow.prepare_batch(x0, x1)
                    pred_velocity = velocity_net(fb.xt, fb.t, condition=bundle)
                    flow_loss = flow.loss(pred_velocity, fb.target_velocity)

            decoded_joints = None
            target_centroids = None
            if mano_decoder is not None:
                with torch.no_grad():
                    target_centroids = graph["finger_centroid"]
                decoded = mano_decoder(mano_x1)
                decoded_joints = decoded["joints"]

            losses = loss_bundle(
                epoch=epoch - 1,
                flow_loss=flow_loss,
                rot6d=mano_x1[:, ROT6D_SLICE],
                joint_angles=mano_x1[:, POSE_SLICE],
                decoded_joints=decoded_joints,
                target_finger_centroids=target_centroids,
            )

            # Physics-aware energy loss (Grok #1): decode the PREDICTED x1
            # through MANO and penalize penetration + reward surface contact.
            # Only applied when t > 0.8 (near-clean samples) so the MANO
            # decoder receives reasonable inputs. Gradient flows back through
            # pred_velocity → velocity_net to teach it to avoid penetration.
            if args.physics_loss_weight > 0.0 and mano_decoder is not None and not args.hierarchical:
                # Only apply physics loss for samples near t=1 (clean end)
                # where the extrapolated x1 is reliable.
                t_mask = fb.t > 0.8  # (B,)
                if t_mask.any():
                    # Predicted x1 from flow: xt + (1-t) * velocity (WITH gradient)
                    pred_velocity_masked = pred_velocity[t_mask]
                    xt_masked = fb.xt[t_mask]
                    t_masked = fb.t[t_mask]

                    if latent_ae is not None:
                        pred_latent = xt_masked + (1.0 - t_masked.unsqueeze(-1)) * pred_velocity_masked
                        # AE decode: no learnable params, but differentiable
                        pred_mano_normed = latent_ae.decode(pred_latent)
                        pred_mano = pred_mano_normed * latent_std + latent_mean
                    else:
                        pred_mano = xt_masked + (1.0 - t_masked.unsqueeze(-1)) * pred_velocity_masked

                    # Decode through MANO — NO detach, gradient flows back
                    pred_decoded = mano_decoder(pred_mano)
                    pred_verts = pred_decoded["vertices"]  # (M, 778, 3)

                    # Penetration
                    obj_pts = batch["object_points"][t_mask]
                    obj_nor = batch["object_normals"][t_mask]
                    hv_sq = pred_verts.pow(2).sum(dim=-1, keepdim=True)
                    op_sq = obj_pts.pow(2).sum(dim=-1).unsqueeze(1)
                    cross = torch.bmm(pred_verts, obj_pts.transpose(-1, -2))
                    dist_sq = (hv_sq + op_sq - 2.0 * cross).clamp_min(0.0)
                    nearest_idx = dist_sq.argmin(dim=-1)
                    idx_exp = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
                    nearest_pts = obj_pts.gather(1, idx_exp)
                    nearest_nor = obj_nor.gather(1, idx_exp)
                    diff = pred_verts - nearest_pts
                    signed = (diff * nearest_nor).sum(dim=-1)
                    penetration = (-signed).clamp_min(0.0).mean()

                    # Surface distance: fingertips close to object
                    finger_tips = pred_decoded["joints"][:, [4, 8, 12, 16, 20], :]
                    tip_dists = torch.cdist(finger_tips, obj_pts).min(dim=-1).values
                    surface_dist = tip_dists.mean()

                    physics_loss = args.physics_loss_weight * (penetration + 0.5 * surface_dist)
                    losses["total"] = losses["total"] + physics_loss

            # Naturalness commitment loss: pull predicted latent toward nearest
            # codebook center. Only applies to latent flow mode.
            if args.naturalness_codebook is not None and latent_ae is not None and not args.hierarchical:
                # The flow's velocity points toward x1. The predicted x1 in latent
                # space is: xt + (1-t)*v. We want this to be near a codebook center.
                pred_z1 = fb.xt + (1.0 - fb.t.unsqueeze(-1)) * pred_velocity  # (B, latent_dim)
                # Find nearest codebook center (no grad through codebook)
                dists = torch.cdist(pred_z1.unsqueeze(0), _nat_codebook.unsqueeze(0)).squeeze(0)  # (B, K)
                nearest_center = _nat_codebook[dists.argmin(dim=-1)]  # (B, latent_dim)
                nat_loss = args.naturalness_weight * (pred_z1 - nearest_center.detach()).pow(2).mean()
                losses["total"] = losses["total"] + nat_loss

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            bs = x1.shape[0]
            train_samples += bs
            for k in train_sums:
                train_sums[k] += float(losses[k].detach().item()) * bs

        train_metrics = {k: v / max(train_samples, 1) for k, v in train_sums.items()}

        # ---- Validation pass (just compute losses; no sampling for the smoke test) ----
        adapter.eval()
        velocity_net.eval()
        val_sums = {"flow": 0.0, "joint_limit": 0.0, "rotation_orth": 0.0, "contact_align": 0.0, "total": 0.0}
        val_samples = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = move_batch_to_device(raw_batch, device)
                # (sphere) unified intent swap (same as train loop).
                if args.use_unified_intent:
                    batch["palm_centroid"] = batch["unified_centroid"].clone()
                # (sphere) sphere-intent swap (same as train).
                if args.use_sphere_intent:
                    radius = batch["unified_spread"].norm(dim=-1, keepdim=True)
                    batch["palm_spread"] = radius.expand(-1, 3).contiguous()
                    batch["palm_entropy"] = batch["unified_entropy"].clone()
                # (sphere) approach-direction swap. At VAL time we
                # NEVER mask — always provide the GT direction (simulates the
                # deployment case where the user provides both position + direction).
                if args.use_approach_direction:
                    hTm_rot = batch["hTm_rot"]
                    fwd_local = torch.tensor([0.0, 0.0, 1.0], device=hTm_rot.device, dtype=hTm_rot.dtype)
                    batch["palm_normal"] = hTm_rot @ fwd_local
                if args.teacher_forcing:
                    graph, active_finger_prob = build_teacher_graph(batch)
                else:
                    head_out = contact_head(
                        object_points=batch["object_points"],
                        object_normals=batch["object_normals"],
                        stage1_contact_input=batch["stage1_contact_input"],
                        m2ae_global=batch["m2ae_global"],
                        m2ae_local=batch["m2ae_local"],
                        patch_centers=batch["patch_centers"],
                    )
                    graph = head_out["graph"]
                    active_finger_prob = head_out["active_finger_prob"]

                val_patch_contact_weight = None
                if args.part_aware_gating:
                    if args.palm_only_intent:
                        from graspauto.conditioning import compute_patch_weight_from_point  # noqa: PLC0415
                        val_patch_contact_weight = compute_patch_weight_from_point(
                            patch_centers=batch["patch_centers"],
                            target_point=batch["palm_centroid"],
                        )
                    else:
                        from graspauto.conditioning import compute_patch_contact_weight  # noqa: PLC0415
                        val_patch_contact_weight = compute_patch_contact_weight(
                            patch_centers=batch["patch_centers"],
                            object_points=batch["object_points"],
                            contact_mask=batch["unified_contact_target"] if args.teacher_forcing else batch["stage1_contact_input"],
                        )
                    if args.topk_gate > 0 and val_patch_contact_weight is not None:
                        _vals, _idx = val_patch_contact_weight.topk(args.topk_gate, dim=-1)
                        _mask = torch.zeros_like(val_patch_contact_weight)
                        _mask.scatter_(-1, _idx, 1.0)
                        val_patch_contact_weight = _mask
                val_intent_ids = batch["intent_id"] if args.intent_token else None
                val_adv_patch_centers = batch["patch_centers"] if args.advanced_gating else None
                val_adv_palm_target = batch["palm_centroid"] if args.advanced_gating else None

                # (sphere) val: ALWAYS pass direction (no mask at val time).
                val_intent_direction = None
                if args.use_intent_direction:
                    _anchor = batch["hTm_trans"]
                    _tap = batch["unified_centroid"]
                    _dir = _tap - _anchor
                    _dir = _dir / _dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    val_intent_direction = _dir

                bundle = adapter(
                    m2ae_local=batch["m2ae_local"],
                    graph=graph,
                    active_finger_prob=active_finger_prob,
                    patch_contact_weight=val_patch_contact_weight,
                    intent_ids=val_intent_ids,
                    patch_centers=val_adv_patch_centers,
                    palm_target=val_adv_palm_target,
                    intent_direction=val_intent_direction,
                )
                mano_x1 = build_target_mano_params(batch)
                if args.hierarchical:
                    wrist_x1 = mano_x1[:, :WRIST_DIM]
                    finger_x1 = mano_x1[:, WRIST_DIM:]
                    wrist_x0 = torch.randn_like(wrist_x1)
                    wrist_fb = flow.prepare_batch(wrist_x0, wrist_x1)
                    wrist_pred_v = velocity_net(wrist_fb.xt, wrist_fb.t, condition=bundle)
                    wrist_loss = flow.loss(wrist_pred_v, wrist_fb.target_velocity)

                    # Use generated wrist for Stage 2 (same as training)
                    wrist_sampled = flow.sample(
                        velocity_net, torch.randn_like(wrist_x1),
                        condition=bundle, num_steps=10, method="euler",
                    )
                    wrist_token = wrist_proj(wrist_sampled).unsqueeze(1)
                    stage2_tokens = torch.cat([bundle.tokens, wrist_token], dim=1)
                    from graspauto.conditioning import ConditioningBundle as _CB  # noqa: PLC0415
                    stage2_bundle = _CB(
                        tokens=stage2_tokens,
                        object_tokens=bundle.object_tokens,
                        contact_tokens=bundle.contact_tokens,
                        token_types=torch.cat([bundle.token_types,
                            torch.full((1,), 5, dtype=torch.long, device=device)]),
                    )
                    finger_x0 = torch.randn_like(finger_x1)
                    finger_fb = flow.prepare_batch(finger_x0, finger_x1)
                    finger_pred_v = velocity_net_finger(finger_fb.xt, finger_fb.t, condition=stage2_bundle)
                    finger_loss = flow.loss(finger_pred_v, finger_fb.target_velocity)
                    flow_loss = wrist_loss + finger_loss
                elif latent_ae is not None:
                    normed = (mano_x1 - latent_mean) / latent_std
                    x1 = latent_ae.encode(normed)
                    x0 = torch.randn_like(x1)
                    fb = flow.prepare_batch(x0, x1)
                    pred_velocity = velocity_net(fb.xt, fb.t, condition=bundle)
                    flow_loss = flow.loss(pred_velocity, fb.target_velocity)
                else:
                    x1 = mano_x1
                    x0 = torch.randn_like(x1)
                    fb = flow.prepare_batch(x0, x1)
                    pred_velocity = velocity_net(fb.xt, fb.t, condition=bundle)
                    flow_loss = flow.loss(pred_velocity, fb.target_velocity)

                decoded_joints = None
                target_centroids = None
                if mano_decoder is not None:
                    target_centroids = graph["finger_centroid"]
                    decoded = mano_decoder(mano_x1)
                    decoded_joints = decoded["joints"]

                losses = loss_bundle(
                    epoch=epoch - 1,
                    flow_loss=flow_loss,
                    rot6d=mano_x1[:, ROT6D_SLICE],
                    joint_angles=mano_x1[:, POSE_SLICE],
                    decoded_joints=decoded_joints,
                    target_finger_centroids=target_centroids,
                )
                bs = x1.shape[0]
                val_samples += bs
                for k in val_sums:
                    val_sums[k] += float(losses[k].item()) * bs

        val_metrics = {k: v / max(val_samples, 1) for k, v in val_sums.items()}
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 3),
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics_log.append(record)
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_total={train_metrics['total']:.4f} "
            f"train_flow={train_metrics['flow']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_flow={val_metrics['flow']:.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        # Save checkpoint at every epoch (smoke test)
        ckpt = {
            "epoch": epoch,
            "adapter_state": adapter.state_dict(),
            "velocity_net_state": velocity_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if velocity_net_finger is not None:
            ckpt["velocity_net_finger_state"] = velocity_net_finger.state_dict()
            ckpt["wrist_proj_state"] = wrist_proj.state_dict()
        torch.save(ckpt, out_dir / "last.pt")
        if epoch == 1 or val_metrics["total"] < min((m["val"]["total"] for m in metrics_log[:-1]), default=float("inf")):
            torch.save(ckpt, out_dir / "best.pt")

    # Save metrics log as JSON
    write_json(out_dir / "metrics.json", {"tag": args.tag, "config": vars(args), "history": metrics_log})
    print(f"\n[done] artifacts in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
