#!/usr/bin/env python3
"""
Unified Contact Module — single source of truth for contact modeling.

Design principles:
  1. Contact is evaluated on FINGER-PAD vertices only (not all 778).
  2. Grasp-type-aware:
     - pinch requires thumb+index pads.
     - power requires all 5 finger pads.
     - palm contact is a BONUS signal (use for scoring/metrics), not a requirement.
  3. SDF-based signed distance distinguishes penetration / contact / free.
  4. Same definitions used in training loss, BoN scoring, TTO, and evaluation.

SDF convention:
  d < 0      → penetrating (inside object)
  0 ≤ d < τ  → contact band (touching surface)
  d ≥ τ      → free (not touching)

Grasp types (user-specified):
  - "pinch": thumb_pad ∪ index_pad
  - "power": all five finger pads (palm is bonus-only)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Literal
import numpy as np

GraspType = Literal["pinch", "power"]

# Finger names (canonical order, matches pad mask keys)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Expected contact fingers per grasp type
REQUIRED_FINGERS: Dict[str, tuple[str, ...]] = {
    "pinch": ("thumb", "index"),
    "power": ("thumb", "index", "middle", "ring", "pinky"),
}

# Default thresholds (meters)
CONTACT_BAND_TAU = 0.005       # 5 mm — hand vert within this of surface = "contact"
FINGER_SOFT_ALPHA = 0.0005     # 0.5 mm — sigmoid temperature for soft finger contact
FINGER_S_MIN = 0.05            # min soft score to count a finger as "contacted" (anti-edge-graze)
PAD_CAPSULE_RADIUS = 0.012     # 12 mm — radius around PIP→DIP→TIP capsule for finger pads
PALM_RADIUS = 0.035            # 35 mm — radius from MCP centroid for palm region


# ============================================================
# 0. SDF Computation — signed distance from hand verts to object mesh
# ============================================================

def compute_sdf_trimesh(
    hand_verts: torch.Tensor,
    obj_mesh,
) -> torch.Tensor:
    """Compute signed distance from hand vertices to object mesh.

    Uses trimesh for watertight meshes: nearest surface point + winding-number
    inside/outside test.

    Args:
        hand_verts: (B, 778, 3) or (778, 3) hand vertex positions.
        obj_mesh: trimesh.Trimesh object (must be watertight or will be filled).

    Returns:
        sdf: same shape as hand_verts[..., 0] — (B, 778) or (778,).
             Negative = inside (penetrating), positive = outside.
    """
    squeeze = hand_verts.dim() == 2
    if squeeze:
        hand_verts = hand_verts.unsqueeze(0)

    device = hand_verts.device
    B = hand_verts.shape[0]

    # Ensure watertight
    if not obj_mesh.is_watertight:
        obj_mesh.fill_holes()

    sdf_list = []
    for i in range(B):
        pts = hand_verts[i].detach().cpu().numpy()  # (778, 3)
        # Unsigned distance to surface
        closest, dist, _ = obj_mesh.nearest.on_surface(pts)
        # Inside/outside via winding number
        inside = obj_mesh.contains(pts)  # (778,) bool
        signed = dist.copy()
        signed[inside] *= -1.0  # negative inside
        sdf_list.append(torch.from_numpy(signed.astype(np.float32)))

    sdf = torch.stack(sdf_list, dim=0).to(device)  # (B, 778)
    if squeeze:
        sdf = sdf.squeeze(0)
    return sdf


def compute_sdf_grid(
    hand_verts: torch.Tensor,
    sdf_grid: torch.Tensor,
    grid_center: torch.Tensor,
    grid_scale: torch.Tensor,
    grid_size: int = 128,
) -> torch.Tensor:
    """Look up SDF from a pre-computed voxel grid (fast, differentiable).

    Args:
        hand_verts: (B, 778, 3) hand vertex positions in world coords.
        sdf_grid: (1, 1, D, H, W) pre-computed SDF volume.
        grid_center: (3,) center of the SDF grid in world coords.
        grid_scale: scalar or (3,) — maps world coords to [-1, 1] grid coords.
        grid_size: grid resolution (D=H=W).

    Returns:
        sdf: (B, 778) signed distances.
    """
    B = hand_verts.shape[0]
    # Transform to grid coords [-1, 1]
    pts_norm = (hand_verts - grid_center.to(hand_verts.device)) * grid_scale.to(hand_verts.device)
    # grid_sample expects (B, N, 1, 1, 3) for 3D, returns (B, 1, N, 1, 1)
    pts_grid = pts_norm.view(B, -1, 1, 1, 3)
    # Note: grid stored as (D,H,W) but grid_sample expects (x,y,z) → need to flip
    pts_grid = pts_grid.flip(-1)
    sdf_vals = F.grid_sample(
        sdf_grid.expand(B, -1, -1, -1, -1),
        pts_grid,
        align_corners=True,
        mode='bilinear',
        padding_mode='border',
    )
    return sdf_vals.view(B, -1)  # (B, 778)


def compute_sdf_nn_approx(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    obj_normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Approximate SDF using NN distance + surface normal direction test.

    When object mesh is unavailable, this gives a pseudo-signed distance:
    - Find nearest object point for each hand vertex
    - If the vector (hand_vert - obj_point) · obj_normal > 0 → outside (positive)
    - If < 0 → inside (negative, penetrating)

    Without normals: falls back to unsigned distance (always positive).

    Args:
        hand_verts: (B, 778, 3)
        obj_pc: (B, N, 3) or (N, 3) object point cloud
        obj_normals: (B, N, 3) or (N, 3) outward-facing surface normals, optional.

    Returns:
        sdf: (B, 778) approximate signed distances.
    """
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
    if obj_pc.dim() == 2:
        obj_pc = obj_pc.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)

    # NN distance
    dists = torch.cdist(hand_verts, obj_pc)  # (B, 778, N)
    nn_dist, nn_idx = dists.min(dim=2)  # (B, 778)

    if obj_normals is None:
        return nn_dist  # unsigned fallback

    if obj_normals.dim() == 2:
        obj_normals = obj_normals.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)

    # Gather nearest object normals
    B, V = nn_idx.shape
    nn_normals = torch.gather(obj_normals, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, 778, 3)
    nn_points = torch.gather(obj_pc, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, 778, 3)

    # Direction test: (hand_vert - nearest_obj_point) · obj_normal
    direction = hand_verts - nn_points  # (B, 778, 3)
    dot = (direction * nn_normals).sum(dim=-1)  # (B, 778)

    # Sign: positive outside, negative inside
    sign = torch.sign(dot)
    sign[sign == 0] = 1.0  # on surface → outside

    return nn_dist * sign


# ============================================================
# 1. Vertex Masks
# ============================================================

def build_contact_masks(
    mano_layer,
    pad_radius: float = PAD_CAPSULE_RADIUS,
    palm_radius: float = PALM_RADIUS,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build boolean vertex masks (778,) for contact-relevant regions.

    Uses MANO rest pose (betas=0, pose=0) to identify pad vertices.
    Vertex indices are topology-stable across poses, so masks are reusable.

    Method:
      - Finger pads: capsule along PIP→DIP→TIP, palmar side only.
      - Palm: sphere at MCP centroid, palmar side only.
      - "Palmar side" = vertex normal dot palm_normal > 0.

    Returns dict with keys:
      - thumb_pad, index_pad, middle_pad, ring_pad, pinky_pad  (per-finger)
      - palm
      - pinch      = thumb_pad ∪ index_pad
      - power      = all five finger pads (REQUIRED contact set)
      - power_all  = all five finger pads ∪ palm (for bonus metrics/scoring)
    """
    if device is None:
        device = _infer_device(mano_layer)

    with torch.no_grad():
        pose = torch.zeros(1, 48, device=device)
        betas = torch.zeros(1, 10, device=device)
        out = mano_layer(pose, betas)
        joints = out.joints[0]   # (21, 3)
        verts = out.verts[0]     # (778, 3)

    faces = mano_layer.th_faces.to(device)  # (F, 3)

    # --- Compute palmar side mask via vertex normals ---
    palmar = _palmar_mask(verts, joints, faces)

    # --- Finger pads: capsule PIP→DIP→TIP, palmar only ---
    finger_joints = {
        "thumb":  (2, 3, 4),   # PIP, DIP, TIP
        "index":  (6, 7, 8),
        "middle": (10, 11, 12),
        "ring":   (14, 15, 16),
        "pinky":  (18, 19, 20),
    }

    masks = {}
    for name, (pip_id, dip_id, tip_id) in finger_joints.items():
        d1 = _capsule_dist(verts, joints[pip_id], joints[dip_id])
        d2 = _capsule_dist(verts, joints[dip_id], joints[tip_id])
        min_d = torch.minimum(d1, d2)
        masks[f"{name}_pad"] = (min_d < pad_radius) & palmar

    # --- Palm: MCP centroid, palmar only ---
    palm_center = joints[[0, 5, 9, 13, 17]].mean(dim=0)
    masks["palm"] = ((verts - palm_center).norm(dim=1) < palm_radius) & palmar

    masks["pinch"] = masks["thumb_pad"] | masks["index_pad"]

    # Power grasp: REQUIRE the 5 finger pads. Palm contact is bonus-only.
    masks["power"] = (
        masks["thumb_pad"] | masks["index_pad"] | masks["middle_pad"]
        | masks["ring_pad"] | masks["pinky_pad"]
    )
    masks["power_all"] = masks["power"] | masks["palm"]

    return masks


def get_active_mask(
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
) -> torch.Tensor:
    """Return the boolean vertex mask (778,) for a given grasp type."""
    return masks[grasp_type]


# ============================================================
# 2. Metrics (non-differentiable, for evaluation / reporting)
# ============================================================

def contact_metrics(
    sdf_values: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    tau: float = CONTACT_BAND_TAU,
) -> Dict[str, float]:
    """Compute contact metrics on the REQUIRED active vertices only.

    Note: for "power", the required set is the 5 finger pads (no palm).
    Palm is reported separately as a bonus signal (if available).

    Args:
        sdf_values: (B, 778) SDF values for all hand vertices.
        masks: from build_contact_masks().
        grasp_type: "pinch" or "power".
        tau: contact band threshold in meters.

    Returns dict with per-batch-mean values:
        c_band:   fraction of active verts in contact band [0, τ)
        p_rate:   fraction of active verts penetrating (sdf < 0)
        p_depth:  mean |sdf| of penetrating active verts (meters)
        free:     fraction of active verts with sdf ≥ τ
    """
    if sdf_values.dim() == 1:
        sdf_values = sdf_values.unsqueeze(0)
    B = sdf_values.shape[0]
    device = sdf_values.device

    active = get_active_mask(masks, grasp_type).to(device)  # (778,)
    n_active = active.sum().item()
    if n_active == 0:
        return {"c_band": 0.0, "p_rate": 0.0, "p_depth": 0.0, "free": 1.0}

    # Extract active vertex SDF: (B, n_active)
    sdf_active = sdf_values[:, active]

    # Per-sample metrics, then average over batch
    in_band = ((sdf_active >= 0) & (sdf_active < tau)).float().mean(dim=1)    # (B,)
    penetrating = (sdf_active < 0).float()
    p_rate = penetrating.mean(dim=1)                                           # (B,)

    # Mean penetration depth (only over penetrating verts)
    pen_depths = (-sdf_active).clamp(min=0)  # (B, n_active), positive where penetrating
    pen_sum = (pen_depths * penetrating).sum(dim=1)                            # (B,)
    pen_count = penetrating.sum(dim=1).clamp(min=1)                            # (B,)
    p_depth = pen_sum / pen_count                                              # (B,)

    free = (sdf_active >= tau).float().mean(dim=1)                             # (B,)

    out = {
        "c_band":  in_band.mean().item(),
        "p_rate":  p_rate.mean().item(),
        "p_depth": p_depth.mean().item(),
        "free":    free.mean().item(),
    }

    # Bonus-only palm metric (useful for power grasps).
    if "palm" in masks:
        palm = masks["palm"].to(device)
        if palm.any():
            sdf_palm = sdf_values[:, palm]
            out["palm_c_band"] = ((sdf_palm >= 0) & (sdf_palm < tau)).float().mean().item()

    return out


def contact_metrics_nn(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    tau: float = CONTACT_BAND_TAU,
) -> Dict[str, float]:
    """Approximate contact metrics using unsigned NN distance (no SDF needed).

    Useful when SDF grids are unavailable (e.g., OakInk objects).
    Cannot distinguish penetration from contact — reports them together as
    "c_near" (within τ of surface, either side).

    Args:
        hand_verts: (B, 778, 3)
        obj_pc: (B, N_obj, 3) or (N_obj, 3)
        masks: from build_contact_masks().
        grasp_type: "pinch" or "power".
        tau: distance threshold in meters.

    Returns dict:
        c_near:    fraction of active verts within τ of any object point (unsigned)
        mean_dist: mean NN distance of active verts to object (meters)
    """
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
    if obj_pc.dim() == 2:
        obj_pc = obj_pc.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)

    B = hand_verts.shape[0]
    device = hand_verts.device
    active = get_active_mask(masks, grasp_type).to(device)

    # Active verts only: (B, n_active, 3)
    active_verts = hand_verts[:, active]

    # NN distance
    dists = torch.cdist(active_verts, obj_pc)  # (B, n_active, N_obj)
    nn_dist = dists.min(dim=2).values           # (B, n_active)

    c_near = (nn_dist < tau).float().mean(dim=1).mean().item()
    mean_dist = nn_dist.mean().item()

    return {"c_near": c_near, "mean_dist": mean_dist}


# ============================================================
# 3. Differentiable Losses (for training & TTO)
# ============================================================

def loss_penetration(
    sdf_values: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    all_verts: bool = True,
) -> torch.Tensor:
    """Penetration loss: push penetrating vertices out of the object.

    L_pen = mean(ReLU(-sdf)²)

    Args:
        sdf_values: (B, 778)
        masks: from build_contact_masks().
        grasp_type: "pinch" or "power".
        all_verts: if True, penalize ALL 778 vertices for penetration
                   (penetration is bad everywhere, not just on pads).
                   if False, only penalize active pad vertices.
    """
    if all_verts:
        # Penetration anywhere on the hand is bad
        return (F.relu(-sdf_values) ** 2).mean()
    else:
        active = get_active_mask(masks, grasp_type).to(sdf_values.device)
        sdf_active = sdf_values[:, active]
        return (F.relu(-sdf_active) ** 2).mean()


# (DELETED: loss_attraction, loss_attraction_nn, loss_coverage — replaced by finger-level API)


# ============================================================
# 3b. Finger-Level Losses & Metrics (NEW — replaces vertex-ratio)
# ============================================================

def finger_scores_sdf(
    sdf_values: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    tau: float = CONTACT_BAND_TAU,
    alpha: float = FINGER_SOFT_ALPHA,
) -> Dict[str, torch.Tensor]:
    """Compute per-finger soft contact score s_f ∈ [0, 1] using SDF.

    s_f = mean over finger-pad verts of σ((τ - sdf) / α)

    When sdf is in [0, τ] (contact band), σ ≈ 1.
    When sdf >> τ (far away), σ ≈ 0.
    When sdf < 0 (penetrating), σ > 1 (clamped by sigmoid saturation).

    Args:
        sdf_values: (B, 778) signed distances for all hand vertices.
        masks: from build_contact_masks().
        tau: contact band width (m).
        alpha: sigmoid temperature (m). Smaller = sharper.

    Returns dict: {finger_name: (B,) tensor} for all 5 fingers + "palm".
    """
    if sdf_values.dim() == 1:
        sdf_values = sdf_values.unsqueeze(0)
    device = sdf_values.device
    scores = {}
    for name in FINGER_NAMES:
        mask = masks[f"{name}_pad"].to(device)
        if mask.any():
            sdf_f = sdf_values[:, mask]  # (B, n_f)
            scores[name] = torch.sigmoid((tau - sdf_f) / alpha).mean(dim=1)  # (B,)
        else:
            scores[name] = torch.zeros(sdf_values.shape[0], device=device)
    # Palm (bonus only)
    if "palm" in masks:
        mask = masks["palm"].to(device)
        if mask.any():
            sdf_p = sdf_values[:, mask]
            scores["palm"] = torch.sigmoid((tau - sdf_p) / alpha).mean(dim=1)
        else:
            scores["palm"] = torch.zeros(sdf_values.shape[0], device=device)
    return scores


def finger_scores_nn(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    tau: float = CONTACT_BAND_TAU,
    alpha: float = FINGER_SOFT_ALPHA,
) -> Dict[str, torch.Tensor]:
    """Compute per-finger soft contact score s_f using NN distance (no SDF).

    Unsigned distance — cannot distinguish penetration from contact.
    s_f = mean over finger-pad verts of σ((τ - nn_dist) / α)

    Args:
        hand_verts: (B, 778, 3)
        obj_pc: (B, N, 3) or (N, 3)
        masks: from build_contact_masks().
        tau: contact band (m).
        alpha: sigmoid temperature (m).

    Returns dict: {finger_name: (B,) tensor} for all 5 fingers + "palm".
    """
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
    if obj_pc.dim() == 2:
        obj_pc = obj_pc.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)
    device = hand_verts.device

    # Pre-compute full NN distances: (B, 778)
    nn_dist = torch.cdist(hand_verts, obj_pc).min(dim=2).values

    scores = {}
    for name in FINGER_NAMES:
        mask = masks[f"{name}_pad"].to(device)
        if mask.any():
            d_f = nn_dist[:, mask]  # (B, n_f)
            scores[name] = torch.sigmoid((tau - d_f) / alpha).mean(dim=1)
        else:
            scores[name] = torch.zeros(hand_verts.shape[0], device=device)
    if "palm" in masks:
        mask = masks["palm"].to(device)
        if mask.any():
            d_p = nn_dist[:, mask]
            scores["palm"] = torch.sigmoid((tau - d_p) / alpha).mean(dim=1)
        else:
            scores["palm"] = torch.zeros(hand_verts.shape[0], device=device)
    return scores


def loss_finger_contact(
    finger_scores: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    s_min: float = FINGER_S_MIN,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Finger-level contact loss: make required fingers contact the object.

    Two terms:
      L_req  = mean over required fingers of (1 - s_f)
               → "required fingers should touch"
      L_cov  = mean over required fingers of ReLU(s_min - s_f)²
               → "prevent single-vertex edge-graze from counting"

    Total = L_req + L_cov

    Args:
        finger_scores: from finger_scores_sdf() or finger_scores_nn().
        grasp_type: "pinch" or "power".
        s_min: minimum soft score threshold for coverage floor.

    Returns:
        (loss, info_dict) where info_dict has per-finger scores and rates.
    """
    required = REQUIRED_FINGERS[grasp_type]
    device = finger_scores[required[0]].device

    # L_req: encourage required fingers to contact
    req_scores = torch.stack([finger_scores[f] for f in required], dim=1)  # (B, n_req)
    l_req = (1.0 - req_scores).mean()

    # L_cov: coverage floor — prevent edge-graze
    l_cov = F.relu(s_min - req_scores).pow(2).mean()

    loss = l_req + l_cov

    # Info for logging (non-differentiable)
    info = {}
    with torch.no_grad():
        for name in FINGER_NAMES:
            if name in finger_scores:
                info[f"s_{name}"] = finger_scores[name].mean().item()
        if "palm" in finger_scores:
            info["s_palm"] = finger_scores["palm"].mean().item()
        # Finger hit rate: fraction of required fingers with s_f > s_min
        hits = (req_scores > s_min).float().mean(dim=1)  # (B,)
        info["finger_hit_rate"] = hits.mean().item()
        info["l_req"] = l_req.item()
        info["l_cov"] = l_cov.item()

    return loss, info


def finger_contact_metrics(
    finger_scores: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    s_thresh: float = FINGER_S_MIN,
) -> Dict[str, float]:
    """Non-differentiable finger-level metrics for evaluation / reporting.

    Returns:
        finger_hit_rate: fraction of required fingers contacted (main metric).
        per-finger s_f values and hit booleans.
        palm_bonus: palm s_f (not counted in hit rate).
    """
    required = REQUIRED_FINGERS[grasp_type]
    with torch.no_grad():
        req_scores = torch.stack([finger_scores[f] for f in required], dim=1)  # (B, n_req)
        hits = (req_scores > s_thresh).float()

        out = {
            "finger_hit_rate": hits.mean().item(),
            "finger_hit_count": hits.sum(dim=1).mean().item(),
            "n_required": len(required),
        }
        for name in FINGER_NAMES:
            if name in finger_scores:
                s = finger_scores[name]
                out[f"s_{name}"] = s.mean().item()
                out[f"hit_{name}"] = (s > s_thresh).float().mean().item()
        if "palm" in finger_scores:
            out["s_palm"] = finger_scores["palm"].mean().item()
    return out


# ============================================================
# 4. BoN Scoring (for Best-of-N selection)
# ============================================================

def bon_score_sdf(
    sdf_values: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    tau: float = CONTACT_BAND_TAU,
    w_pen: float = 5.0,
    w_palm: float = 0.0,
) -> torch.Tensor:
    """Score grasps for Best-of-N selection using SDF.

    Higher = better.

    Base score (required contact set only):
      score = c_band - w_pen * p_rate

    Optional bonus (power only, bonus-only palm contact):
      + w_palm * palm_c_band

    Args:
        sdf_values: (N, 778) SDF values for N candidate grasps.
        Returns: (N,) scores.
    """
    active = get_active_mask(masks, grasp_type).to(sdf_values.device)
    sdf_active = sdf_values[:, active]  # (N, n_active)

    c_band = ((sdf_active >= 0) & (sdf_active < tau)).float().mean(dim=1)
    p_rate = (sdf_active < 0).float().mean(dim=1)

    score = c_band - w_pen * p_rate

    # Bonus-only palm contact (if requested and mask is available)
    if w_palm != 0.0 and "palm" in masks:
        palm = masks["palm"].to(sdf_values.device)
        if palm.any():
            sdf_palm = sdf_values[:, palm]
            palm_c_band = ((sdf_palm >= 0) & (sdf_palm < tau)).float().mean(dim=1)
            score = score + w_palm * palm_c_band

    return score


def bon_score_nn(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    grasp_type: GraspType,
    tau: float = CONTACT_BAND_TAU,
) -> torch.Tensor:
    """Score grasps for Best-of-N selection using NN distance (no SDF).

    Higher = better.
    score = fraction of active verts within τ of object surface.

    Note: cannot penalize penetration without SDF.
    """
    if obj_pc.dim() == 2:
        obj_pc = obj_pc.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)

    active = get_active_mask(masks, grasp_type).to(hand_verts.device)
    active_verts = hand_verts[:, active]

    dists = torch.cdist(active_verts, obj_pc)
    nn_dist = dists.min(dim=2).values  # (N, n_active)

    return (nn_dist < tau).float().mean(dim=1)


# ============================================================
# 5. Composite Loss (convenience wrapper)
# ============================================================

# (DELETED: contact_loss_sdf, contact_loss_nn — replaced by finger-level API)


# ============================================================
# 6. Grasp Type Classification (auto-labeling from data)
# ============================================================

def classify_grasp_type(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    tau: float = CONTACT_BAND_TAU,
    active_thresh: float = 0.1,
) -> list[GraspType]:
    """Classify grasp type from hand-object contact pattern.

    For each sample, computes per-region contact ratio (fraction of pad verts
    within τ of object). Then applies simple rules:
      - pinch: thumb + index active, ≤ 2 other fingers active
      - power: everything else (3+ fingers or palm active)

    Args:
        hand_verts: (B, 778, 3)
        obj_pc: (B, N_obj, 3) or (N_obj, 3)
        masks: from build_contact_masks().
        tau: contact distance threshold.
        active_thresh: min contact ratio for a region to count as "active".

    Returns: list of B grasp type strings.
    """
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
    if obj_pc.dim() == 2:
        obj_pc = obj_pc.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)

    B = hand_verts.shape[0]
    device = hand_verts.device

    finger_regions = ["thumb_pad", "index_pad", "middle_pad", "ring_pad", "pinky_pad"]

    # Compute per-region contact ratio: (B, 6) — 5 fingers + palm
    region_contact = []
    for region in finger_regions + ["palm"]:
        active = masks[region].to(device)
        av = hand_verts[:, active]  # (B, n_r, 3)
        dists = torch.cdist(av, obj_pc)  # (B, n_r, N_obj)
        nn = dists.min(dim=2).values       # (B, n_r)
        cr = (nn < tau).float().mean(dim=1)  # (B,)
        region_contact.append(cr)

    rc = torch.stack(region_contact, dim=1)  # (B, 6)
    finger_active = rc[:, :5] > active_thresh  # (B, 5) bool
    palm_active = rc[:, 5] > active_thresh      # (B,) bool
    n_fingers = finger_active.float().sum(dim=1)  # (B,)

    thumb_on = finger_active[:, 0]
    index_on = finger_active[:, 1]
    others_on = finger_active[:, 2:].float().sum(dim=1)  # middle + ring + pinky

    # Pinch: thumb + index both active, at most 1 other finger, no palm
    is_pinch = thumb_on & index_on & (others_on <= 1) & ~palm_active

    labels = []
    for i in range(B):
        labels.append("pinch" if is_pinch[i].item() else "power")

    return labels


def classify_grasp_type_batch(
    hand_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    batch_size: int = 200,
    tau: float = CONTACT_BAND_TAU,
    active_thresh: float = 0.1,
) -> list[GraspType]:
    """Batch version of classify_grasp_type for large datasets."""
    N = hand_verts.shape[0]
    all_labels = []
    for i in range(0, N, batch_size):
        hv = hand_verts[i:i + batch_size]
        op = obj_pc[i:i + batch_size] if obj_pc.dim() == 3 else obj_pc
        labels = classify_grasp_type(hv, op, masks, tau=tau, active_thresh=active_thresh)
        all_labels.extend(labels)
    return all_labels


# ============================================================
# Internal helpers
# ============================================================

def _capsule_dist(pts: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Min distance from each point to line segment a→b."""
    seg = b - a
    seg_len = seg.norm()
    seg_dir = seg / (seg_len + 1e-10)
    proj = ((pts - a) @ seg_dir).clamp(0, seg_len.item())
    closest = a + proj.unsqueeze(1) * seg_dir.unsqueeze(0)
    return (pts - closest).norm(dim=1)


def _palmar_mask(verts: torch.Tensor, joints: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Identify palmar-side vertices using vertex normals and palm frame.

    Returns (778,) bool mask — True for vertices whose normal points toward
    the palm (i.e., the grasping surface, not the back of the hand).
    """
    # Vertex normals from face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0, dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-10)

    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), fn)
    vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-10)

    # Palm frame: forward = wrist→middle_MCP, right = index_MCP→pinky_MCP
    wrist, index_mcp, middle_mcp, pinky_mcp = joints[0], joints[5], joints[9], joints[17]
    pf = middle_mcp - wrist
    pf = pf / (pf.norm() + 1e-10)
    pr = pinky_mcp - index_mcp
    pr = pr / (pr.norm() + 1e-10)
    palm_normal = torch.linalg.cross(pf, pr)
    palm_normal = palm_normal / (palm_normal.norm() + 1e-10)

    return (vn @ palm_normal) > 0


def _infer_device(mano_layer) -> torch.device:
    """Best-effort device inference from a MANO layer."""
    try:
        return next(mano_layer.parameters()).device
    except StopIteration:
        pass
    for attr in (
        "th_faces", "faces_tensor", "faces",
        "th_J_regressor", "J_regressor",
        "th_v_template", "v_template",
    ):
        t = getattr(mano_layer, attr, None)
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")
