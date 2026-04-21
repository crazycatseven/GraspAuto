"""Dataset-agnostic palm 11-D token computed purely from object geometry + tap.

The original AutoGrasp palm token was derived from the stage-1 ContactPose
contact heatmap, which locked the semantic channel to CP-specific
statistics. This module computes the same 11-D layout (centroid 3, normal 3,
spread 3, entropy 1, mass 1) from (object_points, tap_xyz) alone — no
heatmap, no trained head — so it transfers across OakInk / GraspXL /
HOGraspNet / any future dataset AND is computable at Unity inference time
before the hand exists.

Design:

- centroid = tap point verbatim.
- local neighbourhood = Gaussian weighting of the object point cloud around
  the tap with sigma = 6 cm (matches the Gaussian gate sigma the adapter
  already uses, so the palm token and the locality gate see the same
  neighbourhood).
- normal = smallest-eigenvalue eigenvector of the weighted object-point
  covariance (= surface-normal direction of the locally fitted plane).
- spread = square-roots of the three covariance eigenvalues, sorted
  descending (approximate 3-D half-extents of the local region).
- entropy = normalised entropy of the Gaussian weights (0 = all mass at one
  point, 1 = mass uniformly spread).
- mass = sum of Gaussian weights divided by point count (a scale-free
  "how much object material is inside the gate" signal).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def compute_geom_palm_features(
    object_points: torch.Tensor,          # (N, 3) or (B, N, 3)
    tap_xyz: torch.Tensor,                 # (3,)    or (B, 3)
    sigma: float = 0.06,
) -> torch.Tensor:
    """Return the 11-D palm token (centroid + normal + spread + entropy + mass).

    Single-sample form:
        object_points (N, 3), tap_xyz (3,) -> (11,)
    Batched form:
        object_points (B, N, 3), tap_xyz (B, 3) -> (B, 11)

    The layout matches `ContactGraphConditioningAdapter`'s palm_proj input
    order: centroid (3) + normal (3) + spread (3) + entropy (1) + mass (1).
    """
    single = object_points.dim() == 2
    if single:
        object_points = object_points.unsqueeze(0)
        tap_xyz = tap_xyz.unsqueeze(0)

    B, N, _ = object_points.shape
    # Gaussian weights on object points
    d2 = ((object_points - tap_xyz.unsqueeze(1)) ** 2).sum(dim=-1)        # (B, N)
    w = torch.exp(-d2 / (2.0 * sigma * sigma))                             # (B, N)
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-10)                   # (B, 1)
    w_norm = w / w_sum                                                      # (B, N)

    # Local weighted mean
    mean_pt = (w_norm.unsqueeze(-1) * object_points).sum(dim=1)             # (B, 3)

    # Weighted covariance
    centered = object_points - mean_pt.unsqueeze(1)                         # (B, N, 3)
    wc = (w_norm.unsqueeze(-1) * centered)                                  # (B, N, 3)
    cov = torch.einsum("bni,bnj->bij", wc, centered)                        # (B, 3, 3)

    # Eigendecomposition (symmetric) — eigenvalues ascending
    eigvals, eigvecs = torch.linalg.eigh(cov)                               # (B, 3), (B, 3, 3)

    # Normal = smallest-eigenvalue eigenvector (plane-normal direction)
    normal = eigvecs[:, :, 0]                                               # (B, 3)

    # Spread = sqrt of eigenvalues, sorted DESCENDING, safe for zero eigvals
    spread_sorted, _ = torch.sort(eigvals.clamp(min=0.0), dim=-1, descending=True)
    spread = torch.sqrt(spread_sorted)                                       # (B, 3)

    # Normalised Shannon entropy of the weight distribution.
    # H = - sum w_norm log w_norm / log(N) in [0, 1]
    p = w_norm.clamp(min=1e-10)
    log_N = torch.tensor(math.log(max(N, 2)), device=w_norm.device, dtype=w_norm.dtype)
    entropy = -(p * torch.log(p)).sum(dim=-1) / log_N                        # (B,)

    # Mass = fraction of points with appreciable gate weight.
    # Scale-free surrogate for "how much material sits within the gate".
    # We use mean of w (unnormalised) which = effective support fraction.
    mass = w.mean(dim=-1)                                                    # (B,)

    # Assemble (centroid, normal, spread, entropy, mass)
    centroid = tap_xyz                                                       # (B, 3)
    token = torch.cat([
        centroid,
        normal,
        spread,
        entropy.unsqueeze(-1),
        mass.unsqueeze(-1),
    ], dim=-1)                                                                # (B, 11)

    if single:
        token = token.squeeze(0)
    return token


# MANO palm-anchor + fingertip vertex indices (shared across the codebase).
MANO_PALM_ANCHOR_IDS = [117, 95, 4, 218, 98, 55]   # 6 palm anchor verts
MANO_FINGERTIP_IDS   = [744, 320, 444, 555, 672]   # thumb, index, middle, ring, pinky


def grip_sphere_from_mano(
    hand_verts: torch.Tensor,       # (778, 3) or (B, 778, 3)
    radius_reduction: str = "mean",  # "mean" (default) or "min" (tighter grip)
) -> torch.Tensor:
    """Return the 7-D grip-sphere token (center 3 + radius 1 + approach 3).

    Deterministic, purely geometric, dataset-agnostic. Computed from a
    MANO hand mesh at training time; at VR runtime the same 7-D vector
    is supplied directly by the user's controller gesture.

    Layout:
        [center_x, center_y, center_z, radius, approach_x, approach_y, approach_z]

    Definitions:
        center   = mean of 6 MANO palm-anchor vertices (117, 95, 4, 218, 98, 55)
        radius   = mean distance from center to 5 MANO fingertip vertices
                   (744, 320, 444, 555, 672), i.e. average grip aperture
        approach = palm-plane normal (smallest-eigenvalue eigenvector of the
                   palm-anchor covariance), oriented so it points from the
                   palm center toward the mean fingertip — i.e. the "outward"
                   grasp direction. Unit-length.

    Shape:
        single: (778, 3) -> (7,)
        batched: (B, 778, 3) -> (B, 7)
    """
    single = hand_verts.dim() == 2
    if single:
        hand_verts = hand_verts.unsqueeze(0)

    palm_idx = torch.tensor(MANO_PALM_ANCHOR_IDS, device=hand_verts.device, dtype=torch.long)
    tip_idx  = torch.tensor(MANO_FINGERTIP_IDS,   device=hand_verts.device, dtype=torch.long)

    palm = hand_verts.index_select(1, palm_idx)      # (B, 6, 3)
    tips = hand_verts.index_select(1, tip_idx)       # (B, 5, 3)

    # Center = palm anchor mean
    center = palm.mean(dim=1)                         # (B, 3)

    # Radius — default "mean" (hand-enclosing) or "min" (tighter VR-gesture-like)
    tip_dists = (tips - center.unsqueeze(1)).norm(dim=-1)   # (B, 5)
    if radius_reduction == "min":
        radius = tip_dists.min(dim=1, keepdim=True).values  # (B, 1)
    else:
        radius = tip_dists.mean(dim=1, keepdim=True)        # (B, 1)

    # Approach direction = palm plane normal, oriented toward fingertips
    centered = palm - center.unsqueeze(1)             # (B, 6, 3)
    cov = torch.einsum("bni,bnj->bij", centered, centered)  # (B, 3, 3)
    eigvals, eigvecs = torch.linalg.eigh(cov)         # eigenvalues ascending
    normal = eigvecs[:, :, 0]                         # (B, 3), smallest-eigval dir

    # Orient: sign so normal points from center -> mean fingertip
    to_tips = tips.mean(dim=1) - center               # (B, 3)
    dot = (normal * to_tips).sum(dim=-1, keepdim=True)
    # If dot < 0, flip normal; if exactly 0 (degenerate), leave as-is
    sign = torch.where(dot < 0, torch.full_like(dot, -1.0), torch.full_like(dot, 1.0))
    normal = normal * sign

    # Re-normalise (eigenvectors are already unit, but safe under autograd)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    token = torch.cat([center, radius, normal], dim=-1)  # (B, 7)

    if single:
        token = token.squeeze(0)
    return token


def compute_geom_palm_split(
    object_points: torch.Tensor,
    tap_xyz: torch.Tensor,
    sigma: float = 0.06,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same computation as `compute_geom_palm_features` but returns the five
    named sub-components (centroid, normal, spread, entropy, mass) separately,
    matching the keys the existing dataset stores."""
    token = compute_geom_palm_features(object_points, tap_xyz, sigma)
    if token.dim() == 1:
        centroid = token[0:3]
        normal = token[3:6]
        spread = token[6:9]
        entropy = token[9:10].squeeze(0)
        mass = token[10:11].squeeze(0)
    else:
        centroid = token[:, 0:3]
        normal = token[:, 3:6]
        spread = token[:, 6:9]
        entropy = token[:, 9]
        mass = token[:, 10]
    return centroid, normal, spread, entropy, mass
