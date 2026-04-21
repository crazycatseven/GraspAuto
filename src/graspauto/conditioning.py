"""Conditioning adapter: graspauto contact graph features → cross-attention token sequence.

The graspauto velocity network needs to attend over a unified, fixed-size
token sequence representing "what we know about the object and the
intended grasp." This module collects the relevant features from the
existing graspauto contact graph head and packs them into that sequence.

Sources of conditioning information (all already produced by graspauto):

1. **Point-M2AE local patch tokens** — 64 patches × 384 dims, from
   `graspauto.point_m2ae_encoder.PointM2AEObjectEncoder.forward_local`.
   These give the spatial structure of the object surface.

2. **Per-finger graph features** — 5 fingers × (centroid 3 + normal 3 +
   spread 3 + entropy 1 + mass 1 + active_prob 1) = 12 dims per finger.
   From `graspauto.stage3_contact_graph` (`graph["finger_*"]` + `active_finger_prob`).
   These give the predicted finger-by-finger contact intent.

3. **Palm graph token** — 1 token × (centroid 3 + normal 3 + spread 3 +
   entropy 1 + mass 1) = 11 dims. From `graph["palm_*"]`.

4. **Unified contact graph token** — 1 token × (centroid 3 + normal 3 +
   spread 3 + entropy 1) = 10 dims. From `graph["unified_*"]`.

Total: 64 + 5 + 1 + 1 = **71 tokens**, each independently projected to
the velocity network's hidden dim (default 256). A learned token-type
embedding is added so the cross-attention can tell different token kinds
apart (object-patch vs finger vs palm vs unified).

Output: a `ConditioningBundle` with the unified token sequence
(`tokens: (B, 71, hidden_dim)`) plus the object-only and contact-only
sub-sequences for ablation purposes.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

NUM_FINGERS = 5
NUM_OBJECT_PATCHES = 64        # from point_m2ae_encoder
OBJECT_PATCH_DIM = 384          # from point_m2ae_encoder

# Per-finger feature dim: centroid (3) + normal (3) + spread (3) + entropy (1) + mass (1) + active_prob (1) = 12
FINGER_FEATURE_DIM = 3 + 3 + 3 + 1 + 1 + 1
# Palm feature dim: centroid (3) + normal (3) + spread (3) + entropy (1) + mass (1) = 11
PALM_FEATURE_DIM = 3 + 3 + 3 + 1 + 1
# Unified feature dim: centroid (3) + normal (3) + spread (3) + entropy (1) = 10
UNIFIED_FEATURE_DIM = 3 + 3 + 3 + 1

NUM_OBJECT_TOKENS = NUM_OBJECT_PATCHES
NUM_FINGER_TOKENS = NUM_FINGERS
NUM_PALM_TOKENS = 1
NUM_UNIFIED_TOKENS = 1
NUM_INTENT_TOKENS = 1
NUM_DIRECTION_TOKENS = 1  # watermelon r003+
# 64 + 5 + 1 + 1 + 1 = 72 tokens when intent is enabled, 71 otherwise.
TOTAL_TOKENS_WITH_INTENT = NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS + NUM_UNIFIED_TOKENS + NUM_INTENT_TOKENS
TOTAL_TOKENS = NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS + NUM_UNIFIED_TOKENS  # = 71

# Token-type IDs for the type embedding lookup.
TOKEN_TYPE_OBJECT = 0
TOKEN_TYPE_FINGER = 1
TOKEN_TYPE_PALM = 2
TOKEN_TYPE_UNIFIED = 3
TOKEN_TYPE_INTENT = 4
TOKEN_TYPE_DIRECTION = 5  # watermelon r003+
NUM_TOKEN_TYPES = 6

# Intent categories (ContactPose: use vs handoff)
INTENT_USE = 0
INTENT_HANDOFF = 1
NUM_INTENT_CATEGORIES = 2


def compute_patch_weight_from_point(
    patch_centers: Tensor,           # (B, P, 3)
    target_point: Tensor,             # (B, 3) — the user's palm target point
    sigma_m: float = 0.06,            # gaussian radius in meters (≈ 6 cm)
) -> Tensor:
    """Per-patch weight as a Gaussian kernel centered at a single target 3D point.

    Used for palm-only intent mode: the user provides a single 3D point (e.g.,
    by tapping on the object in VR) and the model attends to object patches
    within a neighborhood of that point.

    Args:
        patch_centers: (B, P, 3) patch center positions.
        target_point:  (B, 3) palm target in the same frame.
        sigma_m:       Gaussian std in meters. ~6 cm is roughly the size of a
                       hand palm and gives a soft locality.

    Returns:
        (B, P) weights in [0, 1] — exp(-|patch - target|^2 / (2 sigma^2)).
    """
    B, P, _ = patch_centers.shape
    tgt = target_point.unsqueeze(1)  # (B, 1, 3)
    dist_sq = (patch_centers - tgt).pow(2).sum(dim=-1)  # (B, P)
    return torch.exp(-dist_sq / (2.0 * sigma_m * sigma_m))


def compute_multiscale_attention_bias(
    patch_centers: Tensor,           # (B, P, 3)
    target_point: Tensor,             # (B, 3)
    sigmas_m: tuple[float, ...] = (0.03, 0.06, 0.12),
    bias_floor: float = -2.0,
    bias_scale: float = 0.5,
) -> Tensor:
    """Soft multi-scale log-domain attention bias for distance-to-palm-target.

    Returns a (B, P) tensor suitable to be ADDED to cross-attention logits
    before softmax. Per patch:

        raw = logsumexp_i(-dist²(p, target) / (2 σ_i²)) - log|S|
        bias = clip(bias_scale * raw, min=bias_floor, max=0)

    Design notes (learned from r023 NEGATIVE result):
    - Without clipping, distant patches see bias of -5..-6 which *hard*
      masks them in softmax, stripping global context (symmetry, scale).
      Clipping to bias_floor=-2 keeps ~13% relative attention on a
      far-away patch vs a co-located one, preserving global signal.
    - bias_scale=0.5 softens the pull toward the target so cross-attention
      can still choose to look broadly when useful.
    """
    B, P, _ = patch_centers.shape
    tgt = target_point.unsqueeze(1)
    dist_sq = (patch_centers - tgt).pow(2).sum(dim=-1)   # (B, P)
    logws = []
    for sigma in sigmas_m:
        logws.append(-dist_sq / (2.0 * sigma * sigma))
    stacked = torch.stack(logws, dim=0)
    lse = torch.logsumexp(stacked, dim=0) - float(torch.log(torch.tensor(len(sigmas_m), dtype=torch.float32)))
    return (bias_scale * lse).clamp(min=bias_floor, max=0.0)


def sinusoidal_offset_encoding(offsets: Tensor, num_freqs: int = 8, max_scale_m: float = 0.30) -> Tensor:
    """Per-axis sinusoidal encoding of a 3-D offset.

    Args:
        offsets:      (B, P, 3) displacement vectors (meters), e.g. patch_center - palm_target.
        num_freqs:    Number of log-spaced frequencies per axis.
        max_scale_m:  Largest scale (slowest frequency) in meters. Defaults to
                      30 cm ≈ size of a small object so wrap-around doesn't bite.

    Returns:
        (B, P, 3 * 2 * num_freqs) tensor with sin/cos channels.
    """
    B, P, _ = offsets.shape
    device = offsets.device
    dtype = offsets.dtype
    # Frequencies from slow (1/max_scale) to fast (num_freqs doubles)
    base_freq = 2.0 * 3.141592653589793 / max_scale_m
    freqs = base_freq * (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype))  # (F,)
    # offsets: (B, P, 3); freqs: (F,). Broadcast to (B, P, 3, F)
    scaled = offsets.unsqueeze(-1) * freqs.view(1, 1, 1, num_freqs)
    sin = torch.sin(scaled)
    cos = torch.cos(scaled)
    # Concat on the last axis: 3 axes × 2 (sin/cos) × F
    enc = torch.cat([sin, cos], dim=-1).reshape(B, P, 3 * 2 * num_freqs)
    return enc


def compute_patch_contact_weight(
    patch_centers: Tensor,           # (B, P, 3)
    object_points: Tensor,            # (B, N, 3)
    contact_mask: Tensor,             # (B, N), 0..1
    k_nearest: int = 47,
) -> Tensor:
    """Per-patch contact weight: for each patch, find the `k_nearest` nearest
    object points and take the mean of their contact mask values.

    This is the part-aware spatial gating signal. A patch gets a weight of
    ~1 if it is in the middle of a contact region, ~0 if it is far from any
    contact, and intermediate if it straddles the boundary.

    Args:
        patch_centers: (B, P, 3) patch center positions (Point-M2AE's final-
            level patch centers).
        object_points: (B, N, 3) full object point cloud in the same frame.
        contact_mask:  (B, N) contact labels or probabilities, in [0, 1].
        k_nearest:     Number of nearest object points per patch. Default 47
            ≈ 3000/64, so each of the 64 patches gets roughly its "fair share"
            of points. Smaller k gives tighter locality but more noise.

    Returns:
        (B, P) per-patch contact weight tensor.
    """
    B, P, _ = patch_centers.shape
    N = object_points.shape[1]
    # Pairwise squared distances (B, P, N)
    pc_sq = patch_centers.pow(2).sum(dim=-1, keepdim=True)
    op_sq = object_points.pow(2).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(patch_centers, object_points.transpose(-1, -2))
    dist_sq = (pc_sq + op_sq - 2.0 * cross).clamp_min(0.0)
    # Nearest k indices per patch
    k = min(int(k_nearest), N)
    _, nearest_idx = dist_sq.topk(k, dim=-1, largest=False)  # (B, P, k)
    # Gather contact values for those points
    contact_mask_exp = contact_mask.unsqueeze(1).expand(-1, P, -1)  # (B, P, N)
    nearest_contact = contact_mask_exp.gather(dim=-1, index=nearest_idx)  # (B, P, k)
    return nearest_contact.mean(dim=-1)  # (B, P)


@dataclass
class ConditioningBundle:
    """Container for the conditioning tokens consumed by the velocity network.

    Attributes:
        tokens:        Full token sequence shape (B, TOTAL_TOKENS, hidden_dim).
                       Layout along the token axis:
                         [0:64]  → object patches
                         [64:69] → 5 finger tokens
                         [69:70] → 1 palm token
                         [70:71] → 1 unified token
        object_tokens: View of tokens[:, 0:64, :] for ablation / debug.
        contact_tokens: View of tokens[:, 64:71, :] (finger + palm + unified, 7 tokens).
        token_types:   (TOTAL_TOKENS,) long tensor with the per-position type id.
        attention_bias: Optional (B, T) additive bias for cross-attention logits.
                       When set, the velocity network's cross-attention adds
                       this to each query-key logit before softmax. Used by
                       the advanced part-aware gating mode (multi-scale log
                       domain distance bias).
    """
    tokens: Tensor
    object_tokens: Tensor
    contact_tokens: Tensor
    token_types: Tensor
    attention_bias: Optional[Tensor] = None


def extract_finger_features(graph: Dict[str, Tensor], active_finger_prob: Tensor) -> Tensor:
    """Pack per-finger graph features into (B, 5, FINGER_FEATURE_DIM).

    Args:
        graph: dict produced by `graspauto.stage3_contact_graph.derive_contact_graph`,
            containing finger_centroid (B, 5, 3), finger_normal (B, 5, 3),
            finger_spread (B, 5, 3), finger_entropy (B, 5), finger_mass (B, 5).
        active_finger_prob: (B, 5) sigmoid probabilities from the active head.

    Returns:
        Tensor of shape (B, 5, 12).
    """
    centroid = graph["finger_centroid"]                      # (B, 5, 3)
    normal = graph["finger_normal"]                          # (B, 5, 3)
    spread = graph["finger_spread"]                          # (B, 5, 3)
    entropy = graph["finger_entropy"].unsqueeze(-1)          # (B, 5, 1)
    mass = graph["finger_mass"].unsqueeze(-1)                # (B, 5, 1)
    active = active_finger_prob.unsqueeze(-1)                # (B, 5, 1)
    return torch.cat([centroid, normal, spread, entropy, mass, active], dim=-1)


def extract_palm_features(graph: Dict[str, Tensor]) -> Tensor:
    """Pack palm graph features into (B, 1, PALM_FEATURE_DIM)."""
    centroid = graph["palm_centroid"]                        # (B, 3)
    normal = graph["palm_normal"]                            # (B, 3)
    spread = graph["palm_spread"]                            # (B, 3)
    entropy = graph["palm_entropy"].unsqueeze(-1)            # (B, 1)
    mass = graph["palm_mass"].unsqueeze(-1)                  # (B, 1)
    palm = torch.cat([centroid, normal, spread, entropy, mass], dim=-1)  # (B, 11)
    return palm.unsqueeze(1)                                 # (B, 1, 11)


def extract_unified_features(graph: Dict[str, Tensor]) -> Tensor:
    """Pack unified contact features into (B, 1, UNIFIED_FEATURE_DIM)."""
    centroid = graph["unified_centroid"]                     # (B, 3)
    normal = graph["unified_normal"]                         # (B, 3)
    spread = graph["unified_spread"]                         # (B, 3)
    entropy = graph["unified_entropy"].unsqueeze(-1)         # (B, 1)
    unified = torch.cat([centroid, normal, spread, entropy], dim=-1)  # (B, 10)
    return unified.unsqueeze(1)                              # (B, 1, 10)


class ContactGraphConditioningAdapter(nn.Module):
    """Project graspauto contact graph + Point-M2AE features into a unified token sequence.

    Each kind of token gets its own input projection (so the dimensions
    can differ — object patches are 384-D, finger features are 12-D, etc.)
    and a learned type embedding so cross-attention can distinguish them.

    The output is a single (B, 71, hidden_dim) tensor ready to be used as
    keys+values in the velocity network's cross-attention layers.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        object_patch_dim: int = OBJECT_PATCH_DIM,
        finger_dim: int = FINGER_FEATURE_DIM,
        palm_dim: int = PALM_FEATURE_DIM,
        unified_dim: int = UNIFIED_FEATURE_DIM,
        use_intent_token: bool = False,
        part_aware_gating: bool = False,
        palm_only_intent: bool = False,
        advanced_gating: bool = False,
        adv_sinusoidal_freqs: int = 8,
        residual_modulation: bool = False,
        use_intent_direction: bool = False,  # watermelon r003+
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_intent_token = bool(use_intent_token) and not palm_only_intent
        self.part_aware_gating = bool(part_aware_gating)
        self.palm_only_intent = bool(palm_only_intent)
        self.advanced_gating = bool(advanced_gating)
        self.adv_sinusoidal_freqs = int(adv_sinusoidal_freqs)
        self.residual_modulation = bool(residual_modulation)
        self.use_intent_direction = bool(use_intent_direction)

        # Projection from sinusoidal offset encoding to hidden_dim.
        # Only created when advanced_gating is enabled.
        if self.advanced_gating:
            sin_dim = 3 * 2 * self.adv_sinusoidal_freqs  # 3 axes * (sin,cos) * freqs
            self.offset_pe_proj = nn.Linear(sin_dim, self.hidden_dim)

        self.object_proj = nn.Linear(object_patch_dim, self.hidden_dim)
        self.finger_proj = nn.Linear(finger_dim, self.hidden_dim)
        self.palm_proj = nn.Linear(palm_dim, self.hidden_dim)
        self.unified_proj = nn.Linear(unified_dim, self.hidden_dim)

        # Residual modulation (GPT Pro #1): replaces scalar multiply gate with
        # token = token × (1 + α×w) + β×w×tap_embed. Preserves magnitude.
        # Also adds 3 summary tokens: z_near, z_far, z_global.
        if self.residual_modulation:
            self.mod_alpha = nn.Parameter(torch.tensor(0.5))  # init: mild modulation
            self.mod_beta = nn.Parameter(torch.tensor(0.1))   # init: small additive
            self.tap_proj = nn.Linear(3, self.hidden_dim)     # project palm target 3-D → hidden

        if self.use_intent_token:
            self.intent_embed = nn.Embedding(NUM_INTENT_CATEGORIES, self.hidden_dim)

        # (sphere) explicit user-supplied approach direction.
        # 3-D unit vector (normalize(tap - user_hand_base)) projected to
        # hidden_dim. When user doesn't provide direction (or during the
        # training mask), we substitute a LEARNED "absent" embedding instead
        # of a zero vector — preserves token magnitude so downstream layers
        # don't see an OOD zero-magnitude key.
        if self.use_intent_direction:
            self.direction_proj = nn.Linear(3, self.hidden_dim)
            self.direction_absent = nn.Parameter(torch.zeros(self.hidden_dim))
            nn.init.normal_(self.direction_absent, std=0.02)

        # NUM_TOKEN_TYPES=6 (incl. TOKEN_TYPE_DIRECTION); +1 for summary tokens when residual_modulation is on.
        self.type_embed = nn.Embedding(NUM_TOKEN_TYPES + (1 if self.residual_modulation else 0), self.hidden_dim)

        # Pre-compute the token-type id sequence — registered as a buffer so it
        # follows .to(device) automatically.
        if self.palm_only_intent:
            # Palm-only: 64 object + 1 palm (+1 direction if use_intent_direction)
            # (+ 3 summary if residual_modulation)
            n_summary = 3 if self.residual_modulation else 0
            n_direction = NUM_DIRECTION_TOKENS if self.use_intent_direction else 0
            total = NUM_OBJECT_TOKENS + NUM_PALM_TOKENS + n_direction + n_summary
            types = torch.zeros(total, dtype=torch.long)
            types[0:NUM_OBJECT_TOKENS] = TOKEN_TYPE_OBJECT
            types[NUM_OBJECT_TOKENS:NUM_OBJECT_TOKENS + NUM_PALM_TOKENS] = TOKEN_TYPE_PALM
            cursor = NUM_OBJECT_TOKENS + NUM_PALM_TOKENS
            if self.use_intent_direction:
                types[cursor:cursor + n_direction] = TOKEN_TYPE_DIRECTION
                cursor += n_direction
            if self.residual_modulation:
                types[cursor:cursor + n_summary] = NUM_TOKEN_TYPES  # type 6 = summary
        else:
            total = TOTAL_TOKENS_WITH_INTENT if self.use_intent_token else TOTAL_TOKENS
            types = torch.zeros(total, dtype=torch.long)
            types[0:NUM_OBJECT_TOKENS] = TOKEN_TYPE_OBJECT
            types[NUM_OBJECT_TOKENS:NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS] = TOKEN_TYPE_FINGER
            types[
                NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS:
                NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS
            ] = TOKEN_TYPE_PALM
            types[
                NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS:
                NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS + NUM_UNIFIED_TOKENS
            ] = TOKEN_TYPE_UNIFIED
            if self.use_intent_token:
                types[NUM_OBJECT_TOKENS + NUM_FINGER_TOKENS + NUM_PALM_TOKENS + NUM_UNIFIED_TOKENS:] = TOKEN_TYPE_INTENT
        self.register_buffer("token_types", types, persistent=False)

    def forward(
        self,
        m2ae_local: Tensor,
        graph: Dict[str, Tensor],
        active_finger_prob: Tensor,
        patch_contact_weight: Optional[Tensor] = None,
        intent_ids: Optional[Tensor] = None,
        patch_centers: Optional[Tensor] = None,
        palm_target: Optional[Tensor] = None,
        intent_direction: Optional[Tensor] = None,
        direction_mask: Optional[Tensor] = None,
    ) -> ConditioningBundle:
        """Build the conditioning bundle.

        Args:
            m2ae_local: (B, 64, 384) — Point-M2AE local patch tokens.
            graph: dict with finger_*, palm_*, unified_* keys.
            active_finger_prob: (B, 5) sigmoid output of the active head.
            patch_contact_weight: (B, 64) per-patch contact weight in [0, 1].
                When `part_aware_gating=True` was set at init, the object
                tokens are multiplied by this weight before being emitted
                to the cross-attention stack. Patches far from any contact
                are effectively silenced. Ignored if `part_aware_gating=False`.
            intent_ids: (B,) long tensor of intent category ids (0=use,
                1=handoff). When `use_intent_token=True`, these become an
                extra token appended to the bundle. Ignored otherwise.
        """
        if m2ae_local.dim() != 3:
            raise ValueError(f"m2ae_local must be 3-D (B, P, D), got {m2ae_local.dim()}-D")
        batch_size, num_patches, patch_dim = m2ae_local.shape
        if num_patches != NUM_OBJECT_PATCHES:
            raise ValueError(
                f"expected {NUM_OBJECT_PATCHES} patches, got {num_patches}"
            )
        if patch_dim != self.object_proj.in_features:
            raise ValueError(
                f"object patch dim {patch_dim} != adapter expected {self.object_proj.in_features}"
            )

        # Project each modality to (B, *, hidden_dim)
        object_tokens = self.object_proj(m2ae_local)                                  # (B, 64, H)

        # Part-aware spatial gating: multiply object tokens by their contact
        # weight so patches far from any grasp-relevant region are silenced.
        if self.part_aware_gating and not self.advanced_gating:
            if patch_contact_weight is None:
                raise ValueError("part_aware_gating=True but patch_contact_weight is None")
            if patch_contact_weight.shape != (batch_size, NUM_OBJECT_PATCHES):
                raise ValueError(
                    f"patch_contact_weight must be (B, {NUM_OBJECT_PATCHES}), got {tuple(patch_contact_weight.shape)}"
                )
            w = patch_contact_weight.clamp(min=0.0, max=1.0).unsqueeze(-1)  # (B, 64, 1)

            if self.residual_modulation:
                # GPT Pro #1: residual modulation preserves patch magnitude.
                # token = token × (1 + α×w) + β×w×tap_embed
                tap_embed = self.tap_proj(graph["palm_centroid"]).unsqueeze(1)  # (B, 1, H)
                object_tokens = object_tokens * (1.0 + self.mod_alpha * w) + self.mod_beta * w * tap_embed
            else:
                # Legacy scalar-multiply gate.
                object_tokens = object_tokens * (0.1 + 0.9 * w)

        # --- Advanced gating (multi-scale attention bias + contact-aware positional
        #     encoding). Does NOT silence tokens; instead biases the downstream
        #     cross-attention and enriches each patch with its offset to the palm
        #     target. ---
        attention_bias = None
        if self.advanced_gating:
            if patch_centers is None or palm_target is None:
                raise ValueError(
                    "advanced_gating=True requires patch_centers and palm_target inputs"
                )
            if patch_centers.shape != (batch_size, NUM_OBJECT_PATCHES, 3):
                raise ValueError(
                    f"patch_centers must be (B, {NUM_OBJECT_PATCHES}, 3), got {tuple(patch_centers.shape)}"
                )
            if palm_target.shape != (batch_size, 3):
                raise ValueError(
                    f"palm_target must be (B, 3), got {tuple(palm_target.shape)}"
                )
            # #7 Contact-aware sinusoidal positional encoding:
            # each object patch token sees its 3-D offset from the palm target.
            # r023 lesson: full-weight PE dilutes Point-M2AE semantics. Use 0.25x.
            offsets = patch_centers - palm_target.unsqueeze(1)                # (B, P, 3)
            sin_enc = sinusoidal_offset_encoding(offsets, num_freqs=self.adv_sinusoidal_freqs)
            pe = self.offset_pe_proj(sin_enc)                                 # (B, P, H)
            object_tokens = object_tokens + 0.25 * pe

            # #2 + #4: log-domain multi-scale attention bias over object patches.
            obj_bias = compute_multiscale_attention_bias(
                patch_centers=patch_centers,
                target_point=palm_target,
                sigmas_m=(0.03, 0.06, 0.12),
            )                                                                  # (B, P)

        if self.palm_only_intent:
            # Palm-only: skip finger + unified tokens. Only the palm token is
            # emitted alongside the 64 object patches.
            palm_features = extract_palm_features(graph)                              # (B, 1, 11)
            palm_tokens = self.palm_proj(palm_features)                               # (B, 1, H)
            token_list = [object_tokens, palm_tokens]

            # (sphere) emit the direction token right after palm.
            # If `intent_direction` is None (user didn't supply direction) OR the
            # per-sample `direction_mask` says "masked", we substitute the
            # learned `direction_absent` embedding, which preserves token
            # magnitude (vs a plain zero vector).
            if self.use_intent_direction:
                if intent_direction is None:
                    # Whole batch has no direction — use absent token everywhere.
                    dir_tok = self.direction_absent.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                else:
                    if intent_direction.shape != (batch_size, 3):
                        raise ValueError(f"intent_direction must be (B, 3), got {tuple(intent_direction.shape)}")
                    # Project each (B,3) direction → (B,H)
                    dir_proj = self.direction_proj(intent_direction).unsqueeze(1)  # (B, 1, H)
                    if direction_mask is not None:
                        # direction_mask: (B,) bool, True = masked (use absent)
                        if direction_mask.shape != (batch_size,):
                            raise ValueError(
                                f"direction_mask must be (B,), got {tuple(direction_mask.shape)}"
                            )
                        absent = self.direction_absent.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                        m = direction_mask.view(batch_size, 1, 1).to(dir_proj.dtype)
                        dir_tok = m * absent + (1.0 - m) * dir_proj
                    else:
                        dir_tok = dir_proj
                token_list.append(dir_tok)

            # Summary tokens (GPT Pro #1): structured spatial context.
            if self.residual_modulation and patch_contact_weight is not None:
                w = patch_contact_weight.clamp(min=0.0, max=1.0).unsqueeze(-1)  # (B, 64, 1)
                w_near = w
                w_far = (1.0 - w).clamp_min(0.01)
                # z_near: weighted mean of near-tap patches
                z_near = (object_tokens * w_near).sum(dim=1, keepdim=True) / w_near.sum(dim=1, keepdim=True).clamp_min(1e-6)
                # z_far: weighted mean of far-from-tap patches
                z_far = (object_tokens * w_far).sum(dim=1, keepdim=True) / w_far.sum(dim=1, keepdim=True).clamp_min(1e-6)
                # z_global: unweighted mean
                z_global = object_tokens.mean(dim=1, keepdim=True)
                token_list.extend([z_near, z_far, z_global])
        else:
            finger_features = extract_finger_features(graph, active_finger_prob)      # (B, 5, 12)
            finger_tokens = self.finger_proj(finger_features)                         # (B, 5, H)
            palm_features = extract_palm_features(graph)                              # (B, 1, 11)
            palm_tokens = self.palm_proj(palm_features)                               # (B, 1, H)
            unified_features = extract_unified_features(graph)                        # (B, 1, 10)
            unified_tokens = self.unified_proj(unified_features)                      # (B, 1, H)
            token_list = [object_tokens, finger_tokens, palm_tokens, unified_tokens]

            if self.use_intent_token:
                if intent_ids is None:
                    raise ValueError("use_intent_token=True but intent_ids is None")
                if intent_ids.dim() != 1 or intent_ids.shape[0] != batch_size:
                    raise ValueError(f"intent_ids must be (B,), got {tuple(intent_ids.shape)}")
                intent_tok = self.intent_embed(intent_ids).unsqueeze(1)  # (B, 1, H)
                token_list.append(intent_tok)

        all_tokens = torch.cat(token_list, dim=1)  # (B, 65 / 71 / 72, H)

        # Add type embeddings (broadcast across the batch)
        type_emb = self.type_embed(self.token_types)  # (T, H)
        all_tokens = all_tokens + type_emb.unsqueeze(0)

        contact_tokens = all_tokens[:, NUM_OBJECT_TOKENS:, :]

        # Assemble full attention_bias matching all_tokens' token axis.
        # obj_bias is (B, 64). We prepend it and pad the non-object tokens
        # with 0.0 (no bias).
        if self.advanced_gating:
            total_tokens = all_tokens.shape[1]
            num_non_obj = total_tokens - NUM_OBJECT_TOKENS
            non_obj_bias = torch.zeros(batch_size, num_non_obj, device=all_tokens.device, dtype=obj_bias.dtype)
            attention_bias = torch.cat([obj_bias, non_obj_bias], dim=1)  # (B, T)

        return ConditioningBundle(
            tokens=all_tokens,
            object_tokens=all_tokens[:, :NUM_OBJECT_TOKENS, :],
            contact_tokens=contact_tokens,
            token_types=self.token_types,
            attention_bias=attention_bias,
        )


__all__ = [
    "NUM_FINGERS",
    "NUM_OBJECT_PATCHES",
    "OBJECT_PATCH_DIM",
    "FINGER_FEATURE_DIM",
    "PALM_FEATURE_DIM",
    "UNIFIED_FEATURE_DIM",
    "TOTAL_TOKENS",
    "TOKEN_TYPE_OBJECT",
    "TOKEN_TYPE_FINGER",
    "TOKEN_TYPE_PALM",
    "TOKEN_TYPE_UNIFIED",
    "ConditioningBundle",
    "ContactGraphConditioningAdapter",
    "extract_finger_features",
    "extract_palm_features",
    "extract_unified_features",
]
