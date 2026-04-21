"""Stage 3 contact-graph model for the 2026-04-05 mainline.

This module replaces the older Stage 3 Point-M2AE head with a richer
object-centric target:
- unified contact
- 5 finger-specific soft heatmaps
- palm/support heatmap
- active-finger probabilities
- derived finger-contact graph statistics
- learned graph embedding for multi-positive code retrieval
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from graspauto.utils import (
    DEFAULT_GEOMETRY_CACHE,
    FINGER_NAMES,
    move_batch_to_device,
    resolve_device,
)
from graspauto.preprocessing import (
    DEFAULT_M2AE_WEIGHTS,
    precompute_object_m2ae_cache,
)
from graspauto.stage3_utils import MultiPositiveContrastiveLoss, ProjectionHead

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE3_CONTACT_GRAPH_ROOT = PROJECT_ROOT / "outputs" / "stage3_contact_graph"
DEFAULT_OBJECT_M2AE_CACHE = DEFAULT_STAGE3_CONTACT_GRAPH_ROOT / "object_m2ae_cache.pt"
DEFAULT_CODEBANK_METADATA = DEFAULT_STAGE3_CONTACT_GRAPH_ROOT / "codebank_metadata.pt"
EPS = 1e-8


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


def _sigmoid_dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    denom = (pred_prob + target).sum(dim=dims).clamp_min(EPS)
    dice = 2.0 * (pred_prob * target).sum(dim=dims) / denom
    return 1.0 - dice.mean()


def _weighted_stats_batch(
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if weights.dim() == 2:
        weights = weights.unsqueeze(-1)
    weights = weights.clamp_min(0.0)
    mass = weights.sum(dim=1)
    probs = weights / mass.unsqueeze(1).clamp_min(EPS)

    centroid = torch.einsum("bnd,bnf->bfd", object_points, probs)
    normal = _safe_normalize(torch.einsum("bnd,bnf->bfd", object_normals, probs), dim=-1)

    centered = object_points.unsqueeze(2) - centroid.unsqueeze(1)
    cov = torch.einsum("bnfi,bnfj,bnf->bfij", centered, centered, probs)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    spread = torch.sqrt(eigvals)
    entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=1)

    return {
        "centroid": centroid,
        "normal": normal,
        "cov": cov,
        "spread": spread,
        "entropy": entropy,
        "mass": mass,
    }


def _pairwise_dirs(centroids: torch.Tensor) -> torch.Tensor:
    diffs = centroids.unsqueeze(2) - centroids.unsqueeze(1)
    norms = diffs.norm(dim=-1, keepdim=True)
    dirs = diffs / norms.clamp_min(EPS)
    diag = torch.eye(centroids.shape[1], device=centroids.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    dirs = dirs.masked_fill(diag, 0.0)
    return dirs


def derive_contact_graph(
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    unified_prob: torch.Tensor,
    finger_prob: torch.Tensor,
    palm_prob: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    finger = _weighted_stats_batch(object_points, object_normals, finger_prob)
    palm = _weighted_stats_batch(object_points, object_normals, palm_prob)
    unified = _weighted_stats_batch(object_points, object_normals, unified_prob)

    finger_pairwise_dist = torch.cdist(finger["centroid"], finger["centroid"])
    finger_pairwise_dir = _pairwise_dirs(finger["centroid"])
    finger_pairwise_normal_cos = torch.einsum("bfd,bgd->bfg", finger["normal"], finger["normal"])
    finger_dirs_from_palm = _safe_normalize(finger["centroid"] - palm["centroid"], dim=-1)

    return {
        "finger_centroid": finger["centroid"],
        "finger_normal": finger["normal"],
        "finger_cov": finger["cov"],
        "finger_spread": finger["spread"],
        "finger_entropy": finger["entropy"],
        "finger_mass": finger["mass"],
        "palm_centroid": palm["centroid"].squeeze(1),
        "palm_normal": palm["normal"].squeeze(1),
        "palm_cov": palm["cov"].squeeze(1),
        "palm_spread": palm["spread"].squeeze(1),
        "palm_entropy": palm["entropy"].squeeze(1),
        "palm_mass": palm["mass"].squeeze(1),
        "unified_centroid": unified["centroid"].squeeze(1),
        "unified_normal": unified["normal"].squeeze(1),
        "unified_cov": unified["cov"].squeeze(1),
        "unified_spread": unified["spread"].squeeze(1),
        "unified_entropy": unified["entropy"].squeeze(1),
        "finger_pairwise_dist": finger_pairwise_dist,
        "finger_pairwise_dir": finger_pairwise_dir,
        "finger_pairwise_normal_cos": finger_pairwise_normal_cos,
        "finger_dirs_from_palm": finger_dirs_from_palm,
    }


def build_graph_feature_vector(graph: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            graph["finger_centroid"].reshape(graph["finger_centroid"].shape[0], -1),
            graph["finger_normal"].reshape(graph["finger_normal"].shape[0], -1),
            graph["finger_spread"].reshape(graph["finger_spread"].shape[0], -1),
            graph["finger_entropy"],
            graph["finger_mass"],
            graph["finger_pairwise_dist"].reshape(graph["finger_pairwise_dist"].shape[0], -1),
            graph["finger_pairwise_normal_cos"].reshape(graph["finger_pairwise_normal_cos"].shape[0], -1),
            graph["finger_dirs_from_palm"].reshape(graph["finger_dirs_from_palm"].shape[0], -1),
            graph["palm_centroid"],
            graph["palm_normal"],
            graph["palm_spread"],
            graph["palm_entropy"].unsqueeze(-1),
            graph["palm_mass"].unsqueeze(-1),
            graph["unified_centroid"],
            graph["unified_normal"],
            graph["unified_spread"],
            graph["unified_entropy"].unsqueeze(-1),
        ],
        dim=-1,
    )


class Stage3ContactGraphDataset(Dataset):
    """Dataset for the new Stage 3 contact-graph path."""

    def __init__(
        self,
        split_path: Path | str,
        geometry_path: Path | str = DEFAULT_GEOMETRY_CACHE,
        object_m2ae_cache_path: Path | str = DEFAULT_OBJECT_M2AE_CACHE,
        limit: Optional[int] = None,
    ) -> None:
        self.geometry = torch.load(Path(geometry_path), map_location="cpu", weights_only=False)
        self.split = torch.load(Path(split_path), map_location="cpu", weights_only=False)
        self.object_cache = torch.load(Path(object_m2ae_cache_path), map_location="cpu", weights_only=False)

        self.object_points = self.geometry["object_points"].float()
        self.object_normals = self.geometry["object_normals"].float()
        self.object_id = self.split["object_id"].long()
        self.stage1_contact_input = self.split["stage1_contact_input"].float()
        self.unified_contact_target = self.split["unified_contact_target"].float()
        self.finger_contact_target = self.split["finger_contact_target"].float()
        self.palm_contact_target = self.split["palm_contact_target"].float()
        self.active_finger_mask = self.split["active_finger_mask"].float()
        self.active_finger_score = self.split["active_finger_score"].float()
        self.gt_code_id = self.split["gt_code_id"].long()
        self.pose_48 = self.split["pose_48"].float()
        self.betas = self.split["betas"].float()
        self.hTm_rot = self.split["hTm_rot"].float()
        self.hTm_trans = self.split["hTm_trans"].float()
        self.gt_local_verts = self.split["gt_local_verts"].float()
        self.gt_world_verts = self.split["gt_world_verts"].float()

        self.finger_centroid = self.split["finger_centroid"].float()
        self.finger_normal = self.split["finger_normal"].float()
        self.finger_cov = self.split["finger_cov"].float()
        self.finger_spread = self.split["finger_spread"].float()
        self.finger_entropy = self.split["finger_entropy"].float()
        self.finger_mass = self.split["finger_mass"].float()
        self.palm_centroid = self.split["palm_centroid"].float()
        self.palm_normal = self.split["palm_normal"].float()
        self.palm_spread = self.split["palm_spread"].float()
        self.palm_entropy = self.split["palm_entropy"].float()
        self.palm_mass = self.split["palm_mass"].float()
        self.unified_centroid = self.split["unified_centroid"].float()
        self.unified_normal = self.split["unified_normal"].float()
        self.unified_spread = self.split["unified_spread"].float()
        self.unified_entropy = self.split["unified_entropy"].float()
        self.finger_pairwise_dist = self.split["finger_pairwise_dist"].float()
        self.finger_pairwise_dir = self.split["finger_pairwise_dir"].float()
        self.finger_pairwise_normal_cos = self.split["finger_pairwise_normal_cos"].float()
        if "finger_dirs_from_palm" in self.split:
            self.finger_dirs_from_palm = self.split["finger_dirs_from_palm"].float()
        else:
            self.finger_dirs_from_palm = _safe_normalize(self.finger_centroid - self.palm_centroid.unsqueeze(1), dim=-1)

        self.soft_code_id = self.split.get("soft_code_id")
        self.soft_code_weight = self.split.get("soft_code_weight")
        self.soft_code_error_mm = self.split.get("soft_code_error_mm")
        self.oracle_code_id = self.split.get("oracle_code_id")
        self.oracle_code_weight = self.split.get("oracle_code_weight")
        self.oracle_code_error_mm = self.split.get("oracle_code_error_mm")

        self.m2ae_global = self.object_cache["m2ae_global"].float()
        self.m2ae_local = self.object_cache["m2ae_local"].float()
        self.patch_centers = self.object_cache["patch_centers"].float()

        # Intent label (ContactPose: 'use' / 'handoff'). Stored in the split
        # file as a list[str]; convert to a long tensor for batching. Unknown
        # values default to 0 (use).
        raw_intent = self.split.get("intent")
        if raw_intent is not None:
            _INTENT_MAP = {"use": 0, "handoff": 1}
            self.intent_id = torch.tensor(
                [_INTENT_MAP.get(str(x).strip().lower(), 0) for x in raw_intent],
                dtype=torch.long,
            )
        else:
            self.intent_id = torch.zeros(int(self.object_id.shape[0]), dtype=torch.long)

        if self.soft_code_id is not None:
            self.soft_code_id = self.soft_code_id.long()
            self.soft_code_weight = self.soft_code_weight.float()
            self.soft_code_error_mm = self.soft_code_error_mm.float()
        if self.oracle_code_id is not None:
            self.oracle_code_id = self.oracle_code_id.long()
            self.oracle_code_weight = self.oracle_code_weight.float()
            self.oracle_code_error_mm = self.oracle_code_error_mm.float()

        if limit is not None:
            limit = min(int(limit), self.object_id.shape[0])
            base_len = int(self.object_id.shape[0])
            for name in list(self.__dict__.keys()):
                value = getattr(self, name)
                if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == base_len:
                    setattr(self, name, value[:limit])

    def __len__(self) -> int:
        return int(self.object_id.shape[0])

    @property
    def num_codes(self) -> int:
        if self.soft_code_id is not None:
            return int(max(self.gt_code_id.max().item(), self.soft_code_id.max().item()) + 1)
        return int(self.gt_code_id.max().item()) + 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        object_id = int(self.object_id[index].item())
        sample = {
            "sample_index": torch.tensor(index, dtype=torch.long),
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "object_points": self.object_points[object_id],
            "object_normals": self.object_normals[object_id],
            "stage1_contact_input": self.stage1_contact_input[index],
            "m2ae_global": self.m2ae_global[object_id],
            "m2ae_local": self.m2ae_local[object_id],
            "patch_centers": self.patch_centers[object_id],
            "unified_contact_target": self.unified_contact_target[index],
            "finger_contact_target": self.finger_contact_target[index],
            "palm_contact_target": self.palm_contact_target[index],
            "active_finger_mask": self.active_finger_mask[index],
            "active_finger_score": self.active_finger_score[index],
            "gt_code_id": self.gt_code_id[index],
            "pose_48": self.pose_48[index],
            "betas": self.betas[index],
            "hTm_rot": self.hTm_rot[index],
            "hTm_trans": self.hTm_trans[index],
            "gt_local_verts": self.gt_local_verts[index],
            "gt_world_verts": self.gt_world_verts[index],
            "finger_centroid": self.finger_centroid[index],
            "finger_normal": self.finger_normal[index],
            "finger_cov": self.finger_cov[index],
            "finger_spread": self.finger_spread[index],
            "finger_entropy": self.finger_entropy[index],
            "finger_mass": self.finger_mass[index],
            "palm_centroid": self.palm_centroid[index],
            "palm_normal": self.palm_normal[index],
            "palm_spread": self.palm_spread[index],
            "palm_entropy": self.palm_entropy[index],
            "palm_mass": self.palm_mass[index],
            "unified_centroid": self.unified_centroid[index],
            "unified_normal": self.unified_normal[index],
            "unified_spread": self.unified_spread[index],
            "unified_entropy": self.unified_entropy[index],
            "finger_pairwise_dist": self.finger_pairwise_dist[index],
            "finger_pairwise_dir": self.finger_pairwise_dir[index],
            "finger_pairwise_normal_cos": self.finger_pairwise_normal_cos[index],
            "finger_dirs_from_palm": self.finger_dirs_from_palm[index],
            "intent_id": self.intent_id[index],
        }
        if self.soft_code_id is not None:
            sample["soft_code_id"] = self.soft_code_id[index]
            sample["soft_code_weight"] = self.soft_code_weight[index]
            sample["soft_code_error_mm"] = self.soft_code_error_mm[index]
        if self.oracle_code_id is not None:
            sample["oracle_code_id"] = self.oracle_code_id[index]
            sample["oracle_code_weight"] = self.oracle_code_weight[index]
            sample["oracle_code_error_mm"] = self.oracle_code_error_mm[index]
        return sample


class PointM2AEContactGraphModel(nn.Module):
    """Point-M2AE contact-graph predictor with learned retrieval embedding."""

    def __init__(
        self,
        num_codes: int,
        hidden_dim: int = 256,
        token_dim: int = 384,
        global_dim: int = 1024,
        num_heads: int = 4,
        graph_embed_dim: int = 256,
        dropout: float = 0.1,
        use_metric_projection: bool = False,
        metric_dim: int = 256,
        code_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.graph_embed_dim = int(graph_embed_dim)
        self.use_metric_projection = bool(use_metric_projection)
        self.metric_dim = int(metric_dim)

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.point_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.point_norm = nn.LayerNorm(hidden_dim)
        self.point_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.unified_contact_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.finger_contact_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(FINGER_NAMES)))
        self.palm_contact_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.active_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(FINGER_NAMES)),
        )

        graph_feature_dim = 15 + 15 + 15 + 5 + 5 + 25 + 25 + 15 + 3 + 3 + 3 + 1 + 1 + 3 + 3 + 3 + 1
        self.graph_stat_proj = nn.Sequential(
            nn.LayerNorm(graph_feature_dim),
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        fusion_dim = global_dim + hidden_dim * 5
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.graph_embed_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + len(FINGER_NAMES), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_embed_dim),
        )
        self.codebook_embed = nn.Embedding(num_codes, graph_embed_dim)
        self.query_projection = None
        self.code_projection = None
        if self.use_metric_projection:
            self.query_projection = ProjectionHead(
                in_dim=graph_embed_dim,
                hidden_dim=hidden_dim * 2,
                out_dim=self.metric_dim,
                dropout=dropout,
            )
            if code_features is not None:
                code_features = code_features.float()
                if int(code_features.shape[0]) != self.num_codes:
                    raise ValueError(f"Expected {self.num_codes} code features, got {tuple(code_features.shape)}")
                self.register_buffer("code_feature_bank", code_features, persistent=False)
                self.code_projection = ProjectionHead(
                    in_dim=int(code_features.shape[-1]),
                    hidden_dim=hidden_dim * 2,
                    out_dim=self.metric_dim,
                    dropout=dropout,
                )
        if not hasattr(self, "code_feature_bank"):
            self.code_feature_bank = None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(
        self,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        stage1_contact_input: torch.Tensor,
        m2ae_global: torch.Tensor,
        m2ae_local: torch.Tensor,
        patch_centers: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        token_hidden = self.token_proj(torch.cat([m2ae_local, patch_centers], dim=-1))
        point_input = torch.cat([object_points, object_normals, stage1_contact_input.unsqueeze(-1)], dim=-1)
        point_hidden = self.point_proj(point_input)
        attended, _ = self.cross_attn(point_hidden, token_hidden, token_hidden)
        point_hidden = self.point_norm(point_hidden + attended)
        point_hidden = point_hidden + self.point_ffn(point_hidden)

        unified_contact_logits = self.unified_contact_head(point_hidden).squeeze(-1)
        finger_contact_logits = self.finger_contact_head(point_hidden)
        palm_contact_logits = self.palm_contact_head(point_hidden).squeeze(-1)

        unified_contact_prob = torch.sigmoid(unified_contact_logits)
        finger_contact_prob = torch.sigmoid(finger_contact_logits)
        palm_contact_prob = torch.sigmoid(palm_contact_logits)
        union_from_parts = torch.maximum(finger_contact_prob.max(dim=-1).values, palm_contact_prob)

        stage1_w = stage1_contact_input / stage1_contact_input.sum(dim=1, keepdim=True).clamp_min(EPS)
        unified_w = unified_contact_prob / unified_contact_prob.sum(dim=1, keepdim=True).clamp_min(EPS)
        union_w = union_from_parts / union_from_parts.sum(dim=1, keepdim=True).clamp_min(EPS)
        stage1_context = (point_hidden * stage1_w.unsqueeze(-1)).sum(dim=1)
        unified_context = (point_hidden * unified_w.unsqueeze(-1)).sum(dim=1)
        union_context = (point_hidden * union_w.unsqueeze(-1)).sum(dim=1)
        point_mean = point_hidden.mean(dim=1)
        point_max = point_hidden.max(dim=1).values

        active_input = torch.cat([unified_context, union_context], dim=-1)
        active_finger_logits = self.active_head(active_input)
        active_finger_prob = torch.sigmoid(active_finger_logits)

        graph = derive_contact_graph(
            object_points=object_points,
            object_normals=object_normals,
            unified_prob=unified_contact_prob,
            finger_prob=finger_contact_prob,
            palm_prob=palm_contact_prob,
        )
        graph_feature_vector = build_graph_feature_vector(graph)
        graph_context = self.graph_stat_proj(graph_feature_vector)
        global_hidden = self.global_mlp(
            torch.cat([m2ae_global, point_mean, point_max, stage1_context, unified_context, graph_context], dim=-1)
        )
        graph_embed = self.graph_embed_head(torch.cat([global_hidden, graph_context, active_finger_prob], dim=-1))
        graph_embed = F.normalize(graph_embed, dim=-1)
        if self.use_metric_projection and self.query_projection is not None:
            query_embed = self.query_projection(graph_embed)
            if self.code_projection is not None and self.code_feature_bank is not None:
                code_metric_embed = self.code_projection(self.code_feature_bank)
            else:
                code_metric_embed = F.normalize(self.codebook_embed.weight, dim=-1)
        else:
            query_embed = graph_embed
            code_metric_embed = F.normalize(self.codebook_embed.weight, dim=-1)
        logit_scale = self.logit_scale.exp().clamp_max(100.0)
        code_logits = torch.matmul(query_embed, code_metric_embed.t()) * logit_scale

        return {
            "unified_contact_logits": unified_contact_logits,
            "unified_contact_prob": unified_contact_prob,
            "finger_contact_logits": finger_contact_logits,
            "finger_contact_prob": finger_contact_prob,
            "palm_contact_logits": palm_contact_logits,
            "palm_contact_prob": palm_contact_prob,
            "union_from_parts": union_from_parts,
            "active_finger_logits": active_finger_logits,
            "active_finger_prob": active_finger_prob,
            "graph": graph,
            "graph_feature_vector": graph_feature_vector,
            "graph_context": graph_context,
            "global_hidden": global_hidden,
            "graph_embed": graph_embed,
            "query_embed": query_embed,
            "code_metric_embed": code_metric_embed,
            "code_logits": code_logits,
            "logit_scale_value": logit_scale,
            "point_hidden": point_hidden,
        }


def _soft_code_cross_entropy(
    logits: torch.Tensor,
    soft_ids: torch.Tensor,
    soft_weights: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=1, index=soft_ids)
    return -(gathered * soft_weights).sum(dim=1).mean()


def _soft_positive_prototype_loss(
    graph_embed: torch.Tensor,
    code_embed_weight: torch.Tensor,
    soft_ids: torch.Tensor,
    soft_weights: torch.Tensor,
) -> torch.Tensor:
    positive_embed = code_embed_weight.index_select(0, soft_ids.reshape(-1))
    positive_embed = positive_embed.reshape(soft_ids.shape[0], soft_ids.shape[1], -1)
    positive_embed = (positive_embed * soft_weights.unsqueeze(-1)).sum(dim=1)
    positive_embed = F.normalize(positive_embed, dim=-1)
    cosine = (graph_embed * positive_embed).sum(dim=-1)
    return (1.0 - cosine).mean()


def _offdiag_mean(x: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(x.shape[-1], device=x.device, dtype=torch.bool).unsqueeze(0)
    return x[mask].mean()


def compute_contact_graph_losses(
    model_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    unified_target = batch["unified_contact_target"]
    finger_target = batch["finger_contact_target"]
    palm_target = batch["palm_contact_target"]
    active_target = batch["active_finger_mask"]
    pred_graph = model_out["graph"]

    unified_bce = F.binary_cross_entropy_with_logits(model_out["unified_contact_logits"], unified_target)
    unified_dice = _sigmoid_dice_loss(model_out["unified_contact_prob"], unified_target, dims=(1,))
    finger_bce = F.binary_cross_entropy_with_logits(model_out["finger_contact_logits"], finger_target)
    finger_dice = _sigmoid_dice_loss(model_out["finger_contact_prob"], finger_target, dims=(1, 2))
    palm_bce = F.binary_cross_entropy_with_logits(model_out["palm_contact_logits"], palm_target)
    palm_dice = _sigmoid_dice_loss(model_out["palm_contact_prob"], palm_target, dims=(1,))
    active_bce = F.binary_cross_entropy_with_logits(model_out["active_finger_logits"], active_target)

    centroid_l1 = F.l1_loss(pred_graph["finger_centroid"], batch["finger_centroid"])
    normal_cos = 1.0 - (pred_graph["finger_normal"] * batch["finger_normal"]).sum(dim=-1)
    normal_loss = normal_cos.mean()
    spread_l1 = F.l1_loss(pred_graph["finger_spread"], batch["finger_spread"])
    entropy_l1 = F.l1_loss(pred_graph["finger_entropy"], batch["finger_entropy"])
    mass_l1 = F.l1_loss(pred_graph["finger_mass"], batch["finger_mass"])
    graph_dist_l1 = F.l1_loss(pred_graph["finger_pairwise_dist"], batch["finger_pairwise_dist"])

    pred_dir = pred_graph["finger_pairwise_dir"]
    gt_dir = batch["finger_pairwise_dir"]
    dir_cos = 1.0 - (pred_dir * gt_dir).sum(dim=-1)
    diag = torch.eye(dir_cos.shape[-1], device=dir_cos.device, dtype=torch.bool).unsqueeze(0)
    graph_dir_loss = dir_cos.masked_select(~diag).mean()
    graph_normal_cos_l1 = F.l1_loss(pred_graph["finger_pairwise_normal_cos"], batch["finger_pairwise_normal_cos"])

    palm_centroid_l1 = F.l1_loss(pred_graph["palm_centroid"], batch["palm_centroid"])
    palm_normal_loss = (1.0 - (pred_graph["palm_normal"] * batch["palm_normal"]).sum(dim=-1)).mean()
    unified_centroid_l1 = F.l1_loss(pred_graph["unified_centroid"], batch["unified_centroid"])
    unified_normal_loss = (1.0 - (pred_graph["unified_normal"] * batch["unified_normal"]).sum(dim=-1)).mean()

    union_consistency = F.l1_loss(model_out["union_from_parts"], model_out["unified_contact_prob"])
    mass_target = batch["finger_mass"] / batch["finger_mass"].amax(dim=-1, keepdim=True).clamp_min(EPS)
    mass_active = F.l1_loss(model_out["active_finger_prob"], mass_target.clamp(0.0, 1.0))

    if "oracle_code_id" in batch and "oracle_code_weight" in batch:
        target_code_id = batch["oracle_code_id"]
        target_code_weight = batch["oracle_code_weight"]
    elif "soft_code_id" in batch and "soft_code_weight" in batch:
        target_code_id = batch["soft_code_id"]
        target_code_weight = batch["soft_code_weight"]
    else:
        target_code_id = None
        target_code_weight = None

    contrastive = MultiPositiveContrastiveLoss(temperature=0.07)
    if target_code_id is not None:
        code_contrastive = contrastive(
            model_out["query_embed"],
            model_out["code_metric_embed"],
            target_code_id,
            target_code_weight,
            scale=model_out.get("logit_scale_value"),
        )
    else:
        code_contrastive = F.cross_entropy(model_out["code_logits"], batch["gt_code_id"])
    code_hard_ce = F.cross_entropy(model_out["code_logits"], batch["gt_code_id"])

    total = (
        loss_weights.get("unified_bce", 1.0) * unified_bce
        + loss_weights.get("unified_dice", 1.0) * unified_dice
        + loss_weights.get("finger_bce", 1.0) * finger_bce
        + loss_weights.get("finger_dice", 1.0) * finger_dice
        + loss_weights.get("palm_bce", 0.5) * palm_bce
        + loss_weights.get("palm_dice", 0.5) * palm_dice
        + loss_weights.get("active_bce", 0.5) * active_bce
        + loss_weights.get("centroid_l1", 1.0) * centroid_l1
        + loss_weights.get("normal_loss", 0.25) * normal_loss
        + loss_weights.get("spread_l1", 0.25) * spread_l1
        + loss_weights.get("entropy_l1", 0.1) * entropy_l1
        + loss_weights.get("mass_l1", 0.1) * mass_l1
        + loss_weights.get("graph_dist_l1", 0.5) * graph_dist_l1
        + loss_weights.get("graph_dir_loss", 0.25) * graph_dir_loss
        + loss_weights.get("graph_normal_cos_l1", 0.25) * graph_normal_cos_l1
        + loss_weights.get("palm_centroid_l1", 0.1) * palm_centroid_l1
        + loss_weights.get("palm_normal_loss", 0.05) * palm_normal_loss
        + loss_weights.get("unified_centroid_l1", 0.1) * unified_centroid_l1
        + loss_weights.get("unified_normal_loss", 0.05) * unified_normal_loss
        + loss_weights.get("union_consistency", 0.25) * union_consistency
        + loss_weights.get("mass_active", 0.25) * mass_active
        + loss_weights.get("code_contrastive", 1.0) * code_contrastive
        + loss_weights.get("code_hard_ce", 0.25) * code_hard_ce
    )

    return {
        "loss": total,
        "unified_bce": unified_bce,
        "unified_dice": unified_dice,
        "finger_bce": finger_bce,
        "finger_dice": finger_dice,
        "palm_bce": palm_bce,
        "palm_dice": palm_dice,
        "active_bce": active_bce,
        "centroid_l1": centroid_l1,
        "normal_loss": normal_loss,
        "spread_l1": spread_l1,
        "entropy_l1": entropy_l1,
        "mass_l1": mass_l1,
        "graph_dist_l1": graph_dist_l1,
        "graph_dir_loss": graph_dir_loss,
        "graph_normal_cos_l1": graph_normal_cos_l1,
        "palm_centroid_l1": palm_centroid_l1,
        "palm_normal_loss": palm_normal_loss,
        "unified_centroid_l1": unified_centroid_l1,
        "unified_normal_loss": unified_normal_loss,
        "union_consistency": union_consistency,
        "mass_active": mass_active,
        "code_contrastive": code_contrastive,
        "code_hard_ce": code_hard_ce,
    }


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    topk = logits.topk(k=min(int(k), logits.shape[-1]), dim=-1).indices
    return (topk == target.unsqueeze(1)).any(dim=1).float()


__all__ = [
    "DEFAULT_STAGE3_CONTACT_GRAPH_ROOT",
    "DEFAULT_OBJECT_M2AE_CACHE",
    "DEFAULT_CODEBANK_METADATA",
    "DEFAULT_M2AE_WEIGHTS",
    "Stage3ContactGraphDataset",
    "PointM2AEContactGraphModel",
    "derive_contact_graph",
    "build_graph_feature_vector",
    "compute_contact_graph_losses",
    "topk_accuracy",
    "move_batch_to_device",
    "precompute_object_m2ae_cache",
    "resolve_device",
]
