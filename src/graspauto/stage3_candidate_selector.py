"""Learned candidate-selector inference for Stage 3 contact-graph retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from graspauto.utils import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE3_CANDIDATE_SELECTOR_DIR = PROJECT_ROOT / "outputs" / "stage3_candidate_selector_short4_pairfeat_v2_hard32x4"
DEFAULT_STAGE3_CANDIDATE_SELECTOR_CHECKPOINT = DEFAULT_STAGE3_CANDIDATE_SELECTOR_DIR / "best.pt"

FEATURE_KEYS = [
    "candidate_scores",
    "candidate_anchor_rmse_mm",
    "candidate_refine_loss",
    "candidate_active_mismatch",
    "candidate_center_residual_mm",
    "candidate_palm_residual_mm",
    "candidate_finger_overlap",
    "candidate_unified_overlap",
    "candidate_palm_overlap",
    "pairwise_dir_min",
    "code_active_count",
    "weights_max",
    "rank_inv",
]

FEATURE_KEY_ALIASES = {
    "candidate_scores": "coarse_score",
    "candidate_anchor_rmse_mm": "anchor_residual",
    "candidate_refine_loss": "refine_loss",
    "candidate_active_mismatch": "active_mismatch",
    "candidate_center_residual_mm": "center_residual",
    "candidate_palm_residual_mm": "palm_residual",
    "candidate_finger_overlap": "finger_overlap",
    "candidate_unified_overlap": "unified_overlap",
    "candidate_palm_overlap": "palm_overlap",
}

EXCLUDED_CANDIDATE_FEATURE_KEYS = frozenset(
    {
        "code_id",
        "vertex_err_mm",
        "is_gt_code",
    }
)


class CandidateSelector(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.active_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.geom_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm(features)
        gate = self.gate(x)
        active_score = self.active_branch(x)
        geom_score = self.geom_branch(x)
        return (gate * active_score + (1.0 - gate) * geom_score).squeeze(-1)


def _canonical_feature_key(feature_key: str) -> str:
    return FEATURE_KEY_ALIASES.get(feature_key, feature_key)


def _assembly_feature_value(assembly: object, feature_key: str, rank: int) -> float:
    key = _canonical_feature_key(feature_key)
    if key == "coarse_score":
        return float(assembly.coarse_score.item())
    if key == "anchor_residual":
        return float(assembly.anchor_rmse.item() * 1000.0)
    if key == "refine_loss":
        return float(assembly.refine_loss.item())
    if key == "active_mismatch":
        return float(assembly.active_mismatch.item())
    if key == "center_residual":
        return float(assembly.center_residual.item() * 1000.0)
    if key == "palm_residual":
        return float(assembly.palm_residual.item() * 1000.0)
    if key == "finger_overlap":
        return float(assembly.finger_overlap.item())
    if key == "unified_overlap":
        return float(assembly.unified_overlap.item())
    if key == "palm_overlap":
        return float(assembly.palm_overlap.item())
    if key == "finger_overlap_max":
        return float(assembly.finger_overlap_max.item())
    if key == "finger_overlap_min":
        return float(assembly.finger_overlap_min.item())
    if key == "weights_mean":
        return float(assembly.weights_mean.item())
    if key == "weights_max":
        return float(assembly.weights_max.item())
    if key == "weights_min":
        return float(assembly.weights_min.item())
    if key == "pred_active_count":
        return float(assembly.pred_active_count.item())
    if key == "code_active_count":
        return float(assembly.code_active_count.item())
    if key == "confident_fingers":
        return float(assembly.confident_fingers.item())
    if key == "pairwise_dist_mismatch":
        return float(assembly.pairwise_dist_mismatch.item())
    if key == "pairwise_dist_max":
        return float(assembly.pairwise_dist_max.item())
    if key == "pairwise_dist_top3_mean":
        return float(assembly.pairwise_dist_top3_mean.item())
    if key == "pairwise_dir_align":
        return float(assembly.pairwise_dir_align.item())
    if key == "pairwise_dir_min":
        return float(assembly.pairwise_dir_min.item())
    if key == "pairwise_dir_bottom3_mean":
        return float(assembly.pairwise_dir_bottom3_mean.item())
    if key == "active_subgraph_dist_mismatch":
        return float(assembly.active_subgraph_dist_mismatch.item())
    if key == "active_subgraph_dir_bottom_mean":
        return float(assembly.active_subgraph_dir_bottom_mean.item())
    if key == "rerank_minus_coarse":
        return float((assembly.rerank_score - assembly.coarse_score).item())
    if key == "rank":
        return float(rank)
    if key == "rank_inv":
        return 1.0 / float(rank + 1)
    raise KeyError(f"Unsupported selector feature for assemblies: {feature_key}")


def _candidate_dict_feature_value(candidate: Mapping[str, object], feature_key: str, rank: int) -> float:
    key = _canonical_feature_key(feature_key)
    if key == "coarse_score":
        return float(candidate["coarse_score"])
    if key == "anchor_residual":
        return float(candidate["anchor_residual"]) * 1000.0
    if key == "refine_loss":
        return float(candidate["refine_loss"])
    if key == "active_mismatch":
        return float(candidate["active_mismatch"])
    if key == "center_residual":
        return float(candidate["center_residual"]) * 1000.0
    if key == "palm_residual":
        return float(candidate["palm_residual"]) * 1000.0
    if key == "finger_overlap":
        return float(candidate["finger_overlap"])
    if key == "unified_overlap":
        return float(candidate["unified_overlap"])
    if key == "palm_overlap":
        return float(candidate["palm_overlap"])
    if key == "finger_overlap_max":
        return float(candidate["finger_overlap_max"])
    if key == "finger_overlap_min":
        return float(candidate["finger_overlap_min"])
    if key == "weights_mean":
        return float(candidate["weights_mean"])
    if key == "weights_max":
        return float(candidate["weights_max"])
    if key == "weights_min":
        return float(candidate["weights_min"])
    if key == "pred_active_count":
        return float(candidate["pred_active_count"])
    if key == "code_active_count":
        return float(candidate["code_active_count"])
    if key == "confident_fingers":
        return float(candidate["confident_fingers"])
    if key == "pairwise_dist_mismatch":
        return float(candidate["pairwise_dist_mismatch"])
    if key == "pairwise_dist_max":
        return float(candidate["pairwise_dist_max"])
    if key == "pairwise_dist_top3_mean":
        return float(candidate["pairwise_dist_top3_mean"])
    if key == "pairwise_dir_align":
        return float(candidate["pairwise_dir_align"])
    if key == "pairwise_dir_min":
        return float(candidate["pairwise_dir_min"])
    if key == "pairwise_dir_bottom3_mean":
        return float(candidate["pairwise_dir_bottom3_mean"])
    if key == "active_subgraph_dist_mismatch":
        return float(candidate["active_subgraph_dist_mismatch"])
    if key == "active_subgraph_dir_bottom_mean":
        return float(candidate["active_subgraph_dir_bottom_mean"])
    if key == "rerank_minus_coarse":
        return float(candidate["rerank_minus_coarse"])
    if key == "rank":
        return float(candidate.get("rank", rank))
    if key == "rank_inv":
        return float(candidate.get("rank_inv", 1.0 / float(rank + 1)))
    raise KeyError(f"Unsupported selector feature for candidate dicts: {feature_key}")


def infer_candidate_feature_keys(
    rows: Sequence[Mapping[str, object]],
    exclude_keys: Sequence[str] | None = None,
) -> list[str]:
    excluded = set(EXCLUDED_CANDIDATE_FEATURE_KEYS)
    if exclude_keys is not None:
        excluded.update(exclude_keys)
    feature_keys: set[str] = set()
    for row in rows:
        for candidate in row["candidates"]:
            for key, value in candidate.items():
                if key in excluded:
                    continue
                if isinstance(value, (bool, int, float)):
                    feature_keys.add(str(key))
    return sorted(feature_keys)


def candidate_feature_tensor(
    candidates: Sequence[object],
    device: torch.device | None = None,
    feature_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    feature_keys = list(feature_keys or FEATURE_KEYS)
    rows = []
    for rank, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping):
            rows.append([_candidate_dict_feature_value(candidate, key, rank) for key in feature_keys])
            continue
        rows.append([_assembly_feature_value(candidate, key, rank) for key in feature_keys])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def apply_listwise_normalization(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 2:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std
    if features.dim() == 3:
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std
    raise ValueError(f"Expected feature tensor with 2 or 3 dims, got shape={tuple(features.shape)}")


class LoadedCandidateSelector:
    def __init__(
        self,
        model: CandidateSelector,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        feature_keys: Sequence[str],
        checkpoint_path: Path,
        listwise_normalize: bool = False,
    ) -> None:
        self.model = model
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_keys = list(feature_keys)
        self.checkpoint_path = checkpoint_path
        self.listwise_normalize = bool(listwise_normalize)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: str | torch.device = "cpu",
    ) -> "LoadedCandidateSelector":
        checkpoint_path = Path(checkpoint_path)
        resolved_device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        feature_keys = list(checkpoint.get("feature_keys", FEATURE_KEYS))
        model = CandidateSelector(
            feature_dim=len(feature_keys),
            hidden_dim=int(args.get("hidden_dim", 64)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(resolved_device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        feature_mean = checkpoint["feature_mean"].to(resolved_device)
        feature_std = checkpoint["feature_std"].to(resolved_device)
        listwise_normalize = bool(args.get("listwise_normalize", False))
        return cls(
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_keys=feature_keys,
            checkpoint_path=checkpoint_path,
            listwise_normalize=listwise_normalize,
        )

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        normalized = features.to(self.device)
        if self.listwise_normalize:
            normalized = apply_listwise_normalization(normalized)
        return (normalized - self.feature_mean) / self.feature_std

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(self.normalize(features))

    def score_assemblies(self, assemblies: Sequence[object]) -> torch.Tensor:
        features = candidate_feature_tensor(assemblies, device=self.device, feature_keys=self.feature_keys)
        return self.score_features(features)

    def select_assembly(self, assemblies: Sequence[object]) -> tuple[int, torch.Tensor]:
        logits = self.score_assemblies(assemblies)
        return int(logits.argmax().item()), logits


def resolve_candidate_selector_checkpoint(
    checkpoint_path: Path | str | None = None,
    allow_missing: bool = False,
) -> Path | None:
    candidate_paths: list[Path]
    if checkpoint_path is None:
        candidate_paths = [DEFAULT_STAGE3_CANDIDATE_SELECTOR_CHECKPOINT]
    else:
        raw_path = Path(checkpoint_path)
        candidate_paths = [raw_path]
        if not raw_path.is_absolute():
            candidate_paths.append(PROJECT_ROOT / raw_path)
    for path in candidate_paths:
        if path.exists():
            return path
    if allow_missing:
        return None
    tried = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Candidate selector checkpoint not found. Tried: {tried}")


def load_candidate_selector(
    checkpoint_path: Path | str | None = None,
    device: str | torch.device = "cpu",
    allow_missing: bool = False,
) -> LoadedCandidateSelector | None:
    resolved_path = resolve_candidate_selector_checkpoint(checkpoint_path, allow_missing=allow_missing)
    if resolved_path is None:
        return None
    return LoadedCandidateSelector.from_checkpoint(resolved_path, device=device)


__all__ = [
    "CandidateSelector",
    "DEFAULT_STAGE3_CANDIDATE_SELECTOR_CHECKPOINT",
    "DEFAULT_STAGE3_CANDIDATE_SELECTOR_DIR",
    "EXCLUDED_CANDIDATE_FEATURE_KEYS",
    "FEATURE_KEYS",
    "FEATURE_KEY_ALIASES",
    "LoadedCandidateSelector",
    "apply_listwise_normalization",
    "candidate_feature_tensor",
    "infer_candidate_feature_keys",
    "load_candidate_selector",
    "resolve_candidate_selector_checkpoint",
]
