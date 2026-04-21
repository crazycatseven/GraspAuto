"""Utilities for LoST v6 MANO parameter normalization statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# MANO v1.2 standard parameter dimensions, previously imported from the
# deleted `vrgrasp_v5.mano` namespace. Inlined on 2026-04-11 because the
# MANO spec fixes these values — they cannot drift from upstream.
BETAS_DIM = 10
GLOBAL_ORIENT_DIM = 3
HAND_POSE_DIM = 45

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANO_STATS_PATH = PROJECT_ROOT / "outputs" / "v6_2" / "lost" / "stats" / "train_mano_params.pt"


def _default_data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return PROJECT_ROOT / "data" / "unified" / "npy"


def _extract_mano_decoder_params(hand_param: np.ndarray) -> np.ndarray:
    betas = hand_param[:, :BETAS_DIM]
    hand_pose_start = BETAS_DIM + GLOBAL_ORIENT_DIM
    hand_pose_stop = hand_pose_start + HAND_POSE_DIM
    hand_pose = hand_param[:, hand_pose_start:hand_pose_stop]
    return np.concatenate([betas, hand_pose], axis=1)


def compute_mano_param_stats(
    *,
    split: str = "train",
    data_root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _default_data_root(data_root)
    hand_param = np.load(resolved_root / f"hand_param_{split}.npy", mmap_mode="r")
    mano_params = _extract_mano_decoder_params(np.asarray(hand_param, dtype=np.float32))
    mean = torch.from_numpy(mano_params.mean(axis=0, dtype=np.float64)).float()
    std = torch.from_numpy(mano_params.std(axis=0, dtype=np.float64)).float().clamp_min(1e-6)
    return {
        "mean": mean,
        "std": std,
        "split": split,
        "num_samples": int(mano_params.shape[0]),
        "data_root": str(resolved_root),
    }


def load_or_compute_mano_param_stats(
    stats_path: str | Path = DEFAULT_MANO_STATS_PATH,
    *,
    split: str = "train",
    data_root: Path | None = None,
    recompute: bool = False,
) -> dict[str, Any]:
    resolved_path = Path(stats_path)
    if resolved_path.exists() and not recompute:
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
        payload["mean"] = payload["mean"].float()
        payload["std"] = payload["std"].float().clamp_min(1e-6)
        return payload

    payload = compute_mano_param_stats(split=split, data_root=data_root)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved_path)
    return payload


def save_mano_param_stats(
    stats_path: str | Path,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    split: str = "train",
    num_samples: int = 0,
    data_root: str | Path | None = None,
) -> Path:
    resolved_path = Path(stats_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.detach().cpu().float(),
            "std": std.detach().cpu().float().clamp_min(1e-6),
            "split": split,
            "num_samples": int(num_samples),
            "data_root": str(_default_data_root(Path(data_root) if data_root is not None else None)),
        },
        resolved_path,
    )
    return resolved_path


__all__ = [
    "DEFAULT_MANO_STATS_PATH",
    "compute_mano_param_stats",
    "load_or_compute_mano_param_stats",
    "save_mano_param_stats",
]
