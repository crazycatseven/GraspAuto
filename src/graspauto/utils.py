"""Generic utilities for the graspauto codebase (device resolution, seed
setup, JSON I/O, default paths for the contact-graph cache)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_CACHE = (
    PROJECT_ROOT / "outputs" / "contact_vqvae_stage1_v16_film" / "cache" / "geometry_cache.pt"
)
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device, honoring 'auto' for CUDA detection."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch_to_device(
    batch: Dict[str, Any],
    device: str | torch.device,
) -> Dict[str, Any]:
    """Move all torch.Tensor values in a dict batch to the target device, leaving others untouched."""
    dev = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev)
        else:
            out[k] = v
    return out


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) at `path` if missing. Returns the Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write `payload` as pretty-printed JSON to `path`, creating parent dirs as needed.

    Uses `default=str` so non-JSON-native values (PosixPath, numpy scalars,
    torch dtypes, etc.) get coerced to their string representation instead
    of raising TypeError. Callers that already coerce to primitives will be
    unaffected.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU and CUDA) from a single integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
