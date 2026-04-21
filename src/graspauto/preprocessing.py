"""Preprocessing entrypoints for the graspauto codebase.

- `DEFAULT_M2AE_WEIGHTS` — canonical Point-M2AE pretrained-weights path.
- `precompute_object_m2ae_cache` — runs the Point-M2AE backbone over a set
  of object point clouds and saves per-object global, local, and
  patch-center features to `outputs/stage3_contact_graph/object_m2ae_cache.pt`.
  Only needed when a new dataset is added, the cache is deleted, or the
  backbone weights change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from graspauto.point_m2ae_encoder import PointM2AEObjectEncoder
from graspauto.utils import ensure_dir, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M2AE_WEIGHTS = PROJECT_ROOT / "external" / "pretrained" / "point_m2ae_pretrain.pth"


def precompute_object_m2ae_cache(
    *,
    object_points: torch.Tensor,
    object_names: List[str],
    out_path: str | Path,
    weights_path: str | Path,
    device: str | torch.device = "auto",
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Encode each object point cloud with Point-M2AE and save a per-object feature cache.

    The cache dict format (consumed by the Stage 3 training loader) is:

        {
            "object_names":  List[str]                  (length N),
            "m2ae_global":   Tensor (N, 1024)  float32  — adapter-projected global feature,
            "m2ae_local":    Tensor (N, 64, 384) float32 — 64 patch tokens per object,
            "patch_centers": Tensor (N, 64, 3) float32 — 3-D centers of the 64 patches,
            "weights_path":  str                        — record of the backbone weights used.
        }

    Args:
        object_points: Float tensor of shape (N_objects, N_points, 3). Point clouds
            are subsampled internally to 2048 points via FPS if longer.
        object_names: Per-object string identifier, length must match object_points.shape[0].
        out_path: Destination `.pt` file. Parent directories are created if missing.
        weights_path: Path to the pretrained Point-M2AE backbone checkpoint
            (e.g., `external/pretrained/point_m2ae.pth`).
        device: `"auto"` (default), `"cuda"`, `"cpu"`, or a torch.device.
        batch_size: Forward-pass batch size. Memory-bound on the backbone.

    Returns:
        The same dict that was saved to `out_path`.
    """
    if len(object_names) != object_points.shape[0]:
        raise ValueError(
            f"object_names length {len(object_names)} does not match "
            f"object_points.shape[0] = {object_points.shape[0]}"
        )

    dev = resolve_device(str(device)) if not isinstance(device, torch.device) else device

    encoder = PointM2AEObjectEncoder(
        output_dim=1024,
        pretrained_path=str(weights_path),
        freeze_backbone=True,
    ).to(dev).eval()

    object_points = object_points.to(dev).float()
    n_objects = object_points.shape[0]

    globals_list: List[torch.Tensor] = []
    locals_list: List[torch.Tensor] = []
    centers_list: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            end = min(start + batch_size, n_objects)
            batch = object_points[start:end]
            global_feat, local_tokens, patch_centers = encoder.forward_local(batch)
            globals_list.append(global_feat.detach().cpu())
            locals_list.append(local_tokens.detach().cpu())
            centers_list.append(patch_centers.detach().cpu())

    cache: Dict[str, Any] = {
        "object_names": list(object_names),
        "m2ae_global": torch.cat(globals_list, dim=0).contiguous(),
        "m2ae_local": torch.cat(locals_list, dim=0).contiguous(),
        "patch_centers": torch.cat(centers_list, dim=0).contiguous(),
        "weights_path": str(weights_path),
    }

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    torch.save(cache, out_path)

    return cache
