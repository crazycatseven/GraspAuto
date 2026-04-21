#!/usr/bin/env python3
"""Compute a 7-D grip-sphere palm token (center 3 + radius 1 + approach 3)
from GT MANO hand verts and pack it into the existing 11-D palm fields so
the adapter can be warm-started from `graspauto_distill_student` with
minimal code change.

Packing layout into the existing 11-D slot:
    palm_centroid (3) = sphere_center
    palm_normal   (3) = approach_direction (unit)
    palm_spread   (3) = [radius, 0, 0]
    palm_entropy  (1) = 0
    palm_mass     (1) = 0

The velocity net sees a 7-D informative + 4-D zero signal after palm_proj
reinitialisation. 4 zeros cost ~nothing; the full 11-D API is preserved.

Reads:
  - outputs/stage3_contact_graph/{train,val}.pt
  - (optional) data/oakink/v4_cache/{train,val}.pt

Writes:
  - outputs/stage3_contact_graph/{train,val}_sphere.pt
  - (optional) data/oakink/v4_cache/{train,val}_sphere.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graspauto.geom_palm_features import grip_sphere_from_mano


def _pack_sphere_into_11d(sphere_7d: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split 7-D sphere into the existing 11-D palm field layout."""
    assert sphere_7d.shape[-1] == 7
    center = sphere_7d[..., 0:3]
    radius = sphere_7d[..., 3:4]                      # (..., 1)
    approach = sphere_7d[..., 4:7]
    zeros_1 = torch.zeros_like(radius)
    zeros_2 = torch.zeros_like(radius).squeeze(-1)
    spread = torch.cat([radius, zeros_1, zeros_1], dim=-1)  # (..., 3) with radius in dim 0
    return {
        "palm_centroid": center,
        "palm_normal":   approach,
        "palm_spread":   spread,
        "palm_entropy":  zeros_2,
        "palm_mass":     zeros_2,
    }


def preprocess_cp(cp_pt: Path, out_pt: Path) -> None:
    print(f"[cp] loading {cp_pt}")
    d = torch.load(cp_pt, map_location="cpu", weights_only=False)
    hand_verts = d["gt_world_verts"]                  # (N, 778, 3) object-frame
    print(f"[cp] {hand_verts.shape[0]} samples")
    sphere = grip_sphere_from_mano(hand_verts)        # (N, 7)
    packed = _pack_sphere_into_11d(sphere)
    d.update(packed)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out_pt)
    r = sphere[:, 3]
    print(f"[cp] radius mean={r.mean()*1000:.1f}mm std={r.std()*1000:.1f}mm "
          f"range=[{r.min()*1000:.1f},{r.max()*1000:.1f}]mm")
    print(f"[cp] saved {out_pt} ({out_pt.stat().st_size/1024/1024:.1f} MB)")


def preprocess_oakink(cache_pt: Path, out_pt: Path) -> None:
    if not cache_pt.exists():
        print(f"[oakink] SKIP: {cache_pt} not found")
        return
    print(f"[oakink] loading {cache_pt}")
    d = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if "hand_verts" not in d:
        print(f"[oakink] SKIP: no 'hand_verts' in {cache_pt}")
        return
    hand_verts = d["hand_verts"]                      # (N, 778, 3) already object-frame
    N = hand_verts.shape[0]
    print(f"[oakink] {N} samples")
    sphere = grip_sphere_from_mano(hand_verts)
    packed = _pack_sphere_into_11d(sphere)
    d.update(packed)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out_pt)
    # Also sidecar palm_features file for the OakInk loader
    side = "train" if "train" in out_pt.name else "val"
    pf_out = out_pt.parent / f"palm_features_{side}_sphere.pt"
    torch.save({side: packed}, pf_out)
    pc_out = out_pt.parent / f"palm_centroids_{side}_sphere.pt"
    torch.save({"palm_centroid": packed["palm_centroid"]}, pc_out)
    r = sphere[:, 3]
    print(f"[oakink] radius mean={r.mean()*1000:.1f}mm std={r.std()*1000:.1f}mm "
          f"range=[{r.min()*1000:.1f},{r.max()*1000:.1f}]mm")
    print(f"[oakink] saved {out_pt}, {pc_out}, {pf_out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-train", type=Path, default=Path("outputs/stage3_contact_graph/train.pt"))
    ap.add_argument("--cp-val",   type=Path, default=Path("outputs/stage3_contact_graph/val.pt"))
    ap.add_argument("--cp-out-train", type=Path, default=Path("outputs/stage3_contact_graph/train_sphere.pt"))
    ap.add_argument("--cp-out-val",   type=Path, default=Path("outputs/stage3_contact_graph/val_sphere.pt"))
    ap.add_argument("--oakink-train", type=Path, default=Path("data/oakink/v4_cache/train.pt"))
    ap.add_argument("--oakink-val",   type=Path, default=Path("data/oakink/v4_cache/val.pt"))
    ap.add_argument("--oakink-out-train", type=Path, default=Path("data/oakink/v4_cache/train_sphere.pt"))
    ap.add_argument("--oakink-out-val",   type=Path, default=Path("data/oakink/v4_cache/val_sphere.pt"))
    ap.add_argument("--skip-oakink", action="store_true")
    args = ap.parse_args()

    preprocess_cp(args.cp_train, args.cp_out_train)
    preprocess_cp(args.cp_val,   args.cp_out_val)
    if not args.skip_oakink:
        preprocess_oakink(args.oakink_train, args.oakink_out_train)
        preprocess_oakink(args.oakink_val,   args.oakink_out_val)
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
