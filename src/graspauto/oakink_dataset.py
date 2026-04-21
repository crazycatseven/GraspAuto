"""OakInk v4 cache dataset for watermelon mixed training.

Uses `data/oakink/v4_cache/{train,val}.pt` (produced by
the data-prep script) + `data/oakink/v4_cache/m2ae_cache.pt`
(produced by the data-prep script).

Computes the contact-graph fields (palm_centroid, unified_centroid) on the
fly from the precomputed hand_verts + obj_pc since the v4 cache doesn't store
them. This matches the ContactPose + GraspXL convention so that the shared
adapter works unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset


# Palm vertex indices (same as training preprocessing).
PALM_VERT_IDS = torch.tensor(list(range(0, 22)) + [100, 101, 102, 103, 104, 218, 219, 220, 221, 222])


class OakInkDataset(Dataset):
    """OakInk v4 dataset. Returns the same dict keys as ContactPose/GraspXL
    datasets so the shared train loop works.

    Args:
        cache_path: Path to the train.pt / val.pt file.
        m2ae_cache_path: Path to m2ae_cache.pt.
        max_samples: Cap total samples (for testing / mix ratio control).
        quality_filter: Drop grasps with min_dist_mm > this (in meters). Default
            None = keep all.
    """

    def __init__(
        self,
        cache_path: str | Path,
        m2ae_cache_path: str | Path,
        max_samples: Optional[int] = None,
        quality_filter_mm: Optional[float] = None,
        palm_centroid_path: Optional[str | Path] = None,
        palm_features_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.m2ae_cache_path = Path(m2ae_cache_path)

        cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.pose = cache["pose"]                          # (N, 48) = root_rot(3) + hand_pose(45)
        self.transl = cache["transl"]                      # (N, 3)
        self.betas = cache["betas"]                        # (N, 10)
        self.hand_verts = cache["hand_verts"]              # (N, 778, 3)
        self.obj_pc = cache["obj_pc"]                      # (N, 3000, 3)
        self.obj_name_arr = cache["obj_name"]              # (N,) string
        self.min_dist_mm = cache["min_dist_mm"]            # (N,) float

        # Precomputed palm_centroid (matches CP/GraspXL semantics: σ=20mm palm-vert
        # weighted mean + surface projection). Fallback: auto-detect alongside cache.
        if palm_centroid_path is None:
            side = "train" if "train" in self.cache_path.stem else "val"
            palm_centroid_path = self.cache_path.parent / f"palm_centroids_{side}.pt"
        self.palm_centroid_path = Path(palm_centroid_path)
        if self.palm_centroid_path.exists():
            self._palm_centroid_precomputed = torch.load(
                self.palm_centroid_path, map_location="cpu", weights_only=False
            )["palm_centroid"]
            print(f"[OakInk] using precomputed palm_centroid from {self.palm_centroid_path.name}", flush=True)
        else:
            self._palm_centroid_precomputed = None
            print(f"[OakInk] WARNING: no precomputed palm_centroid at {self.palm_centroid_path}; "
                  "falling back to on-the-fly (σ=5mm, inconsistent with CP/GraspXL).", flush=True)

        # Precomputed palm contact features (normal/spread/entropy/mass) — 2026-04-18
        # fix. These match CP's 11-D palm token semantics so the adapter's palm_proj
        # sees a consistent distribution across CP / OakInk / GraspXL batches. Without
        # these, zeros would flood 60% of batches in 3-way mixing and cripple CP
        # quality (verified: r030 CP SEEN 11.85 → 36.65mm with palm features zeroed).
        # Prefer palm_features_v2.pt (contact-head-derived, semantic parity with CP).
        # Falls back to palm_features.pt (geometric Gaussian) if v2 missing.
        pf_v2 = self.cache_path.parent / "palm_features_v2.pt"
        pf_v1 = self.cache_path.parent / "palm_features.pt"
        self._palm_features = None
        side = "train" if "train" in self.cache_path.stem else "val"
        search_paths: list[tuple[Path, str]] = []
        if palm_features_path is not None:
            search_paths.append((Path(palm_features_path), "custom"))
        search_paths.extend([(pf_v2, "v2 (contact-head)"), (pf_v1, "v1 (Gaussian)")])
        for pf_path, tag in search_paths:
            if pf_path.exists():
                _pf = torch.load(pf_path, map_location="cpu", weights_only=False)
                if side in _pf:
                    self._palm_features = _pf[side]
                    print(f"[OakInk] using palm_features {tag}[{side}] from {pf_path.name} "
                          f"(palm_mass mean={self._palm_features['palm_mass'].mean().item():.2f})", flush=True)
                    break
        if self._palm_features is None:
            print(f"[OakInk] WARN: no palm_features.pt; palm_normal/spread/entropy will be zeros "
                  f"(causes 3-way-mix regression — run preprocessing scripts)", flush=True)

        # Quality filter
        keep_mask = torch.ones(len(self.pose), dtype=torch.bool)
        if quality_filter_mm is not None:
            keep_mask &= (self.min_dist_mm < quality_filter_mm)
        if keep_mask.sum() < len(self.pose):
            idx = torch.where(keep_mask)[0]
            print(f"[OakInk] quality filter: {len(idx)}/{len(self.pose)} grasps kept "
                  f"(min_dist < {quality_filter_mm}mm)", flush=True)
            self.pose = self.pose[idx]
            self.transl = self.transl[idx]
            self.betas = self.betas[idx]
            self.hand_verts = self.hand_verts[idx]
            self.obj_pc = self.obj_pc[idx]
            self.obj_name_arr = [self.obj_name_arr[i] for i in idx.tolist()]
            self.min_dist_mm = self.min_dist_mm[idx]
            if self._palm_centroid_precomputed is not None:
                self._palm_centroid_precomputed = self._palm_centroid_precomputed[idx]
            if self._palm_features is not None:
                self._palm_features = {k: v[idx] for k, v in self._palm_features.items()}

        if max_samples is not None and len(self.pose) > max_samples:
            # Deterministic subsample (not random, so training is reproducible)
            idx = torch.linspace(0, len(self.pose) - 1, max_samples).long()
            self.pose = self.pose[idx]
            self.transl = self.transl[idx]
            self.betas = self.betas[idx]
            self.hand_verts = self.hand_verts[idx]
            self.obj_pc = self.obj_pc[idx]
            self.obj_name_arr = [self.obj_name_arr[i] for i in idx.tolist()]
            self.min_dist_mm = self.min_dist_mm[idx]
            if self._palm_centroid_precomputed is not None:
                self._palm_centroid_precomputed = self._palm_centroid_precomputed[idx]
            if self._palm_features is not None:
                self._palm_features = {k: v[idx] for k, v in self._palm_features.items()}

        # Load M2AE cache (per-object, dedup by obj_name)
        m2ae = torch.load(self.m2ae_cache_path, map_location="cpu", weights_only=False)
        self._m2ae_object_names = list(m2ae["object_names"])
        self._m2ae_global = m2ae["m2ae_global"]      # (N_obj, 1024)
        self._m2ae_local = m2ae["m2ae_local"]        # (N_obj, 64, 384)
        self._patch_centers = m2ae["patch_centers"]  # (N_obj, 64, 3)

        # Map each sample's obj_name → row index in m2ae cache
        name_to_row = {n: i for i, n in enumerate(self._m2ae_object_names)}
        self._m2ae_idx = torch.tensor(
            [name_to_row[n] for n in self.obj_name_arr], dtype=torch.long
        )

        # Numeric object IDs for batch-side book-keeping
        unique_names = sorted(set(self.obj_name_arr))
        self._name_to_numeric = {n: i for i, n in enumerate(unique_names)}
        self.object_id = torch.tensor(
            [self._name_to_numeric[n] for n in self.obj_name_arr], dtype=torch.long
        )

        self.num_codes = 128  # dummy; not used for latent flow

        # MANO hands_mean offset — OakInk stores raw axis-angle (no hands_mean
        # baked in), but our decoder uses `flat_hand_mean=True` which does NOT
        # add hands_mean internally. ContactPose processing bakes hands_mean
        # into pose_48 during PCA expansion (process_contactpose.py line 187),
        # so its decode with flat=True is correct. OakInk must do the same.
        # Without this, each OakInk sample has a constant ~25mm MANO offset
        # (confirmed 2026-04-18 in debugging session).
        try:
            from manotorch.manolayer import ManoLayer as _MANO
            _mano_root = str(Path(__file__).resolve().parent.parent.parent / "assets" / "mano_v1_2")
            _m = _MANO(rot_mode="axisang", side="right", center_idx=None, use_pca=False,
                       flat_hand_mean=False, mano_assets_root=_mano_root)
            self._hands_mean = _m.th_hands_mean[0].clone()  # (45,)
            print(f"[OakInk] loaded hands_mean (norm={self._hands_mean.norm().item():.3f}) "
                  f"for flat=True MANO decoder compatibility", flush=True)
        except Exception as e:
            print(f"[OakInk] WARN: could not load hands_mean ({e}); pose will be raw axis-angle", flush=True)
            self._hands_mean = torch.zeros(45)

        print(f"[OakInk] {len(self)} samples loaded from {self.cache_path.name}", flush=True)

    def __len__(self) -> int:
        return self.pose.shape[0]

    def __getitem__(self, idx: int) -> dict:
        m2ae_i = self._m2ae_idx[idx]

        pose_48 = self.pose[idx].clone()         # (48,) = root(3) + hand_pose(45)
        # Bake hands_mean into hand_pose for flat=True MANO decoder compatibility.
        # Without this, model training sees a ~25mm systematic offset between
        # decoded target x1 and stored hand_verts. See __init__ comment above.
        pose_48[3:] = pose_48[3:] + self._hands_mean
        transl = self.transl[idx]                # (3,)
        hand_verts = self.hand_verts[idx]        # (778, 3)
        obj_pc = self.obj_pc[idx]                # (3000, 3)

        # palm_centroid: prefer precomputed (matches CP/GraspXL σ=20mm palm-vert
        # mean + surface projection). Fallback to on-the-fly computation with the
        # SAME recipe (kept consistent with `compute_palm_centroid_contact`).
        if self._palm_centroid_precomputed is not None:
            palm_centroid = self._palm_centroid_precomputed[idx]
        else:
            palm_verts = hand_verts[PALM_VERT_IDS]   # (32, 3)
            dists = torch.cdist(palm_verts.unsqueeze(0), obj_pc.unsqueeze(0)).squeeze(0).min(dim=-1).values
            w = torch.exp(-dists.pow(2) / (2 * 0.02 ** 2))  # σ=20mm, matches CP/GraspXL
            w = (w + 1e-6)
            w = w / w.sum().clamp_min(1e-6)
            centroid = (palm_verts * w.unsqueeze(-1)).sum(dim=0)
            obj_dists = torch.cdist(centroid.unsqueeze(0), obj_pc.unsqueeze(0)).squeeze()
            palm_centroid = obj_pc[obj_dists.argmin()]

        # Unified centroid: same as palm_centroid for simplicity (we can use
        # full MANO verts later if needed).
        unified_centroid = palm_centroid.clone()

        zeros3 = torch.zeros(3)
        zeros1 = torch.zeros(1).squeeze()

        # Pull real palm contact features if precomputed, else fallback zeros.
        if self._palm_features is not None:
            palm_normal_v = self._palm_features["palm_normal"][idx]
            palm_spread_v = self._palm_features["palm_spread"][idx]
            palm_entropy_v = self._palm_features["palm_entropy"][idx]
            palm_mass_v = self._palm_features["palm_mass"][idx]
        else:
            palm_normal_v = zeros3
            palm_spread_v = zeros3
            palm_entropy_v = zeros1
            palm_mass_v = zeros1 + 1.0

        # hTm: OakInk grasps are in world frame. Use identity rot, transl as trans.
        hTm_rot = torch.eye(3, dtype=torch.float32)
        hTm_trans = transl

        return {
            # MANO params
            "pose_48": pose_48,
            "hTm_rot": hTm_rot,
            "hTm_trans": hTm_trans,
            "betas": self.betas[idx],

            # Point-M2AE features (per-object, looked up via m2ae_idx)
            "m2ae_global": self._m2ae_global[m2ae_i],
            "m2ae_local": self._m2ae_local[m2ae_i],
            "patch_centers": self._patch_centers[m2ae_i],

            # Object
            "object_points": obj_pc,
            "object_normals": torch.zeros_like(obj_pc),  # no normals in OakInk cache

            # Contact graph — palm_centroid computed on the fly, others zeroed
            "palm_centroid": palm_centroid,
            "palm_normal": palm_normal_v,
            "palm_spread": palm_spread_v,
            "palm_entropy": palm_entropy_v,
            "palm_mass": palm_mass_v,
            "finger_centroid": torch.zeros(5, 3),
            "finger_normal": torch.zeros(5, 3),
            "finger_spread": torch.zeros(5, 3),
            "finger_entropy": torch.zeros(5),
            "finger_mass": torch.zeros(5),
            "unified_centroid": unified_centroid,
            "unified_normal": zeros3,
            "unified_spread": zeros3,
            "unified_entropy": zeros1,

            # Other
            "active_finger_score": torch.zeros(5),
            "stage1_contact_input": torch.zeros(3000),
            "unified_contact_target": torch.zeros(3000),
            "object_id": self.object_id[idx],
            "intent_id": torch.tensor(0, dtype=torch.long),
            "gt_world_verts": hand_verts,  # OakInk does have GT verts, use them
        }


__all__ = ["OakInkDataset"]
