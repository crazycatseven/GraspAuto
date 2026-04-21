"""GraspXL dataset for graspauto training.

Loads the processed GraspXL shards (from preprocessing scripts)
and the Point-M2AE feature cache (from the m2ae cache builder).
Returns batch dicts compatible with train.py's training loop.

GraspXL grasps are in object space (no hTm transform), so the 54-D target
construction is simpler than ContactPose: rot6d = from global orient,
trans = transl + pivot compensation (with identity hTm_rot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


class GraspXLDataset(Dataset):
    """Mango_v1-compatible dataset backed by GraspXL processed shards.

    Loads all specified shards into memory at init. Each sample returns a
    dict with the same keys the graspauto training loop expects for the
    palm-only-intent + latent-ae path.
    """

    def __init__(
        self,
        shard_dir: str | Path = "data/graspxl_mango",
        m2ae_cache_path: str | Path = "data/graspxl_mango/m2ae_cache.pt",
        shard_ids: Optional[list[int]] = None,
        max_samples: Optional[int] = None,
        quality_filter_mm: Optional[float] = 5.0,
    ):
        shard_dir = Path(shard_dir)
        shard_files = sorted(shard_dir.glob("shard_*.pt"))
        if shard_ids is not None:
            shard_files = [shard_dir / f"shard_{i:04d}.pt" for i in shard_ids]

        # Load M2AE cache and build obj_id → index mapping
        print(f"[GraspXL] loading M2AE cache...", flush=True)
        cache = torch.load(m2ae_cache_path, map_location="cpu", weights_only=False)
        self._m2ae_global = cache["m2ae_global"]       # (N_obj, 1024)
        self._m2ae_local = cache["m2ae_local"]         # (N_obj, 64, 384)
        self._patch_centers = cache["patch_centers"]   # (N_obj, 64, 3)
        obj_id_list = cache["object_ids"]
        self._obj_id_to_idx = {oid: i for i, oid in enumerate(obj_id_list)}
        print(f"[GraspXL] {len(obj_id_list)} objects in M2AE cache", flush=True)

        # Load shards
        all_pose = []
        all_transl = []
        all_betas = []
        all_obj_id = []
        all_palm_centroid = []
        all_obj_pc = []

        for sf in shard_files:
            if not sf.exists():
                print(f"[GraspXL] SKIP missing shard: {sf}", flush=True)
                continue
            shard = torch.load(sf, map_location="cpu", weights_only=False)
            n = shard["pose_48"].shape[0]
            all_pose.append(shard["pose_48"])
            all_transl.append(shard["transl"])
            all_betas.append(shard["betas"])
            all_obj_id.extend(shard["obj_id"])
            all_palm_centroid.append(shard["palm_centroid"])
            all_obj_pc.append(shard["obj_pc"])
            del shard
            print(f"  {sf.name}: {n} samples", flush=True)

        self.pose_48 = torch.cat(all_pose, dim=0)
        self.transl = torch.cat(all_transl, dim=0)
        self.betas = torch.cat(all_betas, dim=0)
        self.obj_id = all_obj_id
        self.palm_centroid = torch.cat(all_palm_centroid, dim=0)
        self.obj_pc = torch.cat(all_obj_pc, dim=0)

        # Quality filter: drop samples where stored hand_verts are not actually touching
        # obj_pc. Root cause (discovered 2026-04-18): process_graspxl.py used
        # manopth.grabManoLayer for quality scoring but process_graspxl_for_mango.py
        # recomputed hand_verts via manotorch with a different th_trans convention,
        # producing ~9% samples where the hand is > 5mm from obj surface even though
        # stored min_dist_mm says <2mm. These "bad" samples teach the model to float.
        if quality_filter_mm is not None:
            qmask_path = shard_dir / "quality_mask_shards0to2.pt"
            if qmask_path.exists():
                qm = torch.load(qmask_path, map_location="cpu", weights_only=False)
                d = qm["distance_mm"]
                _n_current = len(self.pose_48)
                _n_mask = len(d)
                # Align lengths: use first min(len, mask_len) samples
                n = min(_n_current, _n_mask)
                keep = (d[:n] < quality_filter_mm)
                idx = torch.where(keep)[0]
                print(f"[GraspXL] quality filter @ {quality_filter_mm}mm: "
                      f"{len(idx)}/{n} kept ({(1-len(idx)/n)*100:.1f}% dropped)", flush=True)
                self.pose_48 = self.pose_48[:n][idx]
                self.transl = self.transl[:n][idx]
                self.betas = self.betas[:n][idx]
                self.obj_id = [self.obj_id[i] for i in idx.tolist()]
                self.palm_centroid = self.palm_centroid[:n][idx]
                self.obj_pc = self.obj_pc[:n][idx]
                self._quality_keep_idx = idx  # save for palm_features subsample
            else:
                print(f"[GraspXL] WARN: quality filter requested but {qmask_path} not found. "
                      f"Compute via build script; keeping all samples.", flush=True)
                self._quality_keep_idx = None
        else:
            self._quality_keep_idx = None

        if max_samples is not None and max_samples < len(self.pose_48):
            self.pose_48 = self.pose_48[:max_samples]
            self.transl = self.transl[:max_samples]
            self.betas = self.betas[:max_samples]
            self.obj_id = self.obj_id[:max_samples]
            self.palm_centroid = self.palm_centroid[:max_samples]
            self.obj_pc = self.obj_pc[:max_samples]

        # Resolve M2AE indices for each sample
        self._m2ae_idx = torch.tensor(
            [self._obj_id_to_idx[oid] for oid in self.obj_id], dtype=torch.long
        )

        # Dummy object_id as int (hash-based, for compatibility)
        self.object_id = self._m2ae_idx  # reuse as numeric object identifier

        self.num_codes = 128  # dummy, not used for latent flow training

        # MANO hands_mean offset — GraspXL processing stored raw axis-angle
        # (process_graspxl.py line 370-402 stores `rh['pose']` directly without
        # baking hands_mean). Our MangoMANODecoder uses `flat_hand_mean=True`,
        # which does NOT add hands_mean internally. ContactPose processing bakes
        # hands_mean into pose_48 during PCA expansion, and OakInk now bakes it
        # in its dataset class (2026-04-18 fix). GraspXL must do the same, or
        # each GraspXL sample has a ~25mm systematic offset between decoded
        # target x1 and stored hand_verts — the exact bug pattern that broke
        # r031. Confirmed 2026-04-18 in debugging session.
        try:
            from manotorch.manolayer import ManoLayer as _MANO
            _mano_root = str(Path(__file__).resolve().parent.parent.parent / "assets" / "mano_v1_2")
            _m = _MANO(rot_mode="axisang", side="right", center_idx=None, use_pca=False,
                       flat_hand_mean=False, mano_assets_root=_mano_root)
            self._hands_mean = _m.th_hands_mean[0].clone()  # (45,)
            print(f"[GraspXL] loaded hands_mean (norm={self._hands_mean.norm().item():.3f}) "
                  f"for flat=True MANO decoder compatibility", flush=True)
        except Exception as e:
            print(f"[GraspXL] WARN: could not load hands_mean ({e}); pose will be raw axis-angle", flush=True)
            self._hands_mean = torch.zeros(45)

        # Load precomputed palm features (normal/spread/entropy/mass) to match
        # CP's 11-D palm-token distribution. Without this, zeros flood 30% of
        # batches in 3-way mix and cripple CP quality (verified 2026-04-18:
        # r030 CP SEEN 11.85 → 36.65 mm when palm features zeroed at eval).
        # Prefer palm_features_v2.pt (contact-head-derived, semantic parity with CP)
        # over palm_features.pt (Gaussian). Same index alignment.
        self._palm_features = None
        for pf_name, tag in [("palm_features_v2.pt", "v2 (contact-head)"),
                             ("palm_features.pt", "v1 (Gaussian)")]:
            pf_path = shard_dir / pf_name
            if pf_path.exists():
                _pf = torch.load(pf_path, map_location="cpu", weights_only=False)
                total = {k: v for k, v in _pf.items() if k != "shard_boundaries"}
                if self._quality_keep_idx is not None:
                    total = {k: v[self._quality_keep_idx] for k, v in total.items()}
                _N = len(self.pose_48)
                self._palm_features = {k: v[:_N] for k, v in total.items()}
                print(f"[GraspXL] using palm_features {tag} from {pf_name} "
                      f"(palm_mass mean={self._palm_features['palm_mass'].mean().item():.2f})", flush=True)
                break
        else:
            print(f"[GraspXL] WARN: no palm_features.pt; palm_normal/spread/entropy will be zeros "
                  f"(causes 3-way-mix regression — run preprocessing scripts)", flush=True)

        print(f"[GraspXL] {len(self)} samples loaded", flush=True)

    def __len__(self) -> int:
        return self.pose_48.shape[0]

    def __getitem__(self, idx: int) -> dict:
        m2ae_i = self._m2ae_idx[idx]

        # Build dummy graph fields for palm-only mode.
        # Only palm_centroid is meaningful; others are zeros.
        palm_c = self.palm_centroid[idx]  # (3,)
        zeros3 = torch.zeros(3)
        zeros1 = torch.zeros(1).squeeze()

        # GraspXL has no hTm transform — grasps are in object frame directly.
        # Use identity for hTm_rot, transl for hTm_trans.
        hTm_rot = torch.eye(3, dtype=torch.float32)
        hTm_trans = self.transl[idx]

        # Bake hands_mean into hand_pose for flat=True MANO decoder compatibility.
        # Without this, training sees a ~25mm systematic offset between decoded
        # target x1 and stored hand_verts for every GraspXL sample.
        pose_48 = self.pose_48[idx].clone()
        pose_48[3:] = pose_48[3:] + self._hands_mean

        # Real palm contact features (normal/spread/entropy/mass) if available.
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

        return {
            # MANO params (for build_target_mano_params)
            "pose_48": pose_48,
            "hTm_rot": hTm_rot,
            "hTm_trans": hTm_trans,
            # Point-M2AE features
            "m2ae_global": self._m2ae_global[m2ae_i],
            "m2ae_local": self._m2ae_local[m2ae_i],
            "patch_centers": self._patch_centers[m2ae_i],
            # Object point cloud
            "object_points": self.obj_pc[idx],
            "object_normals": torch.zeros_like(self.obj_pc[idx]),  # dummy normals
            # Contact graph (palm-only: only centroid matters)
            "palm_centroid": palm_c,
            "palm_normal": palm_normal_v,
            "palm_spread": palm_spread_v,
            "palm_entropy": palm_entropy_v,
            "palm_mass": palm_mass_v,
            # Finger (dummy for palm-only mode)
            "finger_centroid": torch.zeros(5, 3),
            "finger_normal": torch.zeros(5, 3),
            "finger_spread": torch.zeros(5, 3),
            "finger_entropy": torch.zeros(5),
            "finger_mass": torch.zeros(5),
            # Unified (dummy)
            "unified_centroid": palm_c,
            "unified_normal": zeros3,
            "unified_spread": zeros3,
            "unified_entropy": zeros1,
            # Other
            "active_finger_score": torch.zeros(5),
            "stage1_contact_input": torch.zeros(3000),
            "unified_contact_target": torch.zeros(3000),
            "object_id": self.object_id[idx],
            "intent_id": torch.tensor(0, dtype=torch.long),
            "betas": self.betas[idx],
            "gt_world_verts": torch.zeros(778, 3),  # no GT verts for eval (pretrain only)
        }


__all__ = ["GraspXLDataset"]
