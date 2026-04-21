"""LongHandleAugDataset — loads procedurally-augmented long-handle samples
and returns them in the same schema as Stage3ContactGraphDataset.__getitem__.

The per-sample dict intentionally inlines `object_points`, `m2ae_local`, and
`patch_centers` (rather than looking them up in the shared per-object cache)
because each synthetic object has a unique geometry that must travel with the
sample.

Produced by the data-prep script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset


class LongHandleAugDataset(Dataset):
    def __init__(self, pt_path: str | Path):
        d = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        self.d = d
        self.N = len(d["pose_48"])
        # Stash the num_codes for the trainer (it reads this from the dataset)
        self.num_codes = int(d.get("config", {}).get("num_codes", 128))
        # object_id for synthetic samples is in the 10000+ range; trainer uses
        # this only for holdout filtering (which we don't apply here).
        self.object_id = d["object_id"]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int) -> Dict[str, Any]:
        d = self.d
        out: Dict[str, Any] = {
            # MANO pose targets
            "pose_48": d["pose_48"][i],
            "betas": d["betas"][i],
            "hTm_rot": d["hTm_rot"][i],
            "hTm_trans": d["hTm_trans"][i],
            "gt_local_verts": d["gt_local_verts"][i],
            "gt_world_verts": d["gt_world_verts"][i],
            # Stage1 supervisions (copied from parent — the model uses these
            # only as conditioning proxies, and the parent grasp's supervisions
            # are valid for the cylinder because the hand pose is identical).
            "stage1_contact_input": d["stage1_contact_input"][i],
            "unified_contact_target": d["unified_contact_target"][i],
            "finger_contact_target": d["finger_contact_target"][i],
            "palm_contact_target": d["palm_contact_target"][i],
            "active_finger_mask": d["active_finger_mask"][i],
            "active_finger_score": d["active_finger_score"][i],
            "gt_code_id": d["gt_code_id"][i],
            # Contact-graph finger features (copied from parent — same hand).
            "finger_centroid": d["finger_centroid"][i],
            "finger_normal": d["finger_normal"][i],
            "finger_spread": d["finger_spread"][i],
            "finger_entropy": d["finger_entropy"][i],
            "finger_mass": d["finger_mass"][i],
            # Palm 11-D semantic token — copied from parent (key design choice,
            # see long_handle_augment_v2.py header).
            "palm_centroid": d["palm_centroid"][i],
            "palm_normal": d["palm_normal"][i],
            "palm_spread": d["palm_spread"][i],
            "palm_entropy": d["palm_entropy"][i],
            "palm_mass": d["palm_mass"][i],
            "unified_centroid": d["unified_centroid"][i],
            "unified_normal": d["unified_normal"][i],
            "unified_spread": d["unified_spread"][i],
            "unified_entropy": d["unified_entropy"][i],
            "unified_cov": d["unified_cov"][i],
            "object_id": d["object_id"][i],
            # Inline per-sample object features.
            "object_points": d["syn_object_points"][i],
            "m2ae_local": d["syn_m2ae_local"][i],
            "patch_centers": d["syn_patch_centers"][i],
            # Diagnostics / auxiliary-loss inputs.
            "syn_length": torch.tensor(d["syn_length"][i], dtype=torch.float32),
            "syn_radius": torch.tensor(d["syn_radius"][i], dtype=torch.float32),
            "is_synthetic": torch.tensor(1, dtype=torch.long),
        }
        return out
