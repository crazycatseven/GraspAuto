#!/usr/bin/env python3
"""Diversity-preserving pool distillation v2.

Improves on train_distill_student.py by sampling teacher targets from the
top-M candidates in the 6-ckpt × K=32 pool (192 total per sample) via a
softmin weighting over mode-cover error, instead of always using the
single oracle pick. This exposes the student to multiple valid grasp modes
per tap so it can cover the ensemble's multi-modal distribution rather
than collapsing to a single pseudo-label.

Per-candidate pools are at `outputs/graspauto_selector_data_pool_rXXX/{train,val}.pt`.
Each has: raw_x1 (N*K, 54), targets_mm (N*K,), sample_index (N*K,),
candidate_index (N*K,), num_samples (N), num_candidates (K).

Sampling recipe (per GPT-5 Pro 2026-04-19 audit):
    q(m) ∝ exp(-(e_m - e_1) / τ), m ∈ top-M candidates
    τ = 1-2 mm, M = 8 or 16
    hard-gate boost: r024/r025 candidates weighted ×1.5 for UNSEEN balance

Success threshold: bo-64 oracle SEEN < 9.8 mm OR UNSEEN < 20 mm.

Run:
    .venv/bin/python train/train_distill_student_sphere.py \\
        --epochs 150 --lr 2e-5 --top-m 16 --tau 1.5 \\
        --out-dir outputs/graspauto_sphere_distill_student
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from graspauto.stage3_contact_graph import (
    DEFAULT_OBJECT_M2AE_CACHE,
    DEFAULT_STAGE3_CONTACT_GRAPH_ROOT,
    PointM2AEContactGraphModel,
    Stage3ContactGraphDataset,
)
from graspauto.utils import DEFAULT_GEOMETRY_CACHE, resolve_device
from graspauto.conditioning import ContactGraphConditioningAdapter, compute_patch_weight_from_point
from graspauto.flow_matching import ConditionalFlowMatching
from graspauto.mano_decoder import MangoMANODecoder, MANO_PARAM_DIM
from graspauto.velocity_network import VelocityNetwork
from graspauto.mano_autoencoder import ResidualMANOAutoEncoder, MANOAutoEncoder


CKPT_NAMES = ["r042", "r043", "r044", "r045", "r046", "r047", "r048"]
HARD_GATE_CKPTS = {"r047", "r048"}  # sphere diversity boost — OakInk-filtered + real-MoCap
HOLDOUTS = {3, 11, 20, 25}


def load_pool(split: str):
    """Load 6-ckpt pool and merge by sample_index.

    Returns dict: sample_index -> list of (raw_x1 (54,), targets_mm float, ckpt_name).
    Each sample has up to 6*32 = 192 candidates.
    """
    if split == "train":
        # Train pools are under "selector_data_pool_rXXX/train.pt"
        pool_files = [(n, Path(f"outputs/graspauto_selector_data_pool_sphere_{n}/train.pt")) for n in CKPT_NAMES]
    else:
        pool_files = [(n, Path(f"outputs/graspauto_selector_data_pool_sphere_{n}/val.pt")) for n in CKPT_NAMES]

    per_sample = {}
    for name, p in pool_files:
        if not p.exists():
            print(f"[pool] SKIP missing {p}")
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        raw_x1 = d["raw_x1"]         # (N*K, 54)
        targets = d["targets_mm"]    # (N*K,)
        sidx = d["sample_index"]     # (N*K,)
        for i in range(len(raw_x1)):
            si = int(sidx[i].item())
            per_sample.setdefault(si, []).append({
                "raw_x1": raw_x1[i],
                "error": float(targets[i].item()),
                "ckpt": name,
            })
    return per_sample


class PoolDistillDataset(Dataset):
    """Returns per-sample: (ContactPose fields..., list of candidates)."""

    def __init__(self, split: str, pool: dict, use_unified_intent: bool = True,
                 filter_holdouts: bool = True):
        stage3_split = "train_sphere" if split == "train" else "val_sphere"
        self.ds = Stage3ContactGraphDataset(
            split_path=DEFAULT_STAGE3_CONTACT_GRAPH_ROOT / f"{stage3_split}.pt",
            geometry_path=DEFAULT_GEOMETRY_CACHE,
            object_m2ae_cache_path=DEFAULT_OBJECT_M2AE_CACHE,
            limit=None,
        )
        self.pool = pool
        self.use_unified_intent = use_unified_intent
        # Filter: must have ≥6 candidates in pool AND be non-holdout (for train)
        self.si_list = []
        for si in sorted(pool.keys()):
            if len(pool[si]) < 6:
                continue
            if filter_holdouts and split == "train":
                oid = int(self.ds.object_id[si].item())
                if oid in HOLDOUTS:
                    continue
            self.si_list.append(si)
        print(f"[pool/{split}] {len(self.si_list)} samples with >=6 candidates")

    def __len__(self):
        return len(self.si_list)

    def __getitem__(self, idx):
        si = self.si_list[idx]
        s = self.ds[si]
        if self.use_unified_intent:
            s["palm_centroid"] = s["unified_centroid"].clone()
        # Return full candidate list; training loop does softmin sampling.
        cands = self.pool[si]
        s["cand_raw_x1"] = torch.stack([c["raw_x1"] for c in cands], dim=0)  # (K, 54)
        s["cand_error"] = torch.tensor([c["error"] for c in cands], dtype=torch.float32)  # (K,)
        s["cand_is_hard_gate"] = torch.tensor(
            [c["ckpt"] in HARD_GATE_CKPTS for c in cands], dtype=torch.bool
        )
        return s


def collate_pool(batch_list):
    """Custom collate: stack fields, pad candidates to max K in batch."""
    import torch as T
    B = len(batch_list)
    max_K = max(b["cand_raw_x1"].shape[0] for b in batch_list)
    out = {}
    # Stack simple tensors
    for k in batch_list[0]:
        if k == "cand_raw_x1":
            padded = T.zeros(B, max_K, 54)
            mask = T.zeros(B, max_K, dtype=T.bool)
            for i, b in enumerate(batch_list):
                K = b[k].shape[0]
                padded[i, :K] = b[k]
                mask[i, :K] = True
            out["cand_raw_x1"] = padded
            out["cand_mask"] = mask
        elif k == "cand_error":
            padded = T.full((B, max_K), 1e6)  # large error for padding
            for i, b in enumerate(batch_list):
                K = b[k].shape[0]
                padded[i, :K] = b[k]
            out["cand_error"] = padded
        elif k == "cand_is_hard_gate":
            padded = T.zeros(B, max_K, dtype=T.bool)
            for i, b in enumerate(batch_list):
                K = b[k].shape[0]
                padded[i, :K] = b[k]
            out["cand_is_hard_gate"] = padded
        elif torch.is_tensor(batch_list[0][k]):
            out[k] = torch.stack([b[k] for b in batch_list])
        else:
            out[k] = [b[k] for b in batch_list]
    return out


def sample_teacher_from_topm(cand_x1: torch.Tensor, cand_err: torch.Tensor,
                              cand_hard_gate: torch.Tensor, cand_mask: torch.Tensor,
                              top_m: int, tau: float, hard_gate_boost: float = 1.5):
    """For each batch element, pick top-M by error, then sample one via softmin.

    Args:
        cand_x1: (B, K, 54)
        cand_err: (B, K) in mm
        cand_hard_gate: (B, K) bool — r024/r025 membership
        cand_mask: (B, K) bool — real (non-padding) candidates
        top_m: int
        tau: temperature in mm

    Returns:
        (B, 54) sampled teacher target
    """
    B, K, _ = cand_x1.shape
    # Mask out padding by setting error to +inf
    err_valid = cand_err.masked_fill(~cand_mask, float('inf'))

    # Apply hard-gate boost: reduce effective error of r024/r025 candidates
    # so they have higher probability in softmin. Equivalent to multiplying
    # their softmin weight by hard_gate_boost. err_adj = err - τ*log(boost)
    err_adj = err_valid.clone()
    err_adj[cand_hard_gate] -= tau * torch.log(torch.tensor(hard_gate_boost, device=err_adj.device))

    # Top-M indices per sample
    m = min(top_m, K)
    top_err, top_idx = torch.topk(-err_adj, m, dim=1)  # (B, m), negate for smallest
    top_err = -top_err  # actual errors

    # Softmin weights: q(m) ∝ exp(-(e_m - e_min) / τ)
    e_min = top_err.min(dim=1, keepdim=True).values
    logits = -(top_err - e_min) / tau  # (B, m)
    probs = F.softmax(logits, dim=1)  # (B, m)

    # Sample one index per batch
    sampled_col = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
    chosen_idx = top_idx.gather(1, sampled_col.unsqueeze(-1)).squeeze(-1)  # (B,)

    # Gather raw_x1
    chosen_x1 = cand_x1.gather(1, chosen_idx.view(B, 1, 1).expand(B, 1, 54)).squeeze(1)
    return chosen_x1, chosen_idx, top_err.gather(1, sampled_col.unsqueeze(-1)).squeeze(-1)


def load_ae(device):
    ae_path = Path("outputs/graspauto_ae_joint/best.pt")
    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=False)
    if ae_ckpt.get("residual", False):
        ae = ResidualMANOAutoEncoder(
            input_dim=54, latent_dim=ae_ckpt["latent_dim"],
            hidden_dim=ae_ckpt["res_hidden_dim"], n_blocks=ae_ckpt["res_n_blocks"],
        ).to(device)
    else:
        ae = MANOAutoEncoder(input_dim=54, latent_dim=ae_ckpt["latent_dim"],
                             hidden_dims=ae_ckpt["hidden_dims"]).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ae_ckpt["train_mean"].to(device), ae_ckpt["train_std"].to(device), ae_ckpt["latent_dim"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--top-m", type=int, default=16, help="sample teacher from this many best candidates")
    p.add_argument("--tau", type=float, default=1.5, help="softmin temperature in mm")
    p.add_argument("--hard-gate-boost", type=float, default=1.5,
                   help="multiplier on softmin weight for r024/r025 candidates")
    p.add_argument("--warm-start-from", type=Path,
                   default=Path("outputs/graspauto_distill_student/best.pt"),
                   help="default: continue from the existing single-oracle student")
    p.add_argument("--warm-start-pear-v1", type=Path,
                   default=Path("outputs/stage3_metric_reset_short1/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/graspauto_sphere_distill_student"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    args = p.parse_args()

    device = resolve_device("auto")
    torch.manual_seed(args.seed)

    train_pool = load_pool("train")
    val_pool = load_pool("val")
    train_ds = PoolDistillDataset("train", train_pool)
    val_seen_ds = PoolDistillDataset("seen", val_pool, filter_holdouts=False)
    val_unseen_ds = PoolDistillDataset("unseen", val_pool, filter_holdouts=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, collate_fn=collate_pool)

    ae, latent_mean, latent_std, latent_dim = load_ae(device)

    # Student
    adapter = ContactGraphConditioningAdapter(
        hidden_dim=args.hidden_dim, use_intent_token=False,
        part_aware_gating=True, palm_only_intent=True,
        advanced_gating=False, residual_modulation=False,
    ).to(device)
    velocity_net = VelocityNetwork(
        input_dim=latent_dim, hidden_dim=args.hidden_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)

    # Warm-start from existing distill student (or r011 if not available)
    if args.warm_start_from.exists():
        ws = torch.load(args.warm_start_from, map_location=device, weights_only=False)
        adapter_state = dict(ws.get("adapter_state", {}))
        if "type_embed.weight" in adapter_state:
            saved = adapter_state["type_embed.weight"]
            expected = adapter.type_embed.weight.shape
            if saved.shape[0] < expected[0]:
                pad = torch.zeros(expected[0] - saved.shape[0], saved.shape[1],
                                  dtype=saved.dtype, device=saved.device)
                adapter_state["type_embed.weight"] = torch.cat([saved, pad], dim=0)
        missing_a, unexp_a = adapter.load_state_dict(adapter_state, strict=False)
        missing_v, unexp_v = velocity_net.load_state_dict(ws.get("velocity_net_state", {}), strict=False)
        print(f"[warm-start] {args.warm_start_from.name}: adapter {len(missing_a)} missing, velocity {len(missing_v)} missing")

    mano_decoder = MangoMANODecoder().to(device).eval()
    for pp in mano_decoder.parameters():
        pp.requires_grad_(False)

    flow = ConditionalFlowMatching()

    params = list(adapter.parameters()) + list(velocity_net.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def build_bundle(batch):
        graph = {k: batch[k].to(device) for k in [
            "finger_centroid","finger_normal","finger_spread","finger_entropy","finger_mass",
            "palm_centroid","palm_normal","palm_spread","palm_entropy","palm_mass",
            "unified_centroid","unified_normal","unified_spread","unified_entropy",
        ]}
        active_finger_prob = batch["active_finger_score"].to(device)
        m2ae_local = batch["m2ae_local"].to(device)
        patch_centers = batch["patch_centers"].to(device)
        pcw = compute_patch_weight_from_point(patch_centers, batch["palm_centroid"].to(device))
        return adapter(
            m2ae_local=m2ae_local, graph=graph,
            active_finger_prob=active_finger_prob,
            patch_contact_weight=pcw,
        )

    def forward_train(batch):
        B = batch["object_points"].shape[0]
        # Sample teacher from top-M candidates via softmin
        cand_x1 = batch["cand_raw_x1"].to(device)
        cand_err = batch["cand_error"].to(device)
        cand_hg = batch["cand_is_hard_gate"].to(device)
        cand_mask = batch["cand_mask"].to(device)
        teacher_x1, _, teacher_err = sample_teacher_from_topm(
            cand_x1, cand_err, cand_hg, cand_mask,
            top_m=args.top_m, tau=args.tau, hard_gate_boost=args.hard_gate_boost,
        )
        with torch.no_grad():
            mano_normed = (teacher_x1 - latent_mean.expand_as(teacher_x1)) / latent_std.expand_as(teacher_x1)
            target_z = ae.encode(mano_normed)

        bundle = build_bundle(batch)
        x0 = torch.randn_like(target_z)
        t = torch.rand(B, device=device)
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * target_z
        v_target = target_z - x0
        v_pred = velocity_net(xt, t, condition=bundle)
        loss = F.mse_loss(v_pred, v_target)
        return loss, teacher_err.mean().item()

    def sample_from_student(batch):
        B = batch["object_points"].shape[0]
        bundle = build_bundle(batch)
        x0 = torch.randn(B, latent_dim, device=device)
        z = flow.sample(velocity_net, x0, condition=bundle, num_steps=args.num_flow_steps, method="euler")
        mano_normed = ae.decode(z)
        x1 = mano_normed * latent_std + latent_mean
        verts = mano_decoder(x1)["vertices"]
        return verts

    def eval_split(ds, name):
        if len(ds) == 0:
            return 0.0, 0
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_pool)
        errors = []
        import numpy as np
        for batch in loader:
            for k in list(batch.keys()):
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            obj_ids = batch["object_id"]
            pred_verts = sample_from_student(batch)
            gt_verts = batch["gt_world_verts"]
            err_per = (pred_verts - gt_verts).norm(dim=-1).mean(dim=-1) * 1000.0
            for i in range(len(obj_ids)):
                oid = int(obj_ids[i].item())
                if name == "seen" and oid in HOLDOUTS:
                    continue
                if name == "unseen" and oid not in HOLDOUTS:
                    continue
                errors.append(err_per[i].item())
        return float(np.mean(errors)) if errors else 0.0, len(errors)

    best_seen = float("inf")
    history = []
    for ep in range(1, args.epochs + 1):
        adapter.train()
        velocity_net.train()
        t0 = time.time()
        loss_sum = 0.0
        teacher_err_sum = 0.0
        nb = 0
        for batch in train_loader:
            loss, teacher_err_batch = forward_train(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            loss_sum += loss.item()
            teacher_err_sum += teacher_err_batch
            nb += 1
        loss_avg = loss_sum / nb
        teacher_err_avg = teacher_err_sum / nb

        if ep % args.eval_every == 0 or ep == args.epochs:
            adapter.eval()
            velocity_net.eval()
            with torch.no_grad():
                seen_mean, seen_n = eval_split(val_seen_ds, "seen")
                unseen_mean, unseen_n = eval_split(val_unseen_ds, "unseen")
            dt = time.time() - t0
            history.append({"epoch": ep, "loss": loss_avg, "teacher_err_mm": teacher_err_avg,
                            "seen_mean": seen_mean, "unseen_mean": unseen_mean})
            print(f"[{ep:03d}] loss={loss_avg:.4f} teacher_err={teacher_err_avg:.2f}mm "
                  f"SEEN(bo1)={seen_mean:.2f}mm (n={seen_n}) UNSEEN(bo1)={unseen_mean:.2f}mm (n={unseen_n}) "
                  f"({dt:.1f}s)", flush=True)
            if seen_mean < best_seen:
                best_seen = seen_mean
                torch.save({
                    "adapter_state": adapter.state_dict(),
                    "velocity_net_state": velocity_net.state_dict(),
                    "args": vars(args),
                    "best_seen_bo1": best_seen,
                    "unseen_bo1_at_best": unseen_mean,
                }, args.out_dir / "best.pt")
        else:
            print(f"[{ep:03d}] loss={loss_avg:.4f} teacher_err={teacher_err_avg:.2f}mm ({time.time()-t0:.1f}s)", flush=True)

    torch.save({
        "adapter_state": adapter.state_dict(),
        "velocity_net_state": velocity_net.state_dict(),
        "args": vars(args),
    }, args.out_dir / "last.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] best SEEN bo1 = {best_seen:.2f}mm → {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
