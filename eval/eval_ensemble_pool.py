#!/usr/bin/env python3
"""Candidate-pool ensemble eval for graspauto / watermelon-v1 checkpoints.

For each test sample, generates K candidates from EACH of N checkpoints
independently → pool of N*K candidates → reports oracle (min mpvpe in pool).
This is the truthful "ensemble" behavior, vs velocity averaging which
collapses all models to a single trajectory per candidate.

Usage:
    python eval/eval_ensemble_pool.py \\
        --checkpoints ckpt_a.pt,ckpt_b.pt,ckpt_c.pt \\
        --num-samples-per-cond 64 \\
        --teacher-forcing --use-unified-intent --mode-coverage \\
        --out-dir outputs/ensemble_pool_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from graspauto.stage3_contact_graph import (  # noqa: E402
    DEFAULT_OBJECT_M2AE_CACHE,
    DEFAULT_STAGE3_CONTACT_GRAPH_ROOT,
    PointM2AEContactGraphModel,
    Stage3ContactGraphDataset,
)
from graspauto.utils import DEFAULT_GEOMETRY_CACHE, ensure_dir, resolve_device, write_json  # noqa: E402

from graspauto.conditioning import ContactGraphConditioningAdapter  # noqa: E402
from graspauto.flow_matching import ConditionalFlowMatching  # noqa: E402
from graspauto.mano_decoder import MANO_PARAM_DIM, MangoMANODecoder  # noqa: E402
from graspauto.velocity_network import VelocityNetwork  # noqa: E402

# Reuse helpers from eval_graspauto
from eval.eval import (  # noqa: E402
    nearest_point_min_distance_mm,
    nearest_point_penetration_mm,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, required=True,
                   help="Comma-separated graspauto checkpoints to ensemble.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--num-samples-per-cond", type=int, default=64,
                   help="K candidates per checkpoint per sample. Total pool = N*K.")
    p.add_argument("--preprocess-root", type=Path, default=DEFAULT_STAGE3_CONTACT_GRAPH_ROOT)
    p.add_argument("--val-split", type=Path, default=Path("val_oracle.pt"))
    p.add_argument("--geometry-path", type=Path, default=DEFAULT_GEOMETRY_CACHE)
    p.add_argument("--object-cache", type=Path, default=DEFAULT_OBJECT_M2AE_CACHE)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--teacher-forcing", action="store_true")
    p.add_argument("--use-unified-intent", action="store_true")
    p.add_argument("--mode-coverage", action="store_true")
    p.add_argument("--only-objects", type=str, default="")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--selector-checkpoint", type=Path, default=None,
                   help="Optional learned selector to also rank pool. Reports both oracle and selector mpvpe.")
    p.add_argument("--consensus", action="store_true",
                   help="Consensus (medoid) ranking: pick the candidate whose mean joint distance "
                        "to all other candidates is smallest. Zero-training deployment-friendly ranker.")
    return p.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def load_one_ckpt(ckpt_path: Path, device, flow_input_dim_hint=None):
    """Load a single graspauto ckpt → (adapter, velocity_net, latent_ae, latent_mean, latent_std, train_args)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    hidden_dim = int(train_args.get("hidden_dim", 256))
    n_heads = int(train_args.get("n_heads", 4))
    n_layers = int(train_args.get("n_layers", 6))

    # Latent AE (shared across ensemble normally, but each ckpt knows its own)
    latent_ae = None
    latent_mean = None
    latent_std = None
    flow_input_dim = MANO_PARAM_DIM
    latent_ae_path = train_args.get("latent_ae_ckpt")
    if latent_ae_path is not None:
        ae_path = Path(latent_ae_path)
        ae_ckpt = torch.load(ae_path, map_location=device, weights_only=False)
        ae_latent_dim = ae_ckpt["latent_dim"]
        ae_hidden_dims = ae_ckpt["hidden_dims"]
        if ae_ckpt.get("residual", False):
            from graspauto.mano_autoencoder import ResidualMANOAutoEncoder
            latent_ae = ResidualMANOAutoEncoder(
                input_dim=54, latent_dim=ae_latent_dim,
                hidden_dim=ae_ckpt["res_hidden_dim"], n_blocks=ae_ckpt["res_n_blocks"],
            ).to(device)
        else:
            from graspauto.mano_autoencoder import MANOAutoEncoder
            latent_ae = MANOAutoEncoder(
                input_dim=54, latent_dim=ae_latent_dim, hidden_dims=ae_hidden_dims,
            ).to(device)
        latent_ae.load_state_dict(ae_ckpt["model_state"])
        latent_ae.eval()
        for p in latent_ae.parameters():
            p.requires_grad_(False)
        latent_mean = ae_ckpt["train_mean"].to(device)
        latent_std = ae_ckpt["train_std"].to(device)
        flow_input_dim = ae_latent_dim

    use_intent_token = bool(train_args.get("intent_token", False))
    part_aware_gating = bool(train_args.get("part_aware_gating", False))
    palm_only_intent = bool(train_args.get("palm_only_intent", False))
    advanced_gating = bool(train_args.get("advanced_gating", False))
    residual_modulation = bool(train_args.get("residual_modulation", False))
    ckpt_use_intent_direction = bool(train_args.get("use_intent_direction", False))

    adapter = ContactGraphConditioningAdapter(
        hidden_dim=hidden_dim,
        use_intent_token=use_intent_token,
        part_aware_gating=part_aware_gating,
        palm_only_intent=palm_only_intent,
        advanced_gating=advanced_gating,
        residual_modulation=residual_modulation,
        use_intent_direction=ckpt_use_intent_direction,
    ).to(device)
    adapter_state = dict(ckpt["adapter_state"])
    if "type_embed.weight" in adapter_state:
        saved = adapter_state["type_embed.weight"]
        expected = adapter.type_embed.weight.shape
        if saved.shape[0] < expected[0]:
            pad = torch.zeros(expected[0] - saved.shape[0], saved.shape[1], dtype=saved.dtype, device=saved.device)
            adapter_state["type_embed.weight"] = torch.cat([saved, pad], dim=0)
    adapter.load_state_dict(adapter_state)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    velocity_net = VelocityNetwork(
        input_dim=flow_input_dim, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
    ).to(device)
    velocity_net.load_state_dict(ckpt["velocity_net_state"])
    velocity_net.eval()
    for p in velocity_net.parameters():
        p.requires_grad_(False)

    cfg = dict(
        adapter=adapter, velocity_net=velocity_net,
        latent_ae=latent_ae, latent_mean=latent_mean, latent_std=latent_std,
        flow_input_dim=flow_input_dim, train_args=train_args,
        use_intent_token=use_intent_token, part_aware_gating=part_aware_gating,
        palm_only_intent=palm_only_intent, advanced_gating=advanced_gating,
        ckpt_use_intent_direction=ckpt_use_intent_direction,
        ckpt_path=str(ckpt_path),
    )
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    ckpt_paths = [Path(s.strip()) for s in args.checkpoints.split(",") if s.strip()]
    print(f"[load] {len(ckpt_paths)} checkpoints")
    members = [load_one_ckpt(p, device) for p in ckpt_paths]
    for c in members:
        print(f"  - {c['ckpt_path']}")

    # ---- Dataset ----
    val_path = args.preprocess_root / args.val_split
    val_ds = Stage3ContactGraphDataset(
        split_path=val_path, geometry_path=args.geometry_path,
        object_m2ae_cache_path=args.object_cache, limit=args.val_limit,
    )
    if args.only_objects:
        keep_ids = {int(x) for x in args.only_objects.split(",") if x.strip()}
        keep_idx = [i for i in range(len(val_ds)) if int(val_ds.object_id[i].item()) in keep_ids]
        num_codes = int(val_ds.num_codes)
        from torch.utils.data import Subset
        print(f"[only-objects] {sorted(keep_ids)}: {len(keep_idx)}/{len(val_ds)} kept")
        val_ds = Subset(val_ds, keep_idx)
        val_ds.num_codes = num_codes
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    num_codes = int(val_ds.num_codes)
    print(f"[data] val={len(val_ds)} codes={num_codes}")

    # ---- Mode coverage GT pool ----
    object_id_to_gt_verts = {}
    if args.mode_coverage:
        from collections import defaultdict
        pool = defaultdict(list)
        for split_name in ["train", "val_oracle"]:
            try:
                pool_ds = Stage3ContactGraphDataset(
                    split_path=args.preprocess_root / f"{split_name}.pt",
                    geometry_path=args.geometry_path,
                    object_m2ae_cache_path=args.object_cache, limit=None,
                )
            except Exception:
                continue
            for i in range(len(pool_ds)):
                s = pool_ds[i]
                pool[int(s["object_id"].item())].append(s["gt_world_verts"])
        for oid, vlist in pool.items():
            object_id_to_gt_verts[oid] = torch.stack(vlist, dim=0).to(device)
        print(f"[mode_coverage] {len(object_id_to_gt_verts)} unique objects, "
              f"{sum(v.shape[0] for v in object_id_to_gt_verts.values())} GT refs")

    # ---- Frozen contact head (use first ckpt's warm-start spec; all members share) ----
    # Use codebook size from warm-start ckpt to avoid shape mismatch (we don't actually
    # use the codebook in teacher-forcing mode, but state_dict load is strict on shapes).
    ws_path_str = members[0]["train_args"].get("warm_start_graspauto")
    head_num_codes = num_codes
    ws_ckpt = None
    if ws_path_str is not None:
        ws_ckpt = torch.load(Path(ws_path_str), map_location=device, weights_only=False)
        if "codebook_embed.weight" in ws_ckpt.get("model_state", {}):
            head_num_codes = ws_ckpt["model_state"]["codebook_embed.weight"].shape[0]
    contact_head = PointM2AEContactGraphModel(num_codes=head_num_codes).to(device).eval()
    if ws_ckpt is not None:
        contact_head.load_state_dict(ws_ckpt["model_state"], strict=False)
    for p in contact_head.parameters():
        p.requires_grad_(False)

    mano_decoder = MangoMANODecoder().to(device).eval()
    flow = ConditionalFlowMatching()

    # ---- Optional learned selector ----
    selector_model = None
    selector_mu = None
    selector_sigma = None
    selector_input_dim = 0
    if args.selector_checkpoint is not None:
        sys.path.insert(0, str(PROJECT_ROOT / "train"))
        from train_selector import SelectorMLP
        sel_ckpt = torch.load(args.selector_checkpoint, map_location=device, weights_only=False)
        feature_names = sel_ckpt.get("feature_names", [])
        selector_input_dim = len(feature_names)
        selector_model = SelectorMLP(
            in_dim=selector_input_dim,
            hidden=sel_ckpt.get("hidden_dim", 64),
            depth=sel_ckpt.get("args", {}).get("depth", 3),
            dropout=0.0,
        ).to(device).eval()
        selector_model.load_state_dict(sel_ckpt["model_state"])
        for p in selector_model.parameters():
            p.requires_grad_(False)
        selector_mu = sel_ckpt["mu"].to(device)
        selector_sigma = sel_ckpt["sigma"].to(device)
        print(f"[selector] loaded {args.selector_checkpoint}, input_dim={selector_input_dim}", flush=True)

    # ---- Eval loop ----
    per_sample_records = []
    t0 = time.time()
    K = args.num_samples_per_cond

    with torch.no_grad():
        for raw_batch in val_loader:
            batch = move_batch_to_device(raw_batch, device)
            if args.use_unified_intent:
                batch["palm_centroid"] = batch["unified_centroid"].clone()

            if args.teacher_forcing:
                graph = {k: batch[k] for k in [
                    "finger_centroid","finger_normal","finger_spread","finger_entropy","finger_mass",
                    "palm_centroid","palm_normal","palm_spread","palm_entropy","palm_mass",
                    "unified_centroid","unified_normal","unified_spread","unified_entropy",
                ]}
                active_finger_prob = batch["active_finger_score"]
            else:
                head_out = contact_head(
                    object_points=batch["object_points"],
                    object_normals=batch["object_normals"],
                    stage1_contact_input=batch["stage1_contact_input"],
                    m2ae_global=batch["m2ae_global"],
                    m2ae_local=batch["m2ae_local"],
                    patch_centers=batch["patch_centers"],
                )
                graph = head_out["graph"]
                active_finger_prob = head_out["active_finger_prob"]

            B = batch["object_points"].shape[0]
            gt_verts = batch["gt_world_verts"]

            # Track best (oracle = min vertex error) per sample, per metric.
            best_specific_mm = torch.full((B,), float("inf"), device=device)
            # Mode-cover: per sample, min over all GT refs of object → for each cand
            # we compute mode_cover_mm per sample, then keep best across pool.
            best_modecov_mm = torch.full((B,), float("inf"), device=device) if args.mode_coverage else None
            # Track which model produced the best (for diagnostics).
            best_member_idx = torch.full((B,), -1, dtype=torch.long, device=device)

            # Consensus tracking: collect all candidate joints per sample
            if args.consensus:
                all_cand_joints = [[] for _ in range(B)]  # B lists of (21, 3) tensors
                all_cand_err_specific = [[] for _ in range(B)]
                all_cand_modecov = [[] for _ in range(B)] if args.mode_coverage else None

            # If selector enabled: track selector-picked candidate's mpvpe (NOT oracle).
            best_sel_score = torch.full((B,), float("inf"), device=device) if selector_model is not None else None
            sel_pick_specific_mm = torch.full((B,), float("inf"), device=device) if selector_model is not None else None
            sel_pick_modecov_mm = torch.full((B,), float("inf"), device=device) if (selector_model is not None and args.mode_coverage) else None
            finger_tip_joints = torch.tensor([4, 8, 12, 16, 20], device=device)
            # Use head to get rank target for selector (since teacher-forcing graph is GT, can't use it for ranking honesty).
            if selector_model is not None:
                if args.teacher_forcing:
                    # Run head separately for rank target (matches eval_graspauto honest mode)
                    head_for_rank = contact_head(
                        object_points=batch["object_points"],
                        object_normals=batch["object_normals"],
                        stage1_contact_input=batch["stage1_contact_input"],
                        m2ae_global=batch["m2ae_global"],
                        m2ae_local=batch["m2ae_local"],
                        patch_centers=batch["patch_centers"],
                    )
                    rank_target_finger_centroid = head_for_rank["graph"]["finger_centroid"]
                    rank_target_palm_centroid = head_for_rank["graph"]["palm_centroid"]
                else:
                    rank_target_finger_centroid = graph["finger_centroid"]
                    rank_target_palm_centroid = graph["palm_centroid"]

            for member_idx, m in enumerate(members):
                # Build patch_contact_weight per member's recipe.
                pcw = None
                if m["part_aware_gating"]:
                    if m["palm_only_intent"]:
                        from graspauto.conditioning import compute_patch_weight_from_point
                        palm_target = batch["palm_centroid"]
                        pcw = compute_patch_weight_from_point(
                            patch_centers=batch["patch_centers"], target_point=palm_target,
                        )
                    else:
                        from graspauto.conditioning import compute_patch_contact_weight
                        contact_mask = batch["unified_contact_target"]
                        pcw = compute_patch_contact_weight(
                            patch_centers=batch["patch_centers"],
                            object_points=batch["object_points"], contact_mask=contact_mask,
                        )
                intent_ids = batch["intent_id"] if m["use_intent_token"] else None
                adv_pc = batch["patch_centers"] if m["advanced_gating"] else None
                adv_pt = batch["palm_centroid"] if m["advanced_gating"] else None
                eval_intent_dir = None
                if m["ckpt_use_intent_direction"]:
                    _anchor = batch["hTm_trans"]
                    _tap = batch["unified_centroid"]
                    _dir = _tap - _anchor
                    _dir = _dir / _dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    eval_intent_dir = _dir

                bundle = m["adapter"](
                    m2ae_local=batch["m2ae_local"], graph=graph,
                    active_finger_prob=active_finger_prob, patch_contact_weight=pcw,
                    intent_ids=intent_ids, patch_centers=adv_pc, palm_target=adv_pt,
                    intent_direction=eval_intent_dir,
                )

                # CFG support per-member
                if args.cfg_scale != 1.0:
                    null_tokens = torch.zeros_like(bundle.tokens)
                    null_bundle = bundle.__class__(
                        tokens=null_tokens, object_tokens=null_tokens[:, :64, :],
                        contact_tokens=null_tokens[:, 64:, :], token_types=bundle.token_types,
                    )
                    cfg_s = float(args.cfg_scale)
                    def vc(xt_, t_, _ignored, _bundle=bundle, _null=null_bundle, _vn=m["velocity_net"], _s=cfg_s):
                        v_c = _vn(xt_, t_, condition=_bundle)
                        v_u = _vn(xt_, t_, condition=_null)
                        return v_u + _s * (v_c - v_u)
                    velocity_callable = vc
                    cond_for = None
                else:
                    velocity_callable = m["velocity_net"]
                    cond_for = bundle

                for k in range(K):
                    x0 = torch.randn(B, m["flow_input_dim"], device=device)
                    x1_or_z = flow.sample(
                        velocity_callable, x0, condition=cond_for,
                        num_steps=args.num_flow_steps, method="euler",
                    )
                    if m["latent_ae"] is not None:
                        mano_normed = m["latent_ae"].decode(x1_or_z)
                        x1 = mano_normed * m["latent_std"] + m["latent_mean"]
                    else:
                        x1 = x1_or_z
                    decoded = mano_decoder(x1)
                    pred_verts = decoded["vertices"]
                    pred_joints = decoded["joints"]
                    err_specific_mm = (pred_verts - gt_verts).norm(dim=-1).mean(dim=-1) * 1000.0

                    better_specific = err_specific_mm < best_specific_mm
                    best_specific_mm = torch.where(better_specific, err_specific_mm, best_specific_mm)
                    best_member_idx = torch.where(better_specific, torch.full_like(best_member_idx, member_idx), best_member_idx)

                    # Selector ranking (if enabled) — pick lowest predicted-mpvpe candidate
                    if selector_model is not None:
                        tips = pred_joints.index_select(dim=1, index=finger_tip_joints)
                        contact_mean = (tips - rank_target_finger_centroid).norm(dim=-1).mean(dim=-1) * 1000.0
                        contact_max = (tips - rank_target_finger_centroid).norm(dim=-1).max(dim=-1).values * 1000.0
                        wrist = pred_joints[:, 0]
                        palm_mm = (wrist - rank_target_palm_centroid).norm(dim=-1) * 1000.0
                        pen_mm = nearest_point_penetration_mm(
                            pred_verts, batch["object_points"], batch["object_normals"]
                        )
                        min_dist_mm = nearest_point_min_distance_mm(pred_verts, batch["object_points"])
                        a1 = x1[:, 0:3]; a2 = x1[:, 3:6]
                        rot_orth = ((a1 * a2).sum(dim=-1)).pow(2) + (a1.norm(dim=-1) - 1.0).pow(2) + (a2.norm(dim=-1) - 1.0).pow(2)
                        pose = x1[:, 9:54]
                        over = (pose - 1.5).clamp_min(0.0)
                        under = (-1.5 - pose).clamp_min(0.0)
                        joint_limit = (over.pow(2) + under.pow(2)).mean(dim=-1)
                        trans_norm = x1[:, 6:9].norm(dim=-1)
                        scalar_feats = torch.stack(
                            [contact_mean, contact_max, palm_mm, pen_mm, min_dist_mm, rot_orth, joint_limit, trans_norm],
                            dim=-1,
                        )
                        # Map input_dim to feature_set: 8=scalar, 62=scalar+x1, 71=scalar+joints, 125=all
                        if selector_input_dim == 8:
                            feat_parts = [scalar_feats]
                        elif selector_input_dim == 8 + 54:
                            feat_parts = [scalar_feats, x1]
                        elif selector_input_dim == 8 + 63:
                            feat_parts = [scalar_feats, pred_joints.reshape(B, -1)]
                        elif selector_input_dim == 8 + 54 + 63:
                            feat_parts = [scalar_feats, x1, pred_joints.reshape(B, -1)]
                        else:
                            raise RuntimeError(f"unknown selector input_dim={selector_input_dim}")
                        feats_row = torch.cat(feat_parts, dim=-1)
                        feats_norm = (feats_row - selector_mu) / selector_sigma
                        sel_score = selector_model(feats_norm).squeeze(-1)
                        better_sel = sel_score < best_sel_score
                        best_sel_score = torch.where(better_sel, sel_score, best_sel_score)
                        sel_pick_specific_mm = torch.where(better_sel, err_specific_mm, sel_pick_specific_mm)

                    if args.mode_coverage:
                        # For each sample b in batch, compute min over GT pool of object_id.
                        modecov_b = torch.empty(B, device=device)
                        for b in range(B):
                            oid = int(batch["object_id"][b].item())
                            gt_pool = object_id_to_gt_verts.get(oid)
                            if gt_pool is None:
                                modecov_b[b] = err_specific_mm[b]  # fallback to specific
                                continue
                            # per-grasp mpvpe over the pool
                            errs = (pred_verts[b].unsqueeze(0) - gt_pool).norm(dim=-1).mean(dim=-1) * 1000.0
                            modecov_b[b] = errs.min()
                        better_mc = modecov_b < best_modecov_mm
                        best_modecov_mm = torch.where(better_mc, modecov_b, best_modecov_mm)
                        if selector_model is not None:
                            sel_pick_modecov_mm = torch.where(better_sel, modecov_b, sel_pick_modecov_mm)

                    # Collect for consensus
                    if args.consensus:
                        for b in range(B):
                            all_cand_joints[b].append(pred_joints[b].cpu())
                            all_cand_err_specific[b].append(float(err_specific_mm[b].item()))
                            if all_cand_modecov is not None:
                                all_cand_modecov[b].append(float(modecov_b[b].item()))

            # Consensus medoid selection
            consensus_specific = [None] * B
            consensus_modecov = [None] * B
            if args.consensus:
                for b in range(B):
                    joints_stack = torch.stack(all_cand_joints[b], dim=0)  # (N_cand, 21, 3)
                    N_cand = joints_stack.shape[0]
                    # Pairwise MPJPE: (N, N) matrix
                    flat = joints_stack.reshape(N_cand, -1)  # (N, 63)
                    dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)  # (N, N)
                    mean_dist = dists.mean(dim=1)  # (N,)
                    medoid_idx = mean_dist.argmin().item()
                    consensus_specific[b] = all_cand_err_specific[b][medoid_idx]
                    if all_cand_modecov is not None:
                        consensus_modecov[b] = all_cand_modecov[b][medoid_idx]

            # Record per-sample
            for b in range(B):
                rec = {
                    "object_id": int(batch["object_id"][b].item()),
                    "specific_mpvpe_mm": float(best_specific_mm[b].item()),
                    "best_member": int(best_member_idx[b].item()),
                }
                if args.mode_coverage:
                    rec["mode_cover_mm"] = float(best_modecov_mm[b].item())
                if selector_model is not None:
                    rec["selector_specific_mpvpe_mm"] = float(sel_pick_specific_mm[b].item())
                    if args.mode_coverage:
                        rec["selector_mode_cover_mm"] = float(sel_pick_modecov_mm[b].item())
                if args.consensus:
                    rec["consensus_specific_mpvpe_mm"] = consensus_specific[b]
                    if consensus_modecov[b] is not None:
                        rec["consensus_mode_cover_mm"] = consensus_modecov[b]
                per_sample_records.append(rec)

            # Progress
            done = len(per_sample_records)
            print(f"  [{done}/{len(val_ds)}] elapsed {time.time()-t0:.1f}s", flush=True)

    # Summary
    spec = torch.tensor([r["specific_mpvpe_mm"] for r in per_sample_records])
    summary = {
        "ensemble_checkpoints": [str(p) for p in ckpt_paths],
        "num_samples_per_cond_per_member": K,
        "total_pool_size_per_sample": K * len(members),
        "val_samples_evaluated": len(per_sample_records),
        "elapsed_sec": time.time() - t0,
        "mpvpe_mm": {
            "mean": float(spec.mean()), "median": float(spec.median()),
            "min": float(spec.min()), "max": float(spec.max()),
        },
    }
    if args.mode_coverage:
        mc = torch.tensor([r["mode_cover_mm"] for r in per_sample_records])
        summary["mode_cover_mm"] = {
            "mean": float(mc.mean()), "median": float(mc.median()),
            "p10": float(torch.quantile(mc, 0.10)),
            "p90": float(torch.quantile(mc, 0.90)),
            "min": float(mc.min()), "max": float(mc.max()),
            "frac_under_10mm": float((mc < 10).float().mean()),
            "frac_under_20mm": float((mc < 20).float().mean()),
            "frac_under_30mm": float((mc < 30).float().mean()),
        }
    if selector_model is not None:
        sm = torch.tensor([r["selector_specific_mpvpe_mm"] for r in per_sample_records])
        summary["selector_specific_mpvpe_mm"] = {
            "mean": float(sm.mean()), "median": float(sm.median()),
            "min": float(sm.min()), "max": float(sm.max()),
        }
        if args.mode_coverage:
            smc = torch.tensor([r["selector_mode_cover_mm"] for r in per_sample_records])
            summary["selector_mode_cover_mm"] = {
                "mean": float(smc.mean()), "median": float(smc.median()),
                "p10": float(torch.quantile(smc, 0.10)),
                "p90": float(torch.quantile(smc, 0.90)),
                "min": float(smc.min()), "max": float(smc.max()),
                "frac_under_10mm": float((smc < 10).float().mean()),
                "frac_under_20mm": float((smc < 20).float().mean()),
                "frac_under_30mm": float((smc < 30).float().mean()),
            }
    if args.consensus:
        cs = torch.tensor([r["consensus_specific_mpvpe_mm"] for r in per_sample_records])
        summary["consensus_specific_mpvpe_mm"] = {
            "mean": float(cs.mean()), "median": float(cs.median()),
            "min": float(cs.min()), "max": float(cs.max()),
        }
        if args.mode_coverage and "consensus_mode_cover_mm" in per_sample_records[0]:
            cmc = torch.tensor([r["consensus_mode_cover_mm"] for r in per_sample_records])
            summary["consensus_mode_cover_mm"] = {
                "mean": float(cmc.mean()), "median": float(cmc.median()),
                "p10": float(torch.quantile(cmc, 0.10)),
                "p90": float(torch.quantile(cmc, 0.90)),
                "min": float(cmc.min()), "max": float(cmc.max()),
                "frac_under_10mm": float((cmc < 10).float().mean()),
                "frac_under_20mm": float((cmc < 20).float().mean()),
                "frac_under_30mm": float((cmc < 30).float().mean()),
            }

    # Member contribution histogram
    members_count = {i: 0 for i in range(len(members))}
    for r in per_sample_records:
        members_count[r["best_member"]] = members_count.get(r["best_member"], 0) + 1
    summary["best_member_histogram"] = {
        str(i): {"checkpoint": str(ckpt_paths[i]), "count": c, "fraction": c / len(per_sample_records)}
        for i, c in members_count.items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "per_sample.json", per_sample_records)
    print(f"\n[done] {len(per_sample_records)} samples evaluated in {time.time()-t0:.1f}s")
    print(json.dumps(summary["mode_cover_mm"] if args.mode_coverage else summary["mpvpe_mm"], indent=2))


if __name__ == "__main__":
    main()
