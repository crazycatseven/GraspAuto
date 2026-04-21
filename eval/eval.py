#!/usr/bin/env python3
"""graspauto evaluation entrypoint — vertex-error metric on the validation set.

Loads a trained graspauto checkpoint, runs the conditional flow matching
sampler on the validation set, and computes per-sample mean per-vertex
position error (MPVPE) in millimeters against the GT MANO mesh.

Usage:

    .venv/bin/python eval/eval.py \\
        --checkpoint outputs/graspauto_phase1_bringup/best.pt \\
        --val-split val_oracle.pt \\
        --val-limit 100 \\
        --num-flow-steps 10 \\
        --device auto

This is the script that converts graspauto's "flow loss" (which is in
parameter space and not directly interpretable in millimeters) into a
real, comparable-to-graspauto vertex error number. The eval pipeline:

  Stage3ContactGraphDataset
    → frozen graspauto contact head
    → graspauto ConditioningAdapter
    → ConditionalFlowMatching.sample (RK4, num_steps=10)
    → MangoMANODecoder (54-D → MANO mesh)
    → vertex distance to gt_world_verts

The output is mean / median / p10 / p90 mm vertex error per sample,
plus a JSON dump of per-sample metrics for downstream analysis.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a graspauto training checkpoint (e.g., outputs/graspauto_phase1_bringup/best.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--method", type=str, default="rk4", choices=["euler", "rk4"])
    parser.add_argument("--num-samples-per-cond", type=int, default=1,
                        help="How many samples (different x0) to draw per condition. >1 enables best-of-N selection.")
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_STAGE3_CONTACT_GRAPH_ROOT)
    parser.add_argument("--val-split", type=Path, default=Path("val_oracle.pt"))
    parser.add_argument("--geometry-path", type=Path, default=DEFAULT_GEOMETRY_CACHE)
    parser.add_argument("--object-cache", type=Path, default=DEFAULT_OBJECT_M2AE_CACHE)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for the eval JSON. Defaults to <checkpoint dir>/eval_<timestamp>/.")
    parser.add_argument("--teacher-forcing", action="store_true",
                        help="DIAGNOSTIC: use GT contact graph from the batch as conditioning instead of "
                             "the frozen head's predicted graph. This is what a Phase 2c training run saw "
                             "during training; comparing teacher-forcing vs not isolates the inference-time "
                             "distribution shift from predicted-vs-GT contact features.")
    parser.add_argument("--use-unified-intent", action="store_true",
                        help="(sphere) rebind palm_centroid to unified_centroid at eval time. "
                             "REQUIRED when evaluating a model trained with --use-unified-intent, "
                             "otherwise the tap input is OOD.")
    parser.add_argument("--use-approach-direction", action="store_true",
                        help="(sphere) rebind palm_normal to approach-direction "
                             "(hTm_rot @ [0,0,1]). REQUIRED for r002+ ckpts at eval time.")
    parser.add_argument("--use-sphere-intent", action="store_true",
                        help="(sphere) sphere-patch intent. REQUIRED for r004+ ckpts.")
    parser.add_argument("--rank-by", type=str, default="oracle",
                        choices=["oracle", "contact_align", "penetration", "composite", "learned_selector"],
                        help="How to rank best-of-N candidates. Options:\n"
                             "  'oracle'          : argmin vertex error vs GT (upper bound, cheats)\n"
                             "  'contact_align'   : argmin fingertip→predicted-centroid distance\n"
                             "  'penetration'     : argmin hand-inside-object penetration\n"
                             "  'composite'       : argmin weighted sum of the inference-time terms\n"
                             "  'learned_selector': argmin predicted-mpvpe from a trained MLP")
    parser.add_argument("--composite-weights", type=str, default="1.0,10.0,0.1",
                        help="Comma-separated weights for composite scorer: "
                             "contact_align_mm, penetration_mm, joint_limit. Default '1.0,10.0,0.1' "
                             "puts most weight on avoiding penetration (which otherwise is the dominant "
                             "failure mode — hand through the object).")
    parser.add_argument("--selector-checkpoint", type=Path, default=None,
                        help="Path to a trained graspauto selector (produced by train_graspauto_selector.py). "
                             "Required when --rank-by learned_selector.")
    parser.add_argument("--cfg-anneal", type=str, default=None,
                         help="Optional annealed CFG schedule 'cfg_start,cfg_end' "
                              "(e.g. '2.0,1.0'). If set, CFG linearly decays from "
                              "cfg_start at t=0 to cfg_end at t=1 across Euler steps. "
                              "Overrides --cfg-scale.")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale. 1.0 = no CFG (pure conditional), "
                             ">1 sharpens toward the conditional mode, <1 mixes in unconditional. "
                             "Only meaningful if the checkpoint was trained with cfg_dropout>0.")
    parser.add_argument("--tto-steps", type=int, default=0,
                        help="Number of test-time-optimization steps to run on the BEST-OF-N "
                             "picked candidate. 0 = no TTO. Recommended: 100–300. Each step is "
                             "an Adam update on the full 54-D MANO vector minimizing contact + "
                             "penetration + joint limit + rot orthogonality + stay-close-to-init "
                             "prior. See src/graspauto/tto.py for details.")
    parser.add_argument("--tto-lr", type=float, default=5e-3,
                        help="TTO Adam learning rate.")
    parser.add_argument("--tto-w-pen", type=float, default=5.0,
                        help="TTO penetration loss weight.")
    parser.add_argument("--tto-w-contact", type=float, default=0.0,
                        help="TTO fingertip→predicted-centroid alignment loss weight. "
                             "Default 0 (disabled) because the frozen head's predicted "
                             "centroids are ~26mm off GT, so aligning to them drifts away "
                             "from the true optimum.")
    parser.add_argument("--tto-w-surface", type=float, default=1.0,
                        help="TTO fingertip→nearest-object-surface loss weight. Default 1.0 — "
                             "a GT-free, physics-grounded contact objective.")
    parser.add_argument("--tto-w-prior", type=float, default=0.01,
                        help="TTO prior loss weight (pulls params toward the init; prevents "
                             "drifting off-manifold).")
    parser.add_argument("--mode-coverage", action="store_true",
                        help="Also compute mode-coverage metric: for each predicted grasp, "
                             "report distance to the NEAREST GT grasp of the same object "
                             "(across all train+val GT grasps), not just the single specific GT. "
                             "This is the correct VR metric: 'did you produce SOME valid grasp?' "
                             "Regular mpvpe stays in the output as 'specific_mpvpe_mm'.")
    parser.add_argument("--only-objects", type=str, default="",
                        help="Comma-separated object_ids to KEEP in val (filter out all others). "
                             "Used for held-out-object generalization eval, e.g. "
                             "--only-objects 0,7,14,21 on a model trained with "
                             "--holdout-objects 0,7,14,21 to measure unseen-object performance.")
    parser.add_argument("--ensemble-checkpoints", type=str, default="",
                        help="Comma-separated paths to ADDITIONAL graspauto checkpoints for "
                             "velocity ensemble. At each ODE step, the velocity predictions "
                             "from --checkpoint plus all ensemble members are averaged. All "
                             "checkpoints must share the same architecture (hidden_dim, n_heads, "
                             "n_layers, palm_only_intent, etc). E.g.: "
                             "--checkpoint best_seed42.pt --ensemble-checkpoints best_seed7.pt,best_seed13.pt")
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def nearest_point_min_distance_mm(
    hand_verts: torch.Tensor,
    object_points: torch.Tensor,
) -> torch.Tensor:
    """Min distance (mm) from any hand vertex to any object point."""
    hv_sq = hand_verts.pow(2).sum(dim=-1, keepdim=True)
    op_sq = object_points.pow(2).sum(dim=-1).unsqueeze(1)
    cross = torch.bmm(hand_verts, object_points.transpose(-1, -2))
    dist_sq = (hv_sq + op_sq - 2.0 * cross).clamp_min(0.0)
    return dist_sq.min(dim=-1).values.min(dim=-1).values.sqrt() * 1000.0


def nearest_point_penetration_mm(
    hand_verts: torch.Tensor,       # (B, 778, 3) — candidate hand vertices in object frame
    object_points: torch.Tensor,    # (B, N_obj, 3)
    object_normals: torch.Tensor,   # (B, N_obj, 3), assumed outward-pointing
) -> torch.Tensor:
    """Approximate per-sample penetration depth in mm.

    For each hand vertex, find its nearest object point. If the direction
    from the object point to the hand vertex is on the *opposite* side of
    the object normal (dot product negative), the vertex is "inside" the
    object. The penetration magnitude is |negative dot product|.

    Returns a (B,) tensor of mean penetration per sample in millimeters.
    This is an O(B * V * N) nearest-neighbor search; for V=778 and
    N_obj=3000, on GPU that's ~2M ops per sample which is fast.
    """
    B, V, _ = hand_verts.shape
    N = object_points.shape[1]
    # Pairwise distances: (B, V, N)
    #   using x^2 + y^2 - 2xy expansion via bmm for speed
    hv_sq = hand_verts.pow(2).sum(dim=-1, keepdim=True)       # (B, V, 1)
    op_sq = object_points.pow(2).sum(dim=-1).unsqueeze(1)      # (B, 1, N)
    # cross term: (B, V, 3) @ (B, 3, N) = (B, V, N)
    cross = torch.bmm(hand_verts, object_points.transpose(-1, -2))
    dist_sq = (hv_sq + op_sq - 2.0 * cross).clamp_min(0.0)
    nearest_idx = dist_sq.argmin(dim=-1)                       # (B, V)

    # Gather nearest object points and normals
    idx_exp = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)      # (B, V, 3)
    nearest_pts = object_points.gather(1, idx_exp)             # (B, V, 3)
    nearest_nor = object_normals.gather(1, idx_exp)            # (B, V, 3)

    # Signed distance: (hand - nearest) · normal. Negative means "inside".
    diff = hand_verts - nearest_pts
    signed = (diff * nearest_nor).sum(dim=-1)                  # (B, V)

    # Penetration depth = max(0, -signed), summed and converted to mm
    penetration = (-signed).clamp_min(0.0)                     # (B, V)
    return penetration.mean(dim=-1) * 1000.0                   # (B,) in mm


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    # ---- Load checkpoint ----
    print(f"[load] {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "args" not in ckpt or "adapter_state" not in ckpt or "velocity_net_state" not in ckpt:
        raise ValueError(
            f"checkpoint at {args.checkpoint} is missing required keys "
            f"(expected 'args', 'adapter_state', 'velocity_net_state'); got {list(ckpt.keys())}"
        )
    train_args = ckpt["args"]
    hidden_dim = int(train_args.get("hidden_dim", 256))
    n_heads = int(train_args.get("n_heads", 4))
    n_layers = int(train_args.get("n_layers", 6))
    is_hierarchical = bool(train_args.get("hierarchical", False))
    WRIST_DIM = 9
    FINGER_DIM = 45

    # ---- Latent flow setup ----
    latent_ae = None
    latent_mean = None
    latent_std = None
    flow_input_dim = MANO_PARAM_DIM
    latent_ae_path = train_args.get("latent_ae_ckpt")
    if latent_ae_path is not None:
        ae_path = Path(latent_ae_path)
        if not ae_path.exists():
            raise FileNotFoundError(f"latent AE checkpoint not found: {ae_path}")
        ae_ckpt = torch.load(ae_path, map_location=device, weights_only=False)
        ae_latent_dim = ae_ckpt["latent_dim"]
        ae_hidden_dims = ae_ckpt["hidden_dims"]
        if ae_ckpt.get("residual", False):
            from graspauto.mano_autoencoder import ResidualMANOAutoEncoder  # noqa: PLC0415
            latent_ae = ResidualMANOAutoEncoder(
                input_dim=54, latent_dim=ae_latent_dim,
                hidden_dim=ae_ckpt["res_hidden_dim"], n_blocks=ae_ckpt["res_n_blocks"],
            ).to(device)
        else:
            from graspauto.mano_autoencoder import MANOAutoEncoder  # noqa: PLC0415
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
        print(f"[latent] AE loaded: latent_dim={ae_latent_dim}, eval in {ae_latent_dim}-D latent space", flush=True)

    # ---- Dataset ----
    val_path = args.preprocess_root / args.val_split
    val_ds = Stage3ContactGraphDataset(
        split_path=val_path,
        geometry_path=args.geometry_path,
        object_m2ae_cache_path=args.object_cache,
        limit=args.val_limit,
    )
    if args.only_objects:
        keep_ids = {int(x) for x in args.only_objects.split(",") if x.strip()}
        keep_idx = [i for i in range(len(val_ds)) if int(val_ds.object_id[i].item()) in keep_ids]
        num_codes = int(val_ds.num_codes)
        from torch.utils.data import Subset  # noqa: PLC0415
        print(f"[only-objects] keeping {sorted(keep_ids)}: {len(keep_idx)}/{len(val_ds)} samples", flush=True)
        val_ds = Subset(val_ds, keep_idx)
        val_ds.num_codes = num_codes  # type: ignore[attr-defined]
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    num_codes = int(val_ds.num_codes)
    print(f"[data] val={len(val_ds)} codes={num_codes}", flush=True)

    # ---- Mode-coverage prep: collect per-object pool of GT grasps ----
    # Pool = all GT grasps of each object (from train + val, indexed by object_id).
    # For each predicted sample, we'll compute the minimum vertex distance to ANY
    # GT grasp of that object — the "mode coverage" metric.
    object_id_to_gt_verts: dict[int, torch.Tensor] = {}
    if args.mode_coverage:
        print("[mode_coverage] collecting per-object GT grasp pool...", flush=True)
        from collections import defaultdict as _dd
        pool: dict[int, list[torch.Tensor]] = _dd(list)
        for split_name in ["train", "val_oracle"]:
            try:
                pool_ds = Stage3ContactGraphDataset(
                    split_path=args.preprocess_root / f"{split_name}.pt",
                    geometry_path=args.geometry_path,
                    object_m2ae_cache_path=args.object_cache,
                    limit=None,
                )
            except Exception:
                continue
            for i in range(len(pool_ds)):
                s = pool_ds[i]
                oid = int(s["object_id"].item())
                pool[oid].append(s["gt_world_verts"])
        for oid, verts_list in pool.items():
            object_id_to_gt_verts[oid] = torch.stack(verts_list, dim=0).to(device)  # (K_oid, 778, 3)
        total_refs = sum(v.shape[0] for v in object_id_to_gt_verts.values())
        print(
            f"[mode_coverage] {len(object_id_to_gt_verts)} unique objects, "
            f"{total_refs} total GT-grasp references in pool",
            flush=True,
        )

    # ---- Models ----
    contact_head = PointM2AEContactGraphModel(num_codes=num_codes).to(device).eval()
    # If the checkpoint stored a contact head state (Phase 2+), load it.
    if "contact_head_state" in ckpt:
        contact_head.load_state_dict(ckpt["contact_head_state"], strict=False)
        print("[load] contact head warm-started from checkpoint", flush=True)
    elif train_args.get("warm_start_graspauto") is not None:
        # Re-load from the same warm-start checkpoint the trainer used.
        ws_path = Path(train_args["warm_start_graspauto"])
        if ws_path.exists():
            ws_ckpt = torch.load(ws_path, map_location=device, weights_only=False)
            contact_head.load_state_dict(ws_ckpt["model_state"], strict=False)
            print(f"[load] contact head re-loaded from warm-start: {ws_path}", flush=True)
    for p in contact_head.parameters():
        p.requires_grad_(False)

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
    # Backward-compat: pre-r003 ckpts have type_embed (5, hidden); new arch has
    # (6, hidden) for TOKEN_TYPE_DIRECTION. Pad old weight with a zero row so
    # load_state_dict accepts it (the new slot is only used when use_intent_direction=True).
    adapter_state = dict(ckpt["adapter_state"])
    if "type_embed.weight" in adapter_state:
        saved = adapter_state["type_embed.weight"]
        expected = adapter.type_embed.weight.shape
        if saved.shape[0] < expected[0]:
            pad = torch.zeros(expected[0] - saved.shape[0], saved.shape[1], dtype=saved.dtype, device=saved.device)
            adapter_state["type_embed.weight"] = torch.cat([saved, pad], dim=0)
    adapter.load_state_dict(adapter_state)
    adapter.eval()

    velocity_net_finger = None
    wrist_proj = None
    if is_hierarchical:
        velocity_net = VelocityNetwork(
            input_dim=WRIST_DIM, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
        ).to(device)
        velocity_net.load_state_dict(ckpt["velocity_net_state"])
        velocity_net.eval()
        velocity_net_finger = VelocityNetwork(
            input_dim=FINGER_DIM, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
        ).to(device)
        velocity_net_finger.load_state_dict(ckpt["velocity_net_finger_state"])
        velocity_net_finger.eval()
        import torch.nn as _nn
        wrist_proj = _nn.Linear(WRIST_DIM, hidden_dim).to(device)
        wrist_proj.load_state_dict(ckpt["wrist_proj_state"])
        wrist_proj.eval()
        print(f"[hierarchical] loaded wrist net (9-D) + finger net (45-D) + wrist projection", flush=True)
    else:
        velocity_net = VelocityNetwork(
            input_dim=flow_input_dim, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
        ).to(device)
        velocity_net.load_state_dict(ckpt["velocity_net_state"])
        velocity_net.eval()

    try:
        mano_decoder = MangoMANODecoder().to(device).eval()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"MangoMANODecoder unavailable: {e}") from e

    flow = ConditionalFlowMatching()

    # ---- Ensemble members (optional) ----
    ensemble_members: list[tuple] = []  # list of (adapter_k, velocity_net_k)
    if args.ensemble_checkpoints:
        for ens_path_str in args.ensemble_checkpoints.split(","):
            ens_path = Path(ens_path_str.strip())
            if not ens_path.exists():
                raise FileNotFoundError(f"ensemble checkpoint not found: {ens_path}")
            ens_ckpt = torch.load(ens_path, map_location=device, weights_only=False)
            adapter_k = ContactGraphConditioningAdapter(
                hidden_dim=hidden_dim,
                use_intent_token=use_intent_token,
                part_aware_gating=part_aware_gating,
                palm_only_intent=palm_only_intent,
                advanced_gating=advanced_gating,
            ).to(device)
            adapter_k.load_state_dict(ens_ckpt["adapter_state"])
            adapter_k.eval()
            vnet_k = VelocityNetwork(
                input_dim=flow_input_dim,  # use flow dim (latent or raw)
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_layers,
            ).to(device)
            vnet_k.load_state_dict(ens_ckpt["velocity_net_state"])
            vnet_k.eval()
            for p in adapter_k.parameters():
                p.requires_grad_(False)
            for p in vnet_k.parameters():
                p.requires_grad_(False)
            ensemble_members.append((adapter_k, vnet_k))
            print(f"[ensemble] loaded member: {ens_path}", flush=True)
        print(f"[ensemble] {1 + len(ensemble_members)} total models (primary + {len(ensemble_members)} members)", flush=True)

    # Optionally load the learned selector.
    selector_model = None
    selector_mu = None
    selector_sigma = None
    selector_input_dim = 0
    if args.rank_by == "learned_selector":
        if args.selector_checkpoint is None:
            raise ValueError("--rank-by learned_selector requires --selector-checkpoint")
        sys.path.insert(0, str(PROJECT_ROOT / "train"))
        from train_selector import SelectorMLP  # noqa: E402
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
        print(f"[load] learned selector from {args.selector_checkpoint}, input_dim={selector_input_dim}", flush=True)

    # ---- Sampling + vertex error ----
    per_sample_records: list[dict] = []
    t0 = time.time()

    with torch.no_grad():
        for raw_batch in val_loader:
            batch = move_batch_to_device(raw_batch, device)

            # (sphere) unified intent swap. Must match training-time
            # behavior — otherwise we feed a palm-tap to a model trained on
            # unified-taps, producing out-of-distribution conditioning.
            if getattr(args, "use_unified_intent", False):
                batch["palm_centroid"] = batch["unified_centroid"].clone()
            if getattr(args, "use_approach_direction", False):
                hTm_rot = batch["hTm_rot"]
                fwd_local = torch.tensor([0.0, 0.0, 1.0], device=hTm_rot.device, dtype=hTm_rot.dtype)
                batch["palm_normal"] = hTm_rot @ fwd_local
            if getattr(args, "use_sphere_intent", False):
                radius = batch["unified_spread"].norm(dim=-1, keepdim=True)
                batch["palm_spread"] = radius.expand(-1, 3).contiguous()
                batch["palm_entropy"] = batch["unified_entropy"].clone()

            if args.teacher_forcing:
                graph = {
                    "finger_centroid": batch["finger_centroid"],
                    "finger_normal":   batch["finger_normal"],
                    "finger_spread":   batch["finger_spread"],
                    "finger_entropy":  batch["finger_entropy"],
                    "finger_mass":     batch["finger_mass"],
                    "palm_centroid":   batch["palm_centroid"],
                    "palm_normal":     batch["palm_normal"],
                    "palm_spread":     batch["palm_spread"],
                    "palm_entropy":    batch["palm_entropy"],
                    "palm_mass":       batch["palm_mass"],
                    "unified_centroid":batch["unified_centroid"],
                    "unified_normal":  batch["unified_normal"],
                    "unified_spread":  batch["unified_spread"],
                    "unified_entropy": batch["unified_entropy"],
                }
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
            patch_contact_weight = None
            if part_aware_gating:
                if palm_only_intent:
                    from graspauto.conditioning import compute_patch_weight_from_point  # noqa: PLC0415
                    # Palm target in inference: GT palm centroid when teacher-forcing (simulating
                    # the user tap), else the frozen head's predicted palm_centroid (fallback).
                    if args.teacher_forcing:
                        palm_target = batch["palm_centroid"]
                    else:
                        palm_target = graph["palm_centroid"]
                    patch_contact_weight = compute_patch_weight_from_point(
                        patch_centers=batch["patch_centers"],
                        target_point=palm_target,
                    )
                else:
                    from graspauto.conditioning import compute_patch_contact_weight  # noqa: PLC0415
                    # Training used unified_contact_target (GT) with teacher forcing; at inference
                    # we use the frozen head's stage1_contact_input prediction. Under --teacher-forcing
                    # at eval we could use GT, but that would be cheating — stick with the inference-
                    # time signal to be honest.
                    contact_mask_for_gate = batch["stage1_contact_input"] if not args.teacher_forcing else batch["unified_contact_target"]
                    patch_contact_weight = compute_patch_contact_weight(
                        patch_centers=batch["patch_centers"],
                        object_points=batch["object_points"],
                        contact_mask=contact_mask_for_gate,
                    )
                # r023: apply hard top-K mask if the ckpt was trained with it.
                _topk_gate = int(train_args.get("topk_gate", 0) or 0)
                if _topk_gate > 0 and patch_contact_weight is not None:
                    _vals, _idx = patch_contact_weight.topk(_topk_gate, dim=-1)
                    _mask = torch.zeros_like(patch_contact_weight)
                    _mask.scatter_(-1, _idx, 1.0)
                    patch_contact_weight = _mask
            intent_ids = batch["intent_id"] if use_intent_token else None
            adv_patch_centers = batch["patch_centers"] if advanced_gating else None
            adv_palm_target = batch["palm_centroid"] if advanced_gating else None

            # (sphere) pass intent_direction if the loaded ckpt was trained with it.
            eval_intent_direction = None
            if ckpt_use_intent_direction:
                _anchor = batch["hTm_trans"]
                _tap = batch["unified_centroid"]
                _dir = _tap - _anchor
                _dir = _dir / _dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                eval_intent_direction = _dir

            bundle = adapter(
                m2ae_local=batch["m2ae_local"],
                graph=graph,
                active_finger_prob=active_finger_prob,
                patch_contact_weight=patch_contact_weight,
                intent_ids=intent_ids,
                patch_centers=adv_patch_centers,
                palm_target=adv_palm_target,
                intent_direction=eval_intent_direction,
            )

            B = batch["object_points"].shape[0]

            # For non-oracle ranking, we always need the frozen head's PREDICTED
            # contact graph as the ranking target (regardless of teacher forcing),
            # because at real inference GT is not available. If --teacher-forcing
            # is on for the conditioning we still run the head separately for the
            # scorer target.
            need_rank_graph = args.rank_by in ("contact_align", "composite", "learned_selector")
            if need_rank_graph:
                if args.teacher_forcing:
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
            finger_tip_joints = torch.tensor([4, 8, 12, 16, 20], device=device)

            # Composite scorer weights
            if args.rank_by == "composite":
                try:
                    comp_w = [float(x) for x in args.composite_weights.split(",")]
                except Exception as e:
                    raise ValueError(f"bad --composite-weights: {args.composite_weights}") from e
                if len(comp_w) != 3:
                    raise ValueError(
                        f"--composite-weights must have 3 comma-separated values, got {comp_w}"
                    )
                w_contact, w_pen, w_joint = comp_w

            # Sample K candidates per condition. Track both the ranker's chosen
            # score and the GT error at that pick so we can report mpvpe_mm.
            best_rank_score = torch.full((B,), float("inf"), device=device)
            best_err_at_pick_mm = torch.full((B,), float("inf"), device=device)
            best_picked_x1 = torch.zeros(B, MANO_PARAM_DIM, device=device)

            # Build ensemble bundles from each member's adapter (same batch data).
            ensemble_bundles: list = []
            if ensemble_members:
                for adapter_k, _vnet_k in ensemble_members:
                    bundle_k = adapter_k(
                        m2ae_local=batch["m2ae_local"],
                        graph=graph,
                        active_finger_prob=active_finger_prob,
                        patch_contact_weight=patch_contact_weight,
                        intent_ids=intent_ids,
                        patch_centers=adv_patch_centers,
                        palm_target=adv_palm_target,
                    )
                    ensemble_bundles.append(bundle_k)

            # Build the velocity callable (with CFG if requested).
            # Supports: (a) fixed CFG via --cfg-scale, (b) annealed CFG via --cfg-anneal "start,end"
            cfg_anneal = None
            if getattr(args, "cfg_anneal", None):
                try:
                    _s, _e = args.cfg_anneal.split(",")
                    cfg_anneal = (float(_s), float(_e))
                except Exception:
                    cfg_anneal = None
            cfg_active = cfg_anneal is not None or args.cfg_scale != 1.0
            if cfg_active:
                # Build an "unconditional" bundle by zeroing the conditioning tokens.
                null_tokens = torch.zeros_like(bundle.tokens)
                null_bundle = bundle.__class__(
                    tokens=null_tokens,
                    object_tokens=null_tokens[:, :64, :],
                    contact_tokens=null_tokens[:, 64:, :],
                    token_types=bundle.token_types,
                )
                cfg_scale = float(args.cfg_scale)

                if cfg_anneal is not None:
                    cfg_s0, cfg_s1 = cfg_anneal
                    def cfg_velocity(xt_, t_, _cond_ignored):
                        v_cond = velocity_net(xt_, t_, condition=bundle)
                        v_uncond = velocity_net(xt_, t_, condition=null_bundle)
                        # t_ is a scalar tensor in [0, 1]; linear schedule
                        t_scalar = float(t_.mean() if t_.dim() > 0 else t_)
                        s = cfg_s0 + (cfg_s1 - cfg_s0) * t_scalar
                        return v_uncond + s * (v_cond - v_uncond)
                else:
                    def cfg_velocity(xt_, t_, _cond_ignored):
                        v_cond = velocity_net(xt_, t_, condition=bundle)
                        v_uncond = velocity_net(xt_, t_, condition=null_bundle)
                        return v_uncond + cfg_scale * (v_cond - v_uncond)

                velocity_callable = cfg_velocity
                condition_for_sample = None  # ignored by cfg_velocity
            elif ensemble_members:
                # Velocity ensemble: average predictions from all models.
                # Each model uses its own adapter's conditioning bundle.
                _all_pairs = [(velocity_net, bundle)] + [
                    (vnet_k, bun_k) for (_a, vnet_k), bun_k
                    in zip(ensemble_members, ensemble_bundles)
                ]
                _n_models = len(_all_pairs)

                def ensemble_velocity(xt_, t_, _cond_ignored):
                    total_v = torch.zeros_like(xt_)
                    for vnet_m, bundle_m in _all_pairs:
                        total_v = total_v + vnet_m(xt_, t_, condition=bundle_m)
                    return total_v / _n_models

                velocity_callable = ensemble_velocity
                condition_for_sample = None  # ignored by ensemble_velocity
            else:
                velocity_callable = velocity_net
                condition_for_sample = bundle

            for k in range(args.num_samples_per_cond):
                if is_hierarchical:
                    # Stage 1: generate wrist (9-D)
                    wrist_x0 = torch.randn(B, WRIST_DIM, device=device)
                    wrist_x1 = flow.sample(
                        velocity_net, wrist_x0, condition=bundle,
                        num_steps=args.num_flow_steps, method=args.method,
                    )
                    # Stage 2: generate fingers (45-D) conditioned on generated wrist
                    wrist_token = wrist_proj(wrist_x1).unsqueeze(1)  # (B, 1, H)
                    stage2_tokens = torch.cat([bundle.tokens, wrist_token], dim=1)
                    from graspauto.conditioning import ConditioningBundle as _CB  # noqa: PLC0415
                    stage2_bundle = _CB(
                        tokens=stage2_tokens,
                        object_tokens=bundle.object_tokens,
                        contact_tokens=bundle.contact_tokens,
                        token_types=torch.cat([bundle.token_types,
                            torch.full((1,), 5, dtype=torch.long, device=device)]),
                    )
                    finger_x0 = torch.randn(B, FINGER_DIM, device=device)
                    finger_x1 = flow.sample(
                        velocity_net_finger, finger_x0, condition=stage2_bundle,
                        num_steps=args.num_flow_steps, method=args.method,
                    )
                    x1 = torch.cat([wrist_x1, finger_x1], dim=-1)  # (B, 54)
                else:
                    x0 = torch.randn(B, flow_input_dim, device=device)
                    x1_raw_or_latent = flow.sample(
                        velocity_callable,
                        x0,
                        condition=condition_for_sample,
                        num_steps=args.num_flow_steps,
                        method=args.method,
                    )
                    # If latent flow, decode latent → raw MANO params
                    if latent_ae is not None:
                        z = x1_raw_or_latent  # (B, latent_dim)
                        mano_normed = latent_ae.decode(z)
                        x1 = mano_normed * latent_std + latent_mean  # denormalize
                    else:
                        x1 = x1_raw_or_latent
                decoded = mano_decoder(x1)
                pred_verts = decoded["vertices"]  # (B, 778, 3)
                pred_joints = decoded["joints"]   # (B, 21, 3)

                # GT vertices in world frame for the "true" mpvpe.
                gt_verts = batch["gt_world_verts"]  # (B, 778, 3)
                err_per_sample_mm = (pred_verts - gt_verts).norm(dim=-1).mean(dim=-1) * 1000.0

                if args.rank_by == "oracle":
                    rank_score = err_per_sample_mm
                elif args.rank_by == "contact_align":
                    tips = pred_joints.index_select(dim=1, index=finger_tip_joints)  # (B, 5, 3)
                    rank_score = (
                        (tips - rank_target_finger_centroid).norm(dim=-1).mean(dim=-1) * 1000.0
                    )
                elif args.rank_by == "penetration":
                    rank_score = nearest_point_penetration_mm(
                        pred_verts, batch["object_points"], batch["object_normals"]
                    )
                elif args.rank_by == "composite":
                    tips = pred_joints.index_select(dim=1, index=finger_tip_joints)
                    contact_score_mm = (
                        (tips - rank_target_finger_centroid).norm(dim=-1).mean(dim=-1) * 1000.0
                    )
                    pen_score_mm = nearest_point_penetration_mm(
                        pred_verts, batch["object_points"], batch["object_normals"]
                    )
                    pose = x1[:, 9:54]
                    over = (pose - 1.5).clamp_min(0.0)
                    under = (-1.5 - pose).clamp_min(0.0)
                    joint_score = (over.pow(2) + under.pow(2)).mean(dim=-1)
                    rank_score = (
                        w_contact * contact_score_mm
                        + w_pen * pen_score_mm
                        + w_joint * joint_score
                    )
                elif args.rank_by == "learned_selector":
                    # Compute the 8 scalar features (always).
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
                        feat_parts = [scalar_feats, pred_joints.reshape(pred_joints.shape[0], -1)]
                    elif selector_input_dim == 8 + 54 + 63:
                        feat_parts = [scalar_feats, x1, pred_joints.reshape(pred_joints.shape[0], -1)]
                    else:
                        raise RuntimeError(f"unknown selector input_dim={selector_input_dim}")
                    feats_row = torch.cat(feat_parts, dim=-1)
                    feats_norm = (feats_row - selector_mu) / selector_sigma
                    rank_score = selector_model(feats_norm)
                else:
                    raise ValueError(f"unknown --rank-by {args.rank_by}")

                better = rank_score < best_rank_score
                best_rank_score = torch.where(better, rank_score, best_rank_score)
                best_err_at_pick_mm = torch.where(better, err_per_sample_mm, best_err_at_pick_mm)
                best_picked_x1 = torch.where(better.view(-1, 1), x1, best_picked_x1)

            # Optional TTO on the picked candidate.
            if args.tto_steps > 0:
                # The scorer target (predicted finger centroids) must come from the
                # frozen head at inference time — not from the teacher-forced graph —
                # because TTO is meant to emulate the real inference-time path.
                if args.teacher_forcing:
                    head_for_tto = contact_head(
                        object_points=batch["object_points"],
                        object_normals=batch["object_normals"],
                        stage1_contact_input=batch["stage1_contact_input"],
                        m2ae_global=batch["m2ae_global"],
                        m2ae_local=batch["m2ae_local"],
                        patch_centers=batch["patch_centers"],
                    )
                    tto_target_centroid = head_for_tto["graph"]["finger_centroid"]
                else:
                    tto_target_centroid = graph["finger_centroid"]

                from graspauto.tto import full_params_tto  # noqa: PLC0415
                tto_out = full_params_tto(
                    best_picked_x1,
                    target_finger_centroids=tto_target_centroid,
                    object_points=batch["object_points"],
                    object_normals=batch["object_normals"],
                    decoder=mano_decoder,
                    num_steps=args.tto_steps,
                    lr=args.tto_lr,
                    w_contact=args.tto_w_contact,
                    w_surface=args.tto_w_surface,
                    w_pen=args.tto_w_pen,
                    w_prior=args.tto_w_prior,
                )
                # Re-decode and compute post-TTO mpvpe
                with torch.no_grad():
                    refined_out = mano_decoder(tto_out.refined_params)
                    refined_verts = refined_out["vertices"]
                    gt_verts_final = batch["gt_world_verts"]
                    post_err_mm = (refined_verts - gt_verts_final).norm(dim=-1).mean(dim=-1) * 1000.0
                # Overwrite the reported mpvpe with the post-TTO one.
                best_err_at_pick_mm = post_err_mm

            # Mode-coverage metric: distance to the nearest GT grasp of the same object.
            if args.mode_coverage:
                # Need the predicted vertices at the picked candidate. Re-decode
                # best_picked_x1 (or use refined params if TTO ran).
                final_params_for_mc = tto_out.refined_params if args.tto_steps > 0 else best_picked_x1
                final_verts = mano_decoder(final_params_for_mc)["vertices"]  # (B, 778, 3)
                mode_cover_mm = torch.zeros(B, device=device)
                for i in range(B):
                    oid = int(batch["object_id"][i].item())
                    if oid in object_id_to_gt_verts:
                        pool = object_id_to_gt_verts[oid]  # (K, 778, 3)
                        # min over the pool of mean-per-vertex L2 distance
                        diffs = pool - final_verts[i].unsqueeze(0)  # (K, 778, 3)
                        per_ref_mm = diffs.norm(dim=-1).mean(dim=-1) * 1000.0  # (K,)
                        mode_cover_mm[i] = per_ref_mm.min()
                    else:
                        mode_cover_mm[i] = best_err_at_pick_mm[i]

            for i in range(B):
                rec = {
                    "sample_index": int(batch["sample_index"][i].item()),
                    "object_id": int(batch["object_id"][i].item()),
                    "mpvpe_mm": float(best_err_at_pick_mm[i].item()),
                    "rank_score": float(best_rank_score[i].item()),
                    # Save picked x1 for downstream meta-selector
                    "picked_x1": best_picked_x1[i].detach().cpu().tolist(),
                }
                if args.mode_coverage:
                    rec["mode_cover_mm"] = float(mode_cover_mm[i].item())
                per_sample_records.append(rec)

    elapsed = time.time() - t0
    err_arr = torch.tensor([r["mpvpe_mm"] for r in per_sample_records])
    summary = {
        "checkpoint": str(args.checkpoint),
        "val_samples_evaluated": len(per_sample_records),
        "num_samples_per_cond": args.num_samples_per_cond,
        "num_flow_steps": args.num_flow_steps,
        "method": args.method,
        "elapsed_sec": round(elapsed, 3),
        "mpvpe_mm": {
            "mean": float(err_arr.mean().item()),
            "median": float(err_arr.median().item()),
            "p10": float(err_arr.quantile(0.10).item()),
            "p90": float(err_arr.quantile(0.90).item()),
            "min": float(err_arr.min().item()),
            "max": float(err_arr.max().item()),
        },
    }
    if args.mode_coverage and "mode_cover_mm" in per_sample_records[0]:
        mc_arr = torch.tensor([r["mode_cover_mm"] for r in per_sample_records])
        summary["mode_cover_mm"] = {
            "mean": float(mc_arr.mean().item()),
            "median": float(mc_arr.median().item()),
            "p10": float(mc_arr.quantile(0.10).item()),
            "p90": float(mc_arr.quantile(0.90).item()),
            "min": float(mc_arr.min().item()),
            "max": float(mc_arr.max().item()),
            "frac_under_10mm": float((mc_arr < 10.0).float().mean().item()),
            "frac_under_20mm": float((mc_arr < 20.0).float().mean().item()),
            "frac_under_30mm": float((mc_arr < 30.0).float().mean().item()),
        }

    print("\n=== graspauto eval summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[done] in {elapsed:.1f}s on {len(per_sample_records)} samples", flush=True)

    # ---- Save ----
    if args.out_dir is None:
        args.out_dir = args.checkpoint.parent / f"eval_{int(time.time())}"
    ensure_dir(args.out_dir)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "per_sample.json", {"records": per_sample_records})
    print(f"[saved] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
