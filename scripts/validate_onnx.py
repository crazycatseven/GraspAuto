#!/usr/bin/env python3
"""Validate that the exported ONNX models match their PyTorch reference.

Purpose: before shipping the 4 ONNX files to Unity Sentis on Quest 3, make sure
the numerical outputs agree with the original PyTorch checkpoints to within a
small tolerance. If they drift, Unity will produce garbage grasps — we want to
catch that here, not in VR.

What this script does:
  1. Loads the r047 PyTorch checkpoint (adapter + velocity_net), the joint AE,
     and the Point-M2AE encoder.
  2. Loads the four ONNX files at outputs/onnx_sphere_r047/*.onnx via
     onnxruntime.
  3. Feeds a real ContactPose val sample through both pipelines (adapter
     bundle, velocity at t=0.5, AE decode, Point-M2AE encode).
  4. Prints max / mean abs diff and sets a pass/fail flag per model
     (tolerance = 1e-4).
  5. Saves the reference inputs and expected outputs as float32 `.npy` files
     under `artifacts/onnx_validation/` so the Unity-Sentis port can load the
     same bytes, run inference, and sanity-check its own outputs against ours.

Usage:
    python scripts/validate_onnx.py \\
        --ckpt-dir outputs/graspauto_sphere_r047 \\
        --ae-ckpt outputs/graspauto_ae_joint/best.pt \\
        --onnx-dir outputs/onnx_sphere_r047 \\
        --val outputs/stage3_contact_graph/val_sphere.pt \\
        --cache outputs/stage3_contact_graph/object_m2ae_cache.pt \\
        --val-idx 47 \\
        --out artifacts/onnx_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

try:
    import onnxruntime as ort  # noqa: E402
except ImportError:
    sys.exit("onnxruntime is required: pip install onnxruntime")

from graspauto.conditioning import (  # noqa: E402
    ContactGraphConditioningAdapter,
    compute_patch_weight_from_point,
)
from graspauto.flow_matching import ConditionalFlowMatching  # noqa: E402
from graspauto.mano_autoencoder import ResidualMANOAutoEncoder  # noqa: E402
from graspauto.mano_decoder import MangoMANODecoder  # noqa: E402
from graspauto.velocity_network import VelocityNetwork  # noqa: E402
from graspauto.point_m2ae_encoder import PointM2AEObjectEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# Tolerances. ONNX export introduces per-op round-off on the order of 1e-6,
# compounded by ~10 attention / MLP layers the end-to-end CFM latent at t=1
# can drift by ~1e-4. Set per-model caps that catch bugs but not numerical
# noise.
TOLERANCES = {
    "adapter":      1e-4,
    "velocity_net": 1e-4,
    "ae_decoder":   1e-4,
    "point_m2ae":   5e-3,   # larger backbone, more accumulated noise
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=Path("outputs/graspauto_sphere_r047"))
    p.add_argument("--ae-ckpt", type=Path, default=Path("outputs/graspauto_ae_joint/best.pt"))
    p.add_argument("--onnx-dir", type=Path, default=Path("outputs/onnx_sphere_r047"))
    p.add_argument("--val", type=Path, default=Path("outputs/stage3_contact_graph/val_sphere.pt"))
    p.add_argument("--cache", type=Path,
                   default=Path("outputs/stage3_contact_graph/object_m2ae_cache.pt"))
    p.add_argument("--m2ae-weights", type=Path,
                   default=Path("external/pretrained/point_m2ae_pretrain.pth"))
    p.add_argument("--val-idx", type=int, default=47, help="val sample index to use")
    p.add_argument("--out", type=Path, default=Path("artifacts/onnx_validation"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def summarize(name: str, torch_out: np.ndarray, onnx_out: np.ndarray,
              tolerance: float) -> dict:
    assert torch_out.shape == onnx_out.shape, (
        f"[{name}] shape mismatch: torch {torch_out.shape} vs onnx {onnx_out.shape}"
    )
    diff = np.abs(torch_out.astype(np.float64) - onnx_out.astype(np.float64))
    rel = diff / (np.abs(torch_out.astype(np.float64)) + 1e-9)
    stats = {
        "name":        name,
        "shape":       list(torch_out.shape),
        "max_abs":     float(diff.max()),
        "mean_abs":    float(diff.mean()),
        "max_rel":     float(rel.max()),
        "mean_rel":    float(rel.mean()),
        "torch_norm":  float(np.linalg.norm(torch_out)),
        "onnx_norm":   float(np.linalg.norm(onnx_out)),
        "tolerance":   tolerance,
        "passed":      bool(diff.max() <= tolerance),
    }
    flag = "PASS" if stats["passed"] else "FAIL"
    print(f"  [{flag}] {name:14s} max_abs={stats['max_abs']:.3e}"
          f"  mean_abs={stats['mean_abs']:.3e}  shape={stats['shape']}")
    return stats


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---------------- load val sample --------------------------------------
    print("loading val sample + object cache...")
    val = torch.load(args.val, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    idx = args.val_idx
    obj_id = int(val["object_id"][idx])

    # Apply the sphere-intent transformation (same as eval_graspauto.py).
    unified_centroid = val["unified_centroid"][idx:idx + 1]
    unified_spread = val["unified_spread"][idx:idx + 1]
    unified_entropy = val["unified_entropy"][idx:idx + 1]
    hTm_rot = val["hTm_rot"][idx:idx + 1]
    fwd_local = torch.tensor([0.0, 0.0, 1.0], dtype=hTm_rot.dtype)
    palm_normal = hTm_rot @ fwd_local
    radius = unified_spread.norm(dim=-1, keepdim=True).expand(-1, 3).contiguous()
    palm_mass = torch.zeros(1)

    m2ae_local = cache["m2ae_local"][obj_id:obj_id + 1].to(device)
    patch_centers = cache["patch_centers"][obj_id:obj_id + 1].to(device)
    active_finger_prob = val["active_finger_score"][idx:idx + 1].to(device)

    # ---------------- PyTorch reference ------------------------------------
    print("loading PyTorch reference models...")
    ck = torch.load(args.ckpt_dir / "best.pt", map_location=device, weights_only=False)
    hidden = int(ck["args"].get("hidden_dim", 256))
    n_heads = int(ck["args"].get("n_heads", 4))
    n_layers = int(ck["args"].get("n_layers", 4))
    adapter = ContactGraphConditioningAdapter(
        hidden_dim=hidden, use_intent_token=False,
        part_aware_gating=True, palm_only_intent=True,
    ).to(device).eval()
    adapter.load_state_dict(dict(ck["adapter_state"]), strict=False)

    velnet = VelocityNetwork(
        input_dim=32, hidden_dim=hidden, n_heads=n_heads, n_layers=n_layers,
    ).to(device).eval()
    velnet.load_state_dict(ck["velocity_net_state"])

    ae_ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae = ResidualMANOAutoEncoder(
        input_dim=54, latent_dim=ae_ckpt["latent_dim"],
        hidden_dim=ae_ckpt["res_hidden_dim"], n_blocks=ae_ckpt["res_n_blocks"],
    ).to(device).eval()
    ae.load_state_dict(ae_ckpt["model_state"])

    # ---------------- Adapter ----------------------------------------------
    # Mirror `scripts/export_onnx.py::AdapterONNXWrapper.forward` so the
    # reference matches what was traced into the ONNX: sigma=0.03 gaussian
    # gate, zero active_finger_prob, palm_only_intent signature.
    print("\n[1/4] adapter.onnx")
    sigma = 0.03
    offs = patch_centers - unified_centroid.to(device).unsqueeze(1)
    sqd = offs.pow(2).sum(dim=-1)
    pcw_exact = torch.exp(-sqd / (2.0 * sigma * sigma)).clamp(min=0.0, max=1.0)
    active_finger_prob_zero = torch.zeros(1, 5, device=device)
    with torch.no_grad():
        graph = {
            "palm_centroid":   unified_centroid.to(device),
            "palm_normal":     palm_normal.to(device),
            "palm_spread":     radius.to(device),
            "palm_entropy":    unified_entropy.to(device),
            "palm_mass":       palm_mass.to(device),
            "finger_centroid": torch.zeros(1, 5, 3, device=device),
            "finger_normal":   torch.zeros(1, 5, 3, device=device),
            "finger_spread":   torch.zeros(1, 5, 3, device=device),
            "finger_entropy":  torch.zeros(1, 5, device=device),
            "finger_mass":     torch.zeros(1, 5, device=device),
            "unified_centroid": unified_centroid.to(device),
            "unified_normal":  palm_normal.to(device),
            "unified_spread":  radius.to(device),
            "unified_entropy": unified_entropy.to(device),
        }
        bundle = adapter(
            m2ae_local=m2ae_local, graph=graph,
            active_finger_prob=active_finger_prob_zero,
            patch_contact_weight=pcw_exact,
        )
    torch_bundle = bundle.tokens.detach().cpu().numpy()

    sess_adapter = ort.InferenceSession(str(args.onnx_dir / "adapter.onnx"))
    adapter_inputs = {
        "m2ae_local":    m2ae_local.cpu().numpy(),
        "patch_centers": patch_centers.cpu().numpy(),
        "tap_point":     unified_centroid.cpu().numpy(),
        "palm_normal":   palm_normal.cpu().numpy(),
        "palm_spread":   radius.cpu().numpy(),
        "palm_entropy":  unified_entropy.cpu().numpy(),
        "palm_mass":     palm_mass.cpu().numpy(),
    }
    onnx_bundle = sess_adapter.run(None, adapter_inputs)[0]
    adapter_stats = summarize("adapter", torch_bundle, onnx_bundle, TOLERANCES["adapter"])

    # ---------------- Velocity net -----------------------------------------
    print("\n[2/4] velocity_net.onnx")
    xt = torch.randn(1, 32, device=device, generator=torch.Generator(device=device).manual_seed(0))
    t = torch.tensor([0.5], device=device)
    with torch.no_grad():
        torch_vel = velnet(xt, t, condition=bundle).detach().cpu().numpy()
    sess_velnet = ort.InferenceSession(str(args.onnx_dir / "velocity_net.onnx"))
    velnet_inputs = {
        "xt":            xt.cpu().numpy(),
        "t":             t.cpu().numpy(),
        "bundle_tokens": torch_bundle,   # use the PyTorch adapter output here
    }
    onnx_vel = sess_velnet.run(None, velnet_inputs)[0]
    velnet_stats = summarize("velocity_net", torch_vel, onnx_vel, TOLERANCES["velocity_net"])

    # ---------------- AE decoder -------------------------------------------
    # Match `scripts/export_onnx.py::AEDecoderONNXWrapper.forward` — the ONNX
    # bakes the denormalisation (train_mean, train_std) into the graph, so
    # the PyTorch reference must apply it too.
    print("\n[3/4] ae_decoder.onnx")
    lm = ae_ckpt["train_mean"].to(device)
    ls = ae_ckpt["train_std"].to(device)
    z = torch.randn(1, 32, device=device, generator=torch.Generator(device=device).manual_seed(1))
    with torch.no_grad():
        mano_norm = ae.decode(z)
        torch_mano = (mano_norm * ls + lm).detach().cpu().numpy()
    sess_ae = ort.InferenceSession(str(args.onnx_dir / "ae_decoder.onnx"))
    onnx_mano = sess_ae.run(None, {"z": z.cpu().numpy()})[0]
    ae_stats = summarize("ae_decoder", torch_mano, onnx_mano, TOLERANCES["ae_decoder"])

    # ---------------- Point-M2AE (Mode B only) -----------------------------
    point_m2ae_path = args.onnx_dir / "point_m2ae.onnx"
    m2ae_stats = None
    if point_m2ae_path.exists():
        print("\n[4/4] point_m2ae.onnx")

        # The ONNX export monkey-patches FPS to a deterministic (index-0 start)
        # variant; the default PyTorch FPS picks a random start. Patch the same
        # way here so PyTorch and ONNX take the same 2048-point subset.
        import graspauto.point_m2ae_encoder as _m2ae_mod
        _orig_fps = _m2ae_mod.fps_pytorch

        def _fps_det(xyz, npoint):
            B, N, _ = xyz.shape
            dv = xyz.device
            centroids = torch.zeros(B, npoint, dtype=torch.long, device=dv)
            distance = torch.full((B, N), 1e10, device=dv)
            farthest = torch.zeros(B, dtype=torch.long, device=dv)
            batch_indices = torch.arange(B, dtype=torch.long, device=dv)
            for i in range(npoint):
                centroids[:, i] = farthest
                centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
                dist = torch.sum((xyz - centroid) ** 2, dim=-1)
                distance = torch.min(distance, dist)
                farthest = torch.max(distance, dim=-1)[1]
            return centroids

        _m2ae_mod.fps_pytorch = _fps_det

        if args.m2ae_weights.exists():
            encoder = PointM2AEObjectEncoder(
                pretrained_path=str(args.m2ae_weights),
            ).to(device).eval()
        else:
            encoder = None
            print("  [skip] no pretrained M2AE weights; running ONNX-only consistency check")

        # The ONNX export freezes the number of points at 2048; the geometry
        # cache has 3000, so take the first 2048 (already FPS-sampled upstream).
        geom = torch.load(
            "outputs/contact_vqvae_stage1_v16_film/cache/geometry_cache.pt",
            map_location="cpu", weights_only=False,
        )
        obj_pc_full = geom["object_points"][obj_id:obj_id + 1].to(device)
        obj_pc = obj_pc_full[:, :2048, :]
        sess_m2ae = ort.InferenceSession(str(point_m2ae_path))
        onnx_outs = sess_m2ae.run(None, {"obj_pc": obj_pc.cpu().numpy()})
        onnx_local = onnx_outs[1]  # m2ae_local per ONNX metadata ordering

        if encoder is not None:
            with torch.no_grad():
                # forward_local returns (global [B,1024], local [B,64,384], centers [B,64,3])
                _, torch_local_t, _ = encoder.forward_local(obj_pc)
            torch_local = torch_local_t.cpu().numpy()
            m2ae_stats = summarize("point_m2ae", torch_local, onnx_local,
                                   TOLERANCES["point_m2ae"])
        else:
            # Just compare against cached features (sanity, not a strict tolerance)
            cached = cache["m2ae_local"][obj_id:obj_id + 1].cpu().numpy()
            m2ae_stats = summarize("point_m2ae (vs cache)", cached, onnx_local,
                                   TOLERANCES["point_m2ae"] * 5)
    else:
        print("\n[4/4] point_m2ae.onnx — file missing, skipping")

    # ---------------- Save artifacts ---------------------------------------
    print(f"\nsaving reference tensors for Unity Sentis validation at {args.out}/ ...")
    np.savez(args.out / "adapter_io.npz", **adapter_inputs, output=torch_bundle)
    np.savez(args.out / "velocity_net_io.npz", **velnet_inputs, output=torch_vel)
    np.savez(args.out / "ae_decoder_io.npz", z=z.cpu().numpy(), output=torch_mano)
    if m2ae_stats is not None and m2ae_stats.get("passed"):
        np.savez(args.out / "point_m2ae_io.npz",
                 obj_pc=obj_pc.cpu().numpy(), output=torch_local)

    summary = {
        "val_idx":      args.val_idx,
        "obj_id":       obj_id,
        "tolerances":   TOLERANCES,
        "results":      {
            "adapter":      adapter_stats,
            "velocity_net": velnet_stats,
            "ae_decoder":   ae_stats,
            "point_m2ae":   m2ae_stats,
        },
    }
    with open(args.out / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    all_pass = all(
        s is None or s.get("passed", False)
        for s in summary["results"].values()
    )
    print("\n" + ("ALL MODELS PASS" if all_pass else "SOME MODELS FAILED") + ". "
          f"Details: {args.out / 'validation_summary.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
