#!/usr/bin/env python3
"""Train a small MLP that predicts oracle mpvpe from candidate features.

Consumes the cache produced by `build_selector_data.py`:
  train.pt: {"features": (N_train, F), "targets_mm": (N_train,), ...}
  val.pt:   same format.

The selector is a tiny MLP (F → 64 → 64 → 1) trained with L1 loss
(smooth for outliers) on normalized features. At inference time, each
best-of-N candidate is passed through this selector and the one with
the smallest predicted mpvpe is picked.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from graspauto.utils import ensure_dir, resolve_device  # noqa: E402


class SelectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, F) → (B,) predicted mpvpe_mm."""
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("outputs/graspauto_selector_data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/graspauto_selector"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--feature-set", type=str, default="scalar",
                   choices=["scalar", "scalar_plus_x1", "scalar_plus_joints", "all"],
                   help="Which features to train on. 'scalar' = the 8 hand-coded features only. "
                        "'scalar_plus_x1' adds the raw 54-D x1. 'scalar_plus_joints' adds the flattened "
                        "21x3=63 joint positions. 'all' adds both.")
    p.add_argument("--loss", type=str, default="smooth_l1",
                   choices=["smooth_l1", "pairwise"],
                   help="'smooth_l1' regresses the mpvpe_mm value directly. "
                        "'pairwise' uses a margin ranking loss over (i, j) candidate pairs from the "
                        "same training sample, teaching the selector to rank candidates consistently "
                        "regardless of absolute magnitude. Often better for best-of-N selection.")
    p.add_argument("--pairwise-margin", type=float, default=1.0,
                   help="Margin parameter for pairwise ranking loss (mm).")
    return p.parse_args()


def assemble_features(cache: dict, feature_set: str) -> tuple[torch.Tensor, list[str]]:
    scalar = cache["features"]  # (N, 8)
    names = list(cache["feature_names"])
    parts = [scalar]
    if feature_set in ("scalar_plus_x1", "all"):
        parts.append(cache["raw_x1"])  # (N, 54)
        names += [f"x1_{i}" for i in range(54)]
    if feature_set in ("scalar_plus_joints", "all"):
        joints = cache["raw_joints"].reshape(cache["raw_joints"].shape[0], -1)  # (N, 63)
        parts.append(joints)
        names += [f"j_{i}" for i in range(63)]
    return torch.cat(parts, dim=-1), names


def select_accuracy_topk(features: torch.Tensor,
                          targets: torch.Tensor,
                          candidate_index: torch.Tensor,
                          sample_index: torch.Tensor,
                          model: SelectorMLP,
                          mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, float]:
    """For each unique sample, compute mpvpe at top-1 ranked by the selector
    and compare to oracle top-1 and the mean over candidates."""
    device = features.device
    with torch.no_grad():
        f_norm = (features - mu) / sigma
        preds = model(f_norm)
    unique_samples = sample_index.unique()
    selector_err_mm = []
    oracle_err_mm = []
    single_mean_err_mm = []
    for s in unique_samples:
        mask = sample_index == s
        target_s = targets[mask]
        pred_s = preds[mask]
        selector_err_mm.append(target_s[pred_s.argmin()].item())
        oracle_err_mm.append(target_s.min().item())
        single_mean_err_mm.append(target_s.mean().item())
    return {
        "selector_mean_mpvpe_mm": sum(selector_err_mm) / len(selector_err_mm),
        "oracle_mean_mpvpe_mm": sum(oracle_err_mm) / len(oracle_err_mm),
        "random_mean_mpvpe_mm": sum(single_mean_err_mm) / len(single_mean_err_mm),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_cache = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=False)
    val_cache = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=False)

    train_feats_cpu, feature_names = assemble_features(train_cache, args.feature_set)
    val_feats_cpu, _ = assemble_features(val_cache, args.feature_set)
    F_dim = train_feats_cpu.shape[-1]
    print(f"[data] train rows {train_feats_cpu.shape[0]}, val rows {val_feats_cpu.shape[0]}, feature_set={args.feature_set}, F={F_dim}", flush=True)

    train_feats = train_feats_cpu.to(device)
    train_targets = train_cache["targets_mm"].to(device)
    train_sidx = train_cache["sample_index"].to(device)
    val_feats = val_feats_cpu.to(device)
    val_targets = val_cache["targets_mm"].to(device)
    val_sidx = val_cache["sample_index"].to(device)

    # Normalize features using train mean/std
    mu = train_feats.mean(dim=0)
    sigma = train_feats.std(dim=0).clamp_min(1e-6)
    train_norm = (train_feats - mu) / sigma

    model = SelectorMLP(in_dim=F_dim, hidden=args.hidden_dim, dropout=args.dropout, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[model] SelectorMLP {sum(p.numel() for p in model.parameters())} params", flush=True)

    # For pairwise loss, precompute the mapping from sample_index to the row
    # indices in that sample (so we can sample candidate pairs from the same sample).
    # We also make a python-side grouping for fast batching.
    if args.loss == "pairwise":
        unique_samples = train_sidx.unique()
        sample_to_rows: dict[int, list[int]] = {}
        for s in unique_samples.tolist():
            mask = (train_sidx == s).nonzero(as_tuple=False).view(-1)
            sample_to_rows[s] = mask.tolist()
        sample_ids_list = list(sample_to_rows.keys())
        print(
            f"[pairwise] {len(sample_ids_list)} unique train samples, "
            f"{sum(len(v) for v in sample_to_rows.values()) // len(sample_ids_list)} candidates each on avg",
            flush=True,
        )

    n_train = train_norm.shape[0]
    best_selector_mm = float("inf")
    out_dir = ensure_dir(args.out_dir)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.loss == "smooth_l1":
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, args.batch_size):
                idx = perm[start:start + args.batch_size]
                f_b = train_norm[idx]
                t_b = train_targets[idx]
                pred = model(f_b)
                loss = F.smooth_l1_loss(pred, t_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        else:
            # Pairwise margin ranking. For each training step, sample args.batch_size
            # samples, pick two random candidates from each, and apply a margin loss
            # so the candidate with lower mpvpe scores lower.
            # Repeat enough to roughly match one regression epoch in updates.
            num_steps = max(1, n_train // args.batch_size)
            for _ in range(num_steps):
                import random
                sel = random.sample(sample_ids_list, min(args.batch_size, len(sample_ids_list)))
                idx_a = []
                idx_b = []
                for s in sel:
                    rows = sample_to_rows[s]
                    if len(rows) < 2:
                        continue
                    r = random.sample(rows, 2)
                    idx_a.append(r[0])
                    idx_b.append(r[1])
                if not idx_a:
                    continue
                ta = torch.tensor(idx_a, device=device, dtype=torch.long)
                tb = torch.tensor(idx_b, device=device, dtype=torch.long)
                fa = train_norm[ta]
                fb = train_norm[tb]
                ya = train_targets[ta]
                yb = train_targets[tb]
                sa = model(fa)
                sb = model(fb)
                # We want score_i < score_j iff target_i < target_j.
                # sign = sign(ya - yb); positive means a is worse (higher mpvpe).
                # In that case we want sa > sb (score of a higher than b), so
                # sign * (sa - sb) should be positive (and >= margin).
                sign = torch.sign(ya - yb)
                loss = F.relu(args.pairwise_margin - sign * (sa - sb)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        train_loss = sum(losses) / max(1, len(losses))

        model.eval()
        with torch.no_grad():
            val_pred = model((val_feats - mu) / sigma)
            val_l1 = (val_pred - val_targets).abs().mean().item()

        # Compute top-1 select accuracy vs oracle
        val_metrics = select_accuracy_topk(
            val_feats, val_targets, val_cache["candidate_index"].to(device), val_sidx, model, mu, sigma
        )
        selector_mm = val_metrics["selector_mean_mpvpe_mm"]
        oracle_mm = val_metrics["oracle_mean_mpvpe_mm"]
        random_mm = val_metrics["random_mean_mpvpe_mm"]
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_smoothL1={train_loss:.3f} val_L1={val_l1:.2f}mm  "
            f"select_mpvpe={selector_mm:.2f}mm oracle={oracle_mm:.2f}mm random_mean={random_mm:.2f}mm",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_smooth_l1": train_loss,
            "val_l1_mm": val_l1,
            "val_selector_mean_mpvpe_mm": selector_mm,
            "val_oracle_mean_mpvpe_mm": oracle_mm,
            "val_random_mean_mpvpe_mm": random_mm,
        })

        if selector_mm < best_selector_mm:
            best_selector_mm = selector_mm
            torch.save({
                "model_state": model.state_dict(),
                "mu": mu.cpu(),
                "sigma": sigma.cpu(),
                "feature_names": feature_names,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_epoch": epoch,
                "best_selector_mean_mpvpe_mm": selector_mm,
                "args": vars(args),
            }, out_dir / "best.pt")

    import json
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] best val selector mean mpvpe = {best_selector_mm:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
