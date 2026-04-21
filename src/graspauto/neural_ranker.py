"""Neural ranker with object conditioning.

Scores candidate grasps based on (hand pose, object features, tap point).
Unlike scalar-feature selectors, this sees the object context explicitly,
so it can learn "this hand pose is well-aligned FOR THIS object+tap"
rather than only "this hand is physically plausible".

Input:
    x1:            (B, 54) MANO params (rot_6d + trans + hand_pose)
    joints:        (B, 21, 3) predicted joints
    m2ae_local:    (B, 64, 384) Point-M2AE patch tokens of the object
    palm_centroid: (B, 3) user tap point (= unified_centroid in unified mode)

Output:
    score:         (B,) predicted mpvpe in mm (lower = better)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralRanker(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Hand encoder: x1 (54) + joints flat (63) + (optional) 8-scalar feats
        self.hand_mlp = nn.Sequential(
            nn.Linear(54 + 63 + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Object patch token projection
        self.obj_proj = nn.Linear(384, hidden_dim)
        # Tap point projection
        self.tap_proj = nn.Linear(3, hidden_dim)
        # Cross-attention: hand query → obj+tap keys/values
        self.xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=0.0,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x1, joints, scalar_feats, m2ae_local, palm_centroid):
        """
        x1: (B, 54)
        joints: (B, 21, 3)
        scalar_feats: (B, 8) — hand-crafted physics features (contact/pen/etc.)
        m2ae_local: (B, 64, 384)
        palm_centroid: (B, 3)
        Returns: (B,) scores (lower = better)
        """
        B = x1.shape[0]
        joints_flat = joints.reshape(B, -1)  # (B, 63)
        hand_in = torch.cat([x1, joints_flat, scalar_feats], dim=-1)
        hand_feat = self.hand_mlp(hand_in)  # (B, H)

        obj_tokens = self.obj_proj(m2ae_local)  # (B, 64, H)
        tap_token = self.tap_proj(palm_centroid).unsqueeze(1)  # (B, 1, H)
        ctx = torch.cat([obj_tokens, tap_token], dim=1)  # (B, 65, H)

        q = self.norm_q(hand_feat).unsqueeze(1)  # (B, 1, H)
        kv = self.norm_kv(ctx)
        attn_out, _ = self.xattn(q, kv, kv)  # (B, 1, H)

        fused = hand_feat + attn_out.squeeze(1)
        score = self.head(fused).squeeze(-1)
        return score


__all__ = ["NeuralRanker"]
