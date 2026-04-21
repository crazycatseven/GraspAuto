"""
Point-M2AE encoder adapter for VR Grasp prior.

Standalone reimplementation of Point-M2AE's hierarchical encoder,
using pure PyTorch (no knn_cuda / custom CUDA extensions).
Loads official pre-trained weights from Point-M2AE checkpoint.

Architecture (from paper):
- 3 hierarchical levels: 512→256→64 groups
- Group sizes: 16, 8, 8
- Encoder dims: 96→192→384
- Local self-attention with radius masking
- Output: 384D per-token features at final level (64 tokens)

For grasp prior: mean+max pool → 384D → adapter → 1024D
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ============================================================
# Pure PyTorch replacements for knn_cuda / misc.fps
# ============================================================

def fps_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling (pure PyTorch).
    
    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample
    Returns:
        centroids: (B, npoint) indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids


def knn_pytorch(xyz: torch.Tensor, center: torch.Tensor, k: int) -> torch.Tensor:
    """K-Nearest Neighbors (pure PyTorch).
    
    Args:
        xyz: (B, N, 3) all points
        center: (B, G, 3) query centers
        k: number of neighbors
    Returns:
        idx: (B, G, k) neighbor indices
    """
    dist = torch.cdist(center, xyz, p=2)  # (B, G, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, G, k)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points by batch indices.
    
    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, k)
    Returns:
        indexed: (B, S, C) or (B, S, k, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# ============================================================
# Core modules (from Point-M2AE, adapted for pure PyTorch)
# ============================================================

class Token_Embed(nn.Module):
    """Point patch embedding via mini-PointNet.
    
    Matches official Point-M2AE Token_Embed exactly:
    - in_c == 3: hidden 128→256, concat→512→out_c
    - in_c != 3: hidden in_c→in_c, concat→in_c*2→out_c
    """
    def __init__(self, in_c: int = 3, out_c: int = 96):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1),
            )
        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1),
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1),
            )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_groups: (B, G, K, in_c)
        Returns:
            tokens: (B, G, out_c)
        """
        B, G, K, C = point_groups.shape
        x = point_groups.reshape(B * G, K, C).permute(0, 2, 1)  # (BG, C, K)
        feature = self.first_conv(x)  # (BG, hidden, K)
        feature_global = feature.max(dim=-1, keepdim=True)[0]  # (BG, hidden, 1)
        feature = torch.cat([feature_global.expand(-1, -1, K), feature], dim=1)  # (BG, hidden*2, K)
        feature = self.second_conv(feature)  # (BG, out_c, K)
        feature = feature.max(dim=-1)[0]  # (BG, out_c)
        return feature.reshape(B, G, self.out_c)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        if mask is not None:
            # mask: (B, N, N), True = masked out → additive masking (official style)
            attn = attn + mask.float().unsqueeze(1) * -100000.0
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP matching official Point-M2AE Mlp (fc1/fc2 naming)."""
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class EncoderBlock(nn.Module):
    """Stack of Transformer blocks for one hierarchical level."""
    def __init__(self, embed_dim: int, depth: int, num_heads: int, drop_path_rates: list[float]):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos, mask)
        return x


class Group(nn.Module):
    """FPS + KNN grouping (pure PyTorch)."""
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            neighborhood: (B, G, K, 3) local coordinates
            center: (B, G, 3) group centers
            idx: (B*G, K) flattened neighbor indices (for token merging)
        """
        B, N, _ = xyz.shape
        # FPS
        center_idx = fps_pytorch(xyz, self.num_group)  # (B, G)
        center = index_points(xyz, center_idx)  # (B, G, 3)
        # KNN
        idx = knn_pytorch(xyz, center, self.group_size)  # (B, G, K)
        # Gather neighborhoods
        neighborhood = index_points(xyz, idx)  # (B, G, K, 3)
        # Normalize to local coordinates
        neighborhood = neighborhood - center.unsqueeze(2)
        # Flatten idx for token merging compatibility
        idx_flat = idx  # keep (B, G, K) for now
        return neighborhood, center, idx_flat


class HierarchicalEncoder(nn.Module):
    """Point-M2AE Hierarchical Encoder (finetune mode, no masking)."""
    
    # Default config matching official pre-training
    DEFAULT_CONFIG = {
        'group_sizes': [16, 8, 8],
        'num_groups': [512, 256, 64],
        'encoder_depths': [5, 5, 5],
        'encoder_dims': [96, 192, 384],
        'local_radius': [0.32, 0.64, 1.28],
        'drop_path_rate': 0.1,
        'num_heads': 6,
    }

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or self.DEFAULT_CONFIG
        
        self.group_sizes = config['group_sizes']
        self.num_groups = config['num_groups']
        self.encoder_depths = config['encoder_depths']
        self.encoder_dims = config['encoder_dims']
        self.local_radius = config['local_radius']
        self.feat_dim = self.encoder_dims[-1]  # 384
        
        # Group dividers (FPS + KNN)
        self.group_dividers = nn.ModuleList([
            Group(num_group=self.num_groups[i], group_size=self.group_sizes[i])
            for i in range(len(self.group_sizes))
        ])
        
        # Token embeddings
        self.token_embed = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            in_c = 3 if i == 0 else self.encoder_dims[i - 1]
            self.token_embed.append(Token_Embed(in_c=in_c, out_c=self.encoder_dims[i]))
        
        # Positional embeddings
        self.encoder_pos_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, self.encoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
            )
            for i in range(len(self.encoder_dims))
        ])
        
        # Encoder blocks
        depth_count = 0
        total_depth = sum(self.encoder_depths)
        dpr = [x.item() for x in torch.linspace(0, config['drop_path_rate'], total_depth)]
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(EncoderBlock(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                num_heads=config['num_heads'],
                drop_path_rates=dpr[depth_count: depth_count + self.encoder_depths[i]],
            ))
            depth_count += self.encoder_depths[i]
        
        # Final norm
        self.norm = nn.LayerNorm(self.feat_dim)
    
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, N, 3) input point cloud (N >= 2048, will be FPS'd)
        Returns:
            features: (B, feat_dim) global feature vector
                      = mean + max pooling of final-level tokens
        """
        # Multi-scale grouping
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
        
        # Hierarchical encoding (finetune mode, no masking)
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                # Token merging: gather features by neighbor indices
                b, g1, C = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                # Use idx to gather previous-level features
                x_vis_gathered = index_points(x_vis, idxs[i])  # (B, G2, K2, C)
                group_input_tokens = self.token_embed[i](x_vis_gathered)
            
            # Local attention mask
            if self.local_radius[i] > 0:
                dist = torch.cdist(centers[i], centers[i], p=2)
                mask = dist >= self.local_radius[i]
            else:
                mask = None
            
            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask)
        
        # Final pooling
        x_vis = self.norm(x_vis)
        global_feat = x_vis.mean(dim=1) + x_vis.max(dim=1)[0]  # (B, 384)
        return global_feat
    
    def forward_local(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local tokens (for future cross-attention in prior).
        
        Returns:
            global_feat: (B, 384) 
            local_tokens: (B, 64, 384) final-level per-token features
        """
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
        
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b, g1, C = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_gathered = index_points(x_vis, idxs[i])
                group_input_tokens = self.token_embed[i](x_vis_gathered)
            
            if self.local_radius[i] > 0:
                dist = torch.cdist(centers[i], centers[i], p=2)
                mask = dist >= self.local_radius[i]
            else:
                mask = None
            
            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask)
        
        x_vis = self.norm(x_vis)
        global_feat = x_vis.mean(dim=1) + x_vis.max(dim=1)[0]
        return global_feat, x_vis, centers[-1]  # global (B,384), local (B,64,384), patch_centers (B,64,3)


class PointM2AEObjectEncoder(nn.Module):
    """Object encoder for VR Grasp prior using Point-M2AE backbone.
    
    Replaces PointNet with hierarchical Point-M2AE encoder.
    Output: 1024D feature vector (matching existing prior interface).
    """
    
    def __init__(
        self,
        output_dim: int = 1024,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = HierarchicalEncoder(config)
        feat_dim = self.backbone.feat_dim  # 384
        
        # Adapter: 384 → 1024
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def load_pretrained(self, ckpt_path: str):
        """Load official Point-M2AE pre-trained weights.
        
        The official checkpoint has keys like:
        - 'base_model' containing the full model state dict
        - We only need the encoder parts (h_encoder.*, group_dividers.*, token_embed.*)
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'base_model' in ckpt:
            state_dict = ckpt['base_model']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Map official keys to our keys
        # Official structure: h_encoder.encoder_blocks.{level}.blocks.{depth}.{attn/mlp/norm}
        # Our structure:      backbone.encoder_blocks.{level}.blocks.{depth}.{attn/mlp/norm}
        # The internal structure matches 1:1 (same Attention, Mlp, LayerNorm naming)
        mapped = {}
        for k, v in state_dict.items():
            new_k = None
            if k.startswith('h_encoder.token_embed.'):
                new_k = k.replace('h_encoder.token_embed.', 'backbone.token_embed.')
            elif k.startswith('h_encoder.encoder_pos_embeds.'):
                new_k = k.replace('h_encoder.encoder_pos_embeds.', 'backbone.encoder_pos_embeds.')
            elif k.startswith('h_encoder.encoder_blocks.'):
                new_k = k.replace('h_encoder.encoder_blocks.', 'backbone.encoder_blocks.')
            elif k.startswith('h_encoder.encoder_norms.'):
                # Pre-training has per-level norms; we only have final norm
                # Map the last level norm (level 2, 384D) to our backbone.norm
                if k.startswith('h_encoder.encoder_norms.2.'):
                    new_k = k.replace('h_encoder.encoder_norms.2.', 'backbone.norm.')
            
            if new_k is not None:
                mapped[new_k] = v
        
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        print(f"[PointM2AEObjectEncoder] Loaded {len(mapped)} weights from {ckpt_path}")
        if missing:
            # Filter out adapter (expected to be missing)
            real_missing = [k for k in missing if not k.startswith('adapter.')]
            if real_missing:
                print(f"  Missing backbone keys: {real_missing[:10]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:10]}...")
    
    def forward(self, obj_pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_pc: (B, N, 3) or (B, 3, N) object point cloud
        Returns:
            features: (B, 1024)
        """
        # Handle (B, 3, N) input format
        if obj_pc.shape[1] == 3 and obj_pc.shape[2] != 3:
            obj_pc = obj_pc.transpose(1, 2)  # → (B, N, 3)
        
        # Subsample to 2048 if needed
        B, N, _ = obj_pc.shape
        if N > 2048:
            idx = fps_pytorch(obj_pc, 2048)
            obj_pc = index_points(obj_pc, idx)
        
        global_feat = self.backbone(obj_pc)  # (B, 384)
        features = self.adapter(global_feat)  # (B, 1024)
        return features
    
    def forward_local(self, obj_pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both global and local features (for future cross-attention).
        
        Returns:
            global_feat: (B, 1024)
            local_tokens: (B, 64, 384)
        """
        if obj_pc.shape[1] == 3 and obj_pc.shape[2] != 3:
            obj_pc = obj_pc.transpose(1, 2)
        
        B, N, _ = obj_pc.shape
        if N > 2048:
            idx = fps_pytorch(obj_pc, 2048)
            obj_pc = index_points(obj_pc, idx)
        
        global_feat, local_tokens, patch_centers = self.backbone.forward_local(obj_pc)
        global_feat = self.adapter(global_feat)
        return global_feat, local_tokens, patch_centers


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PointM2AEObjectEncoder(output_dim=1024).to(device)
    print(f"Total params: {count_parameters(encoder)/1e6:.2f}M")
    print(f"  Backbone: {count_parameters(encoder.backbone)/1e6:.2f}M")
    print(f"  Adapter: {count_parameters(encoder.adapter)/1e6:.2f}M")
    
    # Test forward
    pts = torch.randn(2, 3000, 3).to(device)
    feat = encoder(pts)
    print(f"Input: {pts.shape} → Output: {feat.shape}")
    
    # Test local
    global_feat, local_tokens = encoder.forward_local(pts)
    print(f"Global: {global_feat.shape}, Local: {local_tokens.shape}")
