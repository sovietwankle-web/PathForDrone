"""
Cross-Attention Network for Waypoint Allocation.

Architecture (SADCHER-inspired):
  Robot features    → RobotEncoder  → (n_r, d_model)
  Waypoint features → WaypointEncoder → (n_w, d_model)
  Pairwise features → PairwiseEncoder → attention bias (n_r, n_w)

  CrossAttentionBlock × N:
    robot attends to waypoints (with pairwise bias)
    waypoint attends to robots (with transposed pairwise bias)

  ScoreHead: bilinear(robot_i, waypoint_j) → S[i,j]
  ValueHead: pool → MLP → V(s)  (for PPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .config import NeuralAllocatorConfig


class RobotEncoder(nn.Module):
    """MLP encoder for robot features."""

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaypointEncoder(nn.Module):
    """MLP encoder for waypoint features."""

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseEncoder(nn.Module):
    """Encodes pairwise features into attention bias scalars per head."""

    def __init__(self, in_dim: int, n_heads: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_r, n_w, pairwise_feat_dim)
        Returns:
            (batch, n_heads, n_r, n_w) attention bias
        """
        bias = self.net(x)  # (batch, n_r, n_w, n_heads)
        return bias.permute(0, 3, 1, 2)  # (batch, n_heads, n_r, n_w)


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention: robots attend to waypoints, then
    waypoints attend to robots. Each with pairwise attention bias.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Robot → Waypoint attention
        self.q_r = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.out_r = nn.Linear(d_model, d_model)
        self.norm_r1 = nn.LayerNorm(d_model)
        self.ff_r = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_r2 = nn.LayerNorm(d_model)

        # Waypoint → Robot attention
        self.q_w = nn.Linear(d_model, d_model)
        self.k_r = nn.Linear(d_model, d_model)
        self.v_r = nn.Linear(d_model, d_model)
        self.out_w = nn.Linear(d_model, d_model)
        self.norm_w1 = nn.LayerNorm(d_model)
        self.ff_w = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_w2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _multi_head_attn(self, Q: torch.Tensor, K: torch.Tensor,
                          V: torch.Tensor, bias: Optional[torch.Tensor],
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            Q: (batch, n_q, d_model)
            K: (batch, n_k, d_model)
            V: (batch, n_k, d_model)
            bias: (batch, n_heads, n_q, n_k) additive bias
            mask: (batch, 1, 1, n_k) bool mask, True = valid
        Returns:
            (batch, n_q, d_model)
        """
        B, n_q, _ = Q.shape
        n_k = K.shape[1]

        Q = Q.view(B, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, n_k, self.n_heads, self.head_dim).transpose(1, 2)
        # All now (B, n_heads, n_*, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if bias is not None:
            scores = scores + bias

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, n_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, n_q, self.d_model)
        return out

    def forward(self, robot_emb: torch.Tensor, waypoint_emb: torch.Tensor,
                pairwise_bias: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                waypoint_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            robot_emb: (B, n_r, d_model)
            waypoint_emb: (B, n_w, d_model)
            pairwise_bias: (B, n_heads, n_r, n_w)
            robot_mask: (B, n_r) bool
            waypoint_mask: (B, n_w) bool
        Returns:
            updated (robot_emb, waypoint_emb)
        """
        # Prepare masks for attention: (B, 1, 1, n_k)
        w_mask = waypoint_mask.unsqueeze(1).unsqueeze(2) if waypoint_mask is not None else None
        r_mask = robot_mask.unsqueeze(1).unsqueeze(2) if robot_mask is not None else None

        # Robot attends to waypoints
        q = self.q_r(robot_emb)
        k = self.k_w(waypoint_emb)
        v = self.v_w(waypoint_emb)
        attn_out = self.out_r(self._multi_head_attn(q, k, v, pairwise_bias, w_mask))
        robot_emb = self.norm_r1(robot_emb + attn_out)
        robot_emb = self.norm_r2(robot_emb + self.ff_r(robot_emb))

        # Waypoint attends to robots (transpose bias)
        q = self.q_w(waypoint_emb)
        k = self.k_r(robot_emb)
        v = self.v_r(robot_emb)
        bias_t = pairwise_bias.transpose(-2, -1) if pairwise_bias is not None else None
        attn_out = self.out_w(self._multi_head_attn(q, k, v, bias_t, r_mask))
        waypoint_emb = self.norm_w1(waypoint_emb + attn_out)
        waypoint_emb = self.norm_w2(waypoint_emb + self.ff_w(waypoint_emb))

        return robot_emb, waypoint_emb


class ScoreHead(nn.Module):
    """Computes assignment score matrix via bilinear interaction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.bias_proj = nn.Linear(d_model, 1)

    def forward(self, robot_emb: torch.Tensor, waypoint_emb: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            robot_emb: (B, n_r, d_model)
            waypoint_emb: (B, n_w, d_model)
        Returns:
            score_matrix: (B, n_r, n_w)
        """
        # Bilinear: robot_i^T W waypoint_j
        transformed = self.W(robot_emb)  # (B, n_r, d_model)
        scores = torch.bmm(transformed, waypoint_emb.transpose(1, 2))  # (B, n_r, n_w)
        return scores


class ValueHead(nn.Module):
    """State-value estimator for PPO. Pools both embeddings → scalar."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, robot_emb: torch.Tensor, waypoint_emb: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                waypoint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, 1) value estimate."""
        # Masked mean pooling
        if robot_mask is not None:
            r_mask = robot_mask.unsqueeze(-1).float()
            r_pool = (robot_emb * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1)
        else:
            r_pool = robot_emb.mean(dim=1)

        if waypoint_mask is not None:
            w_mask = waypoint_mask.unsqueeze(-1).float()
            w_pool = (waypoint_emb * w_mask).sum(dim=1) / w_mask.sum(dim=1).clamp(min=1)
        else:
            w_pool = waypoint_emb.mean(dim=1)

        combined = torch.cat([r_pool, w_pool], dim=-1)
        return self.net(combined)


class WaypointAllocationNet(nn.Module):
    """
    Full network: encode features → cross-attention → score matrix + value.

    Usage:
        model = WaypointAllocationNet(config)
        scores = model(robot_feats, waypoint_feats, pairwise_feats)
        # scores: (B, n_r, n_w) — decode with Hungarian or greedy

        # For RL:
        logits, value = model.forward_rl(robot_feats, waypoint_feats,
                                          pairwise_feats, assigned_mask)
    """

    def __init__(self, config: Optional[NeuralAllocatorConfig] = None):
        super().__init__()
        cfg = config or NeuralAllocatorConfig()
        self.config = cfg

        # Encoders
        self.robot_encoder = RobotEncoder(cfg.robot_feat_dim, cfg.d_model)
        self.waypoint_encoder = WaypointEncoder(cfg.waypoint_feat_dim, cfg.d_model)
        self.pairwise_encoder = PairwiseEncoder(cfg.pairwise_feat_dim, cfg.n_heads)

        # Cross-attention stack
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout)
            for _ in range(cfg.n_cross_attn_layers)
        ])

        # Heads
        self.score_head = ScoreHead(cfg.d_model)
        self.value_head = ValueHead(cfg.d_model)

    def _encode(self, robot_feats: torch.Tensor, waypoint_feats: torch.Tensor,
                pairwise_feats: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                waypoint_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and run cross-attention. Returns (robot_emb, waypoint_emb)."""
        robot_emb = self.robot_encoder(robot_feats)
        waypoint_emb = self.waypoint_encoder(waypoint_feats)
        pairwise_bias = self.pairwise_encoder(pairwise_feats)

        for layer in self.cross_attn_layers:
            robot_emb, waypoint_emb = layer(
                robot_emb, waypoint_emb, pairwise_bias,
                robot_mask, waypoint_mask
            )

        return robot_emb, waypoint_emb

    def forward(self, robot_feats: torch.Tensor, waypoint_feats: torch.Tensor,
                pairwise_feats: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                waypoint_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass for imitation learning.

        Args:
            robot_feats: (B, n_r, 7)
            waypoint_feats: (B, n_w, 5)
            pairwise_feats: (B, n_r, n_w, 2)
            robot_mask: (B, n_r) bool — True for valid robots
            waypoint_mask: (B, n_w) bool — True for valid waypoints
        Returns:
            score_matrix: (B, n_r, n_w) — assignment logits
        """
        robot_emb, waypoint_emb = self._encode(
            robot_feats, waypoint_feats, pairwise_feats,
            robot_mask, waypoint_mask
        )
        scores = self.score_head(robot_emb, waypoint_emb)

        # Mask invalid positions
        if waypoint_mask is not None:
            scores = scores.masked_fill(~waypoint_mask.unsqueeze(1), float('-inf'))
        if robot_mask is not None:
            scores = scores.masked_fill(~robot_mask.unsqueeze(2), float('-inf'))

        return scores

    def forward_rl(self, robot_feats: torch.Tensor, waypoint_feats: torch.Tensor,
                   pairwise_feats: torch.Tensor,
                   assigned_mask: Optional[torch.Tensor] = None,
                   robot_mask: Optional[torch.Tensor] = None,
                   waypoint_mask: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for RL (autoregressive decoding).

        Args:
            assigned_mask: (B, n_w) bool — True for already-assigned waypoints
        Returns:
            scores: (B, n_r, n_w) with assigned waypoints masked out
            value: (B, 1) state value estimate
        """
        robot_emb, waypoint_emb = self._encode(
            robot_feats, waypoint_feats, pairwise_feats,
            robot_mask, waypoint_mask
        )
        scores = self.score_head(robot_emb, waypoint_emb)

        # Mask assigned waypoints
        if assigned_mask is not None:
            scores = scores.masked_fill(assigned_mask.unsqueeze(1), float('-inf'))
        if waypoint_mask is not None:
            scores = scores.masked_fill(~waypoint_mask.unsqueeze(1), float('-inf'))
        if robot_mask is not None:
            scores = scores.masked_fill(~robot_mask.unsqueeze(2), float('-inf'))

        value = self.value_head(robot_emb, waypoint_emb, robot_mask, waypoint_mask)

        return scores, value
