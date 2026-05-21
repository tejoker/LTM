"""Mode 1 task heads (§5.6)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PolicyHead(nn.Module):
    """Mode 1: tactic-family classification on K_afford.

    Bottlenecks through `hidden` (defaults to in_dim // 32 to roughly recover
    the unpooled dim) so the head doesn't dominate param count.
    """

    def __init__(self, in_dim: int, n_families: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = max(64, in_dim // 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_families),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.net(pooled)


class ValueHead(nn.Module):
    """Pool K_afford to a scalar logit (BCEWithLogits-safe under autocast)."""

    def __init__(self, in_dim: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = max(64, in_dim // 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.net(pooled).squeeze(-1)


class RetrievalHead(nn.Module):
    """score(K, t_i) = φ(K)^⊤ ψ(t_i) with ψ computed independently of K
    (per the retrieval caveat in Theorem 6.3)."""

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        z = self.phi(pooled)
        return z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-6))


class PremiseEncoder(nn.Module):
    """ψ(t_i): independent encoder for premise t_i, never reading K.

    Inputs are lemma-name token ids (or hashed buckets if vocab is huge).
    """

    def __init__(self, n_premise_buckets: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_premise_buckets, embed_dim)

    def forward(self, premise_ids: Tensor) -> Tensor:
        z = self.emb(premise_ids)
        return z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-6))
