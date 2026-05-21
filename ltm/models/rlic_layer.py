"""§5 architecture: Role-conditioned incidence-transport layer (the §4.4 relaxed model).

Theorem 6.3 holds when:
(i)  embeddings depend only on (ℓ, τ, g);                       — CellEncoder
(ii) labels follow the §5.5 preservation policy;                — handled in parser
(iii) all parameter weights are indexed by class-and-role signatures, never cell identity;
(iv) truncation is canonical;                                   — handled in parser
(v)  positional features, if used, are constructed equivariantly;
(vi) pooling is class-canonical.                                — class-canonical pool here

The relaxed transport (§4.4) provides two independently parameterised maps per
role+class-pair signature:
    ρ^{r,↑}_{τ→σ} : R^d_τ → R^d_σ   (face → coface, "upward")
    ρ^{r,↓}_{σ→τ} : R^d_σ → R^d_τ   (coface → face, "downward")
plus a same-grade map indexed by the role pair (r, r') and shared face class.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Cell encoder — class-shared embeddings (Theorem 6.3(i))
# ---------------------------------------------------------------------------


class CellEncoder(nn.Module):
    def __init__(
        self,
        n_labels: int,
        n_types: int,
        n_grades: int,
        d: int,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(n_labels, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.grade_emb = nn.Embedding(n_grades, d)
        self.proj = nn.Linear(3 * d, d)

    def forward(self, label: Tensor, type_tag: Tensor, grade: Tensor) -> Tensor:
        e = torch.cat(
            [self.label_emb(label), self.type_emb(type_tag), self.grade_emb(grade)],
            dim=-1,
        )
        return self.proj(e)


# ---------------------------------------------------------------------------
# Role-conditioned attention per neighbourhood direction (↓, ↑, ↔)
# ---------------------------------------------------------------------------


def _segment_softmax(scores: Tensor, dst: Tensor, n_nodes: int) -> Tensor:
    """Vectorised, numerically-stable softmax over edges grouped by destination.

    Supports scores of shape ``[E]`` or ``[E, H]``. Operates over all heads in
    one pass — avoiding the Python-loop-over-heads that serialised kernels in
    the previous implementation.
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    E, H = scores.shape
    dst_e = dst.unsqueeze(-1).expand(E, H)
    # per-(dst, head) max for numerical stability
    max_per = torch.full((n_nodes, H), -float("inf"),
                          device=scores.device, dtype=scores.dtype)
    max_per = max_per.scatter_reduce(0, dst_e, scores, reduce="amax", include_self=True)
    safe = torch.isfinite(max_per)
    max_per = torch.where(safe, max_per, torch.zeros_like(max_per))
    centred = scores - max_per[dst]
    exp = centred.exp()
    z = torch.zeros((n_nodes, H), device=scores.device, dtype=scores.dtype)
    z.scatter_add_(0, dst_e, exp)
    z = torch.where(z > 0, z, torch.ones_like(z))
    out = exp / z[dst]
    return out.squeeze(-1) if squeeze else out


def _segment_sum(values: Tensor, dst: Tensor, n_nodes: int) -> Tensor:
    """Sum values rows indexed by dst into [n_nodes, dim]. Vectorised."""
    if values.dim() == 1:
        out = torch.zeros((n_nodes,), device=values.device, dtype=values.dtype)
        out.scatter_add_(0, dst, values)
        return out
    out = torch.zeros((n_nodes,) + values.shape[1:],
                       device=values.device, dtype=values.dtype)
    out.scatter_add_(0, dst.unsqueeze(-1).expand_as(values), values)
    return out


class RoleConditionedAttention(nn.Module):
    """Multi-head attention along a single neighbourhood direction.

    Messages are role-conditioned via a role embedding fed into a shared
    transport MLP. This is the relaxed variant of §4.4: an MLP shared across
    cells in the same role-and-class signature, not a sheaf functor.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_role_ids: int = 64,
        role_indexed_kv: bool = False,
    ):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.role_emb = nn.Embedding(n_role_ids, d)
        # Transport MLP φ : (m, role) → m'; shared across signatures (relaxed)
        self.transport = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.W_Q = nn.Linear(d, d)
        if role_indexed_kv:
            # an additional per-role bias on K, V
            self.W_K = nn.Linear(d, d)
            self.W_V = nn.Linear(d, d)
            self.role_bias_K = nn.Embedding(n_role_ids, d)
            self.role_bias_V = nn.Embedding(n_role_ids, d)
        else:
            self.W_K = nn.Linear(d, d)
            self.W_V = nn.Linear(d, d)
            self.role_bias_K = None
            self.role_bias_V = None

    def forward(
        self,
        h: Tensor,            # [N, d] node states
        src: Tensor,          # [E] source cell idx (neighbour)
        dst: Tensor,          # [E] destination cell idx (self)
        role: Tensor,         # [E] role id
    ) -> Tensor:
        N, d = h.shape
        E = src.shape[0]
        if E == 0:
            return torch.zeros_like(h)

        # transported messages: m_ν→σ
        rho = self.role_emb(role)                   # [E, d]
        m = self.transport(torch.cat([h[src], rho], dim=-1))  # [E, d]

        # multi-head Q/K/V
        H = self.h
        dh = d // H

        q = self.W_Q(h).view(N, H, dh)               # [N, H, dh]
        k = self.W_K(m)                              # [E, d]
        v = self.W_V(m)                              # [E, d]
        if self.role_bias_K is not None:
            k = k + self.role_bias_K(role)
            v = v + self.role_bias_V(role)
        k = k.view(E, H, dh)
        v = v.view(E, H, dh)

        # per-head attention scores per edge, then vectorised segment-softmax
        scores = (q[dst] * k).sum(-1) / math.sqrt(dh)   # [E, H]
        alpha = _segment_softmax(scores, dst, N)         # [E, H]
        weighted = v * alpha.unsqueeze(-1)               # [E, H, dh]
        out = _segment_sum(weighted.reshape(E, d), dst, N)  # [N, d]
        return out


# ---------------------------------------------------------------------------
# Layer update (§5.4): h^{l+1} = MLP(h^l, Attn↓, Attn↑, Attn↔)
# ---------------------------------------------------------------------------


class RLICLayer(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_role_ids: int = 64,
        use_side: bool = True,
    ):
        super().__init__()
        self.use_side = use_side
        self.ln_in = nn.LayerNorm(d)
        self.attn_down = RoleConditionedAttention(d, n_heads, n_role_ids)
        self.attn_up = RoleConditionedAttention(d, n_heads, n_role_ids)
        if use_side:
            self.attn_side = RoleConditionedAttention(
                d, n_heads, n_role_ids, role_indexed_kv=True
            )
            in_dim = 4 * d
        else:
            self.attn_side = None
            in_dim = 3 * d
        self.ln_mid = nn.LayerNorm(in_dim)
        self.update = nn.Sequential(
            nn.Linear(in_dim, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def forward(
        self,
        h: Tensor,
        down_src: Tensor, down_dst: Tensor, down_role: Tensor,
        up_src: Tensor, up_dst: Tensor, up_role: Tensor,
        side_src: Tensor = None, side_dst: Tensor = None,
        side_role_self: Tensor = None, side_role_other: Tensor = None,
    ) -> Tensor:
        h0 = h
        h = self.ln_in(h)
        a_down = self.attn_down(h, down_src, down_dst, down_role)
        a_up = self.attn_up(h, up_src, up_dst, up_role)
        if self.use_side and side_src is not None:
            # combine the two roles into a composite role id for the side direction
            combined_role = side_role_self * 64 + side_role_other  # small composite
            a_side = self.attn_side(h, side_src, side_dst, combined_role.clamp(max=64*64-1))
            cat = torch.cat([h, a_down, a_up, a_side], dim=-1)
        else:
            cat = torch.cat([h, a_down, a_up], dim=-1)
        cat = self.ln_mid(cat)
        upd = self.update(cat)
        return h0 + upd


# ---------------------------------------------------------------------------
# Class-canonical pooling (Theorem 6.3(vi))
# ---------------------------------------------------------------------------


def class_canonical_pool(
    h: Tensor,          # [N, d]
    batch_idx: Tensor,  # [N]
    n_graphs: int,
    label: Tensor,
    n_label_buckets: int = 32,
) -> Tensor:
    """Pool per (graph, label-bucket) → mean, then concat across buckets.

    Class-canonical pooling: invariant under cell renumbering, distinguishes by
    cell class so the policy head can read class-typed evidence.
    """
    bucket = label.clamp(max=n_label_buckets - 1)
    key = batch_idx * n_label_buckets + bucket
    n_keys = n_graphs * n_label_buckets
    sums = torch.zeros((n_keys,) + h.shape[1:], device=h.device, dtype=h.dtype)
    sums.scatter_add_(0, key.unsqueeze(-1).expand_as(h), h)
    counts = torch.zeros((n_keys,), device=h.device, dtype=h.dtype)
    counts.scatter_add_(0, key, torch.ones_like(key, dtype=h.dtype))
    counts = counts.clamp_min(1.0)
    means = sums / counts.unsqueeze(-1)
    return means.view(n_graphs, n_label_buckets * h.shape[-1])


def mean_pool(h: Tensor, batch_idx: Tensor, n_graphs: int) -> Tensor:
    d = h.shape[-1]
    sums = torch.zeros((n_graphs, d), device=h.device, dtype=h.dtype)
    sums.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
    counts = torch.zeros((n_graphs,), device=h.device, dtype=h.dtype)
    counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=h.dtype))
    return sums / counts.clamp_min(1.0).unsqueeze(-1)
