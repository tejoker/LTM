"""M2 weak hypergraph and M3 strong role-aware hypergraph baselines (§7.3).

Both treat higher-grade cells as typed hyperedges. M2 has no role conditioning
and no hyperedge hidden states; M3 has both, and uses a bipartite node–hyperedge
message-passing scheme with one MLP per role-and-class signature.

These baselines must be **parameter-count matched** to M5 in real runs (paper
§7.3). The Tiny configs here are close to matched at d=256, L=4, H=4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..data.encoding import CellBatch
from .gnn_layer import _segment_softmax, _segment_sum
from ..models.heads import PolicyHead, RetrievalHead, ValueHead
from ..models.rlic_layer import CellEncoder, class_canonical_pool


@dataclass
class HyperConfig:
    n_labels: int = 512
    n_types: int = 32
    n_grades: int = 16
    n_role_ids: int = 1024
    n_families: int = 8

    d: int = 256
    L: int = 4
    n_heads: int = 4
    use_roles: bool = True       # M3: True; M2: False
    use_hyperedge_state: bool = True  # M3: True; M2: False
    # Collapsed-grade-M5 / M3+ ablation: add node↔node attention via shared
    # hyperedges, indexed by the role pair on that hyperedge. Off by default
    # so legacy M2/M3 configs are unchanged.
    use_side: bool = False


class _NodeHyperedgeAttn(nn.Module):
    """Bipartite message passing: nodes attend over incident hyperedges and
    vice versa. Optionally role-conditioned."""

    def __init__(self, d: int, n_heads: int, use_roles: bool, n_role_ids: int):
        super().__init__()
        self.h = n_heads
        self.d = d
        self.use_roles = use_roles
        if use_roles:
            self.role_emb = nn.Embedding(n_role_ids, d)
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.msg = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(
        self,
        h_dst: Tensor,        # [N_dst, d]
        h_src: Tensor,        # [N_src, d]
        src: Tensor,
        dst: Tensor,
        role: Tensor,
    ) -> Tensor:
        N_dst, d = h_dst.shape
        E = src.shape[0]
        if E == 0:
            return torch.zeros_like(h_dst)
        if self.use_roles:
            rho = self.role_emb(role)
        else:
            rho = torch.zeros((E, d), device=h_src.device, dtype=h_src.dtype)
        m = self.msg(torch.cat([h_src[src], rho], dim=-1))

        import math
        H = self.h
        dh = d // H
        q = self.W_Q(h_dst).view(N_dst, H, dh)
        k = self.W_K(m).view(E, H, dh)
        v = self.W_V(m).view(E, H, dh)
        scores = (q[dst] * k).sum(-1) / math.sqrt(dh)
        alpha = _segment_softmax(scores, dst, N_dst)
        weighted = v * alpha.unsqueeze(-1)
        return _segment_sum(weighted.reshape(E, d), dst, N_dst)


class HypergraphModel(nn.Module):
    """Weak (M2), strong role-aware (M3), or collapsed-grade-rich-transport
    (M3+, the ablation isolating grading) hypergraph encoder.

    Cells with grade ≥ 2 are treated as *hyperedges*; cells with grade ≤ 1 are
    *nodes*. The CellBatch boundary edges already give us node↔hyperedge
    incidence (boundary edges from a node face to its higher-grade coface).

    The optional ``cfg.use_side`` flag adds a third attention direction in
    which each node attends to other nodes sharing one of its hyperedges, with
    the message conditioned on the role pair on that shared hyperedge. This
    isolates the cellular *grading* itself (versus role-conditioned transport
    over typed hyperedges) — see §7.8 of the paper.
    """

    def __init__(self, cfg: HyperConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CellEncoder(cfg.n_labels, cfg.n_types, cfg.n_grades, cfg.d)
        self.node_to_edge = nn.ModuleList(
            [_NodeHyperedgeAttn(cfg.d, cfg.n_heads, cfg.use_roles, cfg.n_role_ids)
             for _ in range(cfg.L)]
        )
        self.edge_to_node = nn.ModuleList(
            [_NodeHyperedgeAttn(cfg.d, cfg.n_heads, cfg.use_roles, cfg.n_role_ids)
             for _ in range(cfg.L)]
        )
        upd_in = (3 * cfg.d) if cfg.use_side else (2 * cfg.d)
        self.upd_node = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(upd_in), nn.Linear(upd_in, cfg.d),
                           nn.GELU(), nn.Linear(cfg.d, cfg.d))
             for _ in range(cfg.L)]
        )
        if cfg.use_hyperedge_state:
            self.upd_edge = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(2 * cfg.d), nn.Linear(2 * cfg.d, cfg.d),
                               nn.GELU(), nn.Linear(cfg.d, cfg.d))
                 for _ in range(cfg.L)]
            )
        if cfg.use_side:
            # Side direction: node ↔ node via shared hyperedge. Composite role
            # is (role_u, role_v) -> u*n_role + v, clamped into the embedding.
            self.node_to_node = nn.ModuleList(
                [_NodeHyperedgeAttn(cfg.d, cfg.n_heads, True,
                                    min(cfg.n_role_ids * cfg.n_role_ids, 64 * 64))
                 for _ in range(cfg.L)]
            )

        self.n_label_buckets = 32
        pooled_dim = cfg.d * self.n_label_buckets
        self.policy = PolicyHead(pooled_dim, cfg.n_families)
        self.value = ValueHead(pooled_dim)
        self.retrieval = RetrievalHead(pooled_dim, 128)

    def forward(self, batch: CellBatch) -> dict[str, Tensor]:
        is_edge = (batch.grade >= 2)
        is_node = ~is_edge

        h = self.encoder(batch.label, batch.type_tag, batch.grade)

        # Boundary edges in CellBatch go face → coface (node → hyperedge).
        # Filter to only those where src is a node and dst is a hyperedge.
        ndown_mask = is_node[batch.down_src] & is_edge[batch.down_dst]
        n2e_src = batch.down_src[ndown_mask]
        n2e_dst = batch.down_dst[ndown_mask]
        n2e_role = batch.down_role[ndown_mask] if self.cfg.use_roles else torch.zeros_like(n2e_src)

        if self.cfg.use_side:
            # Compose role pair (u, v) -> u*64 + v ∈ [0, 64*64).
            r_u = batch.he_share_role_u.clamp(max=63)
            r_v = batch.he_share_role_v.clamp(max=63)
            n2n_role = r_u * 64 + r_v
            n2n_src = batch.he_share_src
            n2n_dst = batch.he_share_dst

        for li in range(self.cfg.L):
            # nodes attend over... themselves? in the weak case, only over
            # hyperedges. We do edge→node and node→edge each layer.
            if self.cfg.use_hyperedge_state:
                a_e = self.node_to_edge[li](h, h, n2e_src, n2e_dst, n2e_role)
                h_e_new = self.upd_edge[li](torch.cat([h, a_e], dim=-1))
                # only update edges
                h = torch.where(is_edge.unsqueeze(-1), h + h_e_new, h)
            a_n = self.edge_to_node[li](h, h, n2e_dst, n2e_src, n2e_role)
            if self.cfg.use_side:
                a_s = self.node_to_node[li](h, h, n2n_src, n2n_dst, n2n_role)
                cat_n = torch.cat([h, a_n, a_s], dim=-1)
            else:
                cat_n = torch.cat([h, a_n], dim=-1)
            h_n_new = self.upd_node[li](cat_n)
            h = torch.where(is_node.unsqueeze(-1), h + h_n_new, h)

        pooled = class_canonical_pool(
            h, batch.batch_idx, batch.n_graphs,
            batch.label, n_label_buckets=self.n_label_buckets,
        )
        return {
            "pooled": pooled,
            "policy_logits": self.policy(pooled),
            "value": self.value(pooled),
            "retrieval_query": self.retrieval(pooled),
        }


def make_weak(cfg: HyperConfig | None = None) -> HypergraphModel:
    """M2 weak hypergraph: no roles, no hyperedge state."""
    cfg = cfg or HyperConfig()
    cfg.use_roles = False
    cfg.use_hyperedge_state = False
    return HypergraphModel(cfg)


def make_strong_role_aware(cfg: HyperConfig | None = None) -> HypergraphModel:
    """M3 strong role-aware hypergraph: role-labelled, hyperedge hidden states."""
    cfg = cfg or HyperConfig()
    cfg.use_roles = True
    cfg.use_hyperedge_state = True
    cfg.use_side = False
    return HypergraphModel(cfg)


def make_collapsed_grade_rich(cfg: HyperConfig | None = None) -> HypergraphModel:
    """M3+ (collapsed-grade M5 with rich transport): all higher cells are
    typed hyperedges (no graded levels) but three role-conditioned attention
    directions are preserved — node→hyperedge, hyperedge→node, and node↔node
    via shared hyperedge. Used to isolate the cellular grading itself."""
    cfg = cfg or HyperConfig()
    cfg.use_roles = True
    cfg.use_hyperedge_state = True
    cfg.use_side = True
    return HypergraphModel(cfg)
