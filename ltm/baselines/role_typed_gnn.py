"""M4: role-typed GNN over the grade-0/grade-1 incidence graph only.

No higher-grade cells: hyperedges and APP_N / EQ_LIT / affordance cells are
dropped. The model still sees role labels on the grade-1 incidences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..data.encoding import CellBatch
from ..models.heads import PolicyHead, RetrievalHead, ValueHead
from ..models.rlic_layer import CellEncoder, class_canonical_pool
from .gnn_layer import _segment_softmax, _segment_sum


@dataclass
class GNNConfig:
    n_labels: int = 512
    n_types: int = 32
    n_grades: int = 16
    n_role_ids: int = 1024
    n_families: int = 8
    d: int = 256
    L: int = 4
    n_heads: int = 4


class RoleTypedGNN(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CellEncoder(cfg.n_labels, cfg.n_types, cfg.n_grades, cfg.d)
        self.role_emb = nn.Embedding(cfg.n_role_ids, cfg.d)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
             for _ in range(cfg.L)]
        )
        self.W_Q = nn.ModuleList([nn.Linear(cfg.d, cfg.d) for _ in range(cfg.L)])
        self.W_K = nn.ModuleList([nn.Linear(cfg.d, cfg.d) for _ in range(cfg.L)])
        self.W_V = nn.ModuleList([nn.Linear(cfg.d, cfg.d) for _ in range(cfg.L)])
        self.upd = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(2 * cfg.d), nn.Linear(2 * cfg.d, cfg.d),
                           nn.GELU(), nn.Linear(cfg.d, cfg.d))
             for _ in range(cfg.L)]
        )

        self.n_label_buckets = 32
        pooled_dim = cfg.d * self.n_label_buckets
        self.policy = PolicyHead(pooled_dim, cfg.n_families)
        self.value = ValueHead(pooled_dim)
        self.retrieval = RetrievalHead(pooled_dim, 128)

    def forward(self, batch: CellBatch) -> dict[str, Tensor]:
        import math
        keep = batch.grade <= 1
        # Filter edges to only those between grade-0/grade-1 cells
        mask = keep[batch.down_src] & keep[batch.down_dst]
        src = batch.down_src[mask]
        dst = batch.down_dst[mask]
        role = batch.down_role[mask]
        # bidirectional
        src_b = torch.cat([src, dst])
        dst_b = torch.cat([dst, src])
        role_b = torch.cat([role, role])

        h = self.encoder(batch.label, batch.type_tag, batch.grade)
        # zero out higher-grade cells' contributions
        h = h * keep.float().unsqueeze(-1)

        N = h.shape[0]
        d = self.cfg.d
        H = self.cfg.n_heads
        dh = d // H

        for li in range(self.cfg.L):
            rho = self.role_emb(role_b)
            m = self.msg[li](torch.cat([h[src_b], rho], dim=-1))
            q = self.W_Q[li](h).view(N, H, dh)
            k = self.W_K[li](m).view(-1, H, dh)
            v = self.W_V[li](m).view(-1, H, dh)
            scores = (q[dst_b] * k).sum(-1) / math.sqrt(dh)
            alpha = _segment_softmax(scores, dst_b, N)
            weighted = v * alpha.unsqueeze(-1)
            a = _segment_sum(weighted.reshape(-1, d), dst_b, N)
            h = h + self.upd[li](torch.cat([h, a], dim=-1))

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
