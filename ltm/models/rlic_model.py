"""Full RLIC model (M5) and ablation variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..data.encoding import CellBatch
from .heads import PolicyHead, RetrievalHead, ValueHead
from .rlic_layer import (
    CellEncoder,
    RLICLayer,
    class_canonical_pool,
    mean_pool,
)


@dataclass
class RLICConfig:
    n_labels: int = 512
    n_types: int = 32
    n_grades: int = 16
    n_role_ids: int = 64 * 64  # supports composite role ids for side direction
    n_families: int = 8
    n_premise_buckets: int = 4096

    d: int = 256
    L: int = 4
    n_heads: int = 4
    use_side: bool = True
    use_role_labels: bool = True   # ablation A1 toggle
    freeze_transport: bool = False # ablation A2 toggle
    drop_spectral_pe: bool = True  # ablation A3 marker (PE not yet used)

    embed_dim: int = 128  # retrieval embed dim


class RLICModel(nn.Module):
    def __init__(self, cfg: RLICConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CellEncoder(cfg.n_labels, cfg.n_types, cfg.n_grades, cfg.d)
        self.layers = nn.ModuleList(
            [
                RLICLayer(
                    d=cfg.d,
                    n_heads=cfg.n_heads,
                    n_role_ids=cfg.n_role_ids,
                    use_side=cfg.use_side,
                )
                for _ in range(cfg.L)
            ]
        )
        # Class-canonical pool size: d * 32 buckets
        self.n_label_buckets = 32
        pooled_dim = cfg.d * self.n_label_buckets
        self.policy = PolicyHead(pooled_dim, cfg.n_families)
        self.value = ValueHead(pooled_dim)
        self.retrieval = RetrievalHead(pooled_dim, cfg.embed_dim)

        if cfg.freeze_transport:
            self._freeze_transport()

    def _freeze_transport(self) -> None:
        """Ablation A2: freeze ρ^↑, ρ^↓ to identity (zero out role conditioning)."""
        for layer in self.layers:
            for attn in [layer.attn_down, layer.attn_up,
                         getattr(layer, "attn_side", None)]:
                if attn is None:
                    continue
                with torch.no_grad():
                    # set transport to (m, role) → m (passing input through)
                    attn.role_emb.weight.zero_()
                    for p in attn.transport.parameters():
                        p.zero_()
                for p in attn.transport.parameters():
                    p.requires_grad = False
                attn.role_emb.weight.requires_grad = False

    def encode(self, batch: CellBatch) -> Tensor:
        # Ablation A1: optionally collapse all roles to a single bucket
        down_role = batch.down_role
        up_role = batch.up_role
        side_self = batch.side_role_self
        side_other = batch.side_role_other
        if not self.cfg.use_role_labels:
            down_role = torch.zeros_like(down_role)
            up_role = torch.zeros_like(up_role)
            side_self = torch.zeros_like(side_self)
            side_other = torch.zeros_like(side_other)

        h = self.encoder(batch.label, batch.type_tag, batch.grade)
        for layer in self.layers:
            h = layer(
                h,
                down_src=batch.down_src, down_dst=batch.down_dst, down_role=down_role,
                up_src=batch.up_src, up_dst=batch.up_dst, up_role=up_role,
                side_src=batch.side_src, side_dst=batch.side_dst,
                side_role_self=side_self, side_role_other=side_other,
            )
        return h

    def forward(self, batch: CellBatch) -> dict[str, Tensor]:
        h = self.encode(batch)
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
