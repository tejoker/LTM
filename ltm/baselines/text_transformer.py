"""M1: small text Transformer over the pretty-printed proof state.

Trained from scratch on a character / byte-level vocabulary so the comparison
to RLIC and hypergraph baselines is fair (no external pretraining).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..proof_state import Expr, ProofState


def pretty_print_expr(e: Expr) -> str:
    if e.op == "var":
        return e.name
    if e.op == "const":
        return e.name
    if e.op == "type":
        return e.name
    if e.op == "eq" and len(e.children) == 2:
        return f"({pretty_print_expr(e.children[0])} = {pretty_print_expr(e.children[1])})"
    if e.op == "app":
        return "(" + " ".join(pretty_print_expr(c) for c in e.children) + ")"
    if e.op == "binder":
        kw = "∀" if e.name == "forall" else "λ"
        if len(e.children) == 2:
            return f"{kw}({pretty_print_expr(e.children[0])}). {pretty_print_expr(e.children[1])}"
    return f"<{e.op}>"


def pretty_print_state(P: ProofState) -> str:
    parts = []
    for h in P.gamma:
        parts.append(f"{h.name} : {pretty_print_expr(h.typ)}")
    if P.goal is not None:
        parts.append("⊢ " + pretty_print_expr(P.goal))
    return " ; ".join(parts)


def tokenize_bytes(s: str, max_len: int = 512) -> list[int]:
    """Byte-level tokens, +3 reserved for [PAD]=0, [CLS]=1, [SEP]=2."""
    ids = [1] + [b + 3 for b in s.encode("utf-8")[: max_len - 2]] + [2]
    return ids


@dataclass
class TextConfig:
    vocab_size: int = 259
    d: int = 256
    L: int = 4
    n_heads: int = 4
    max_len: int = 512
    n_families: int = 8


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d,
            nhead=cfg.n_heads,
            dim_feedforward=2 * cfg.d,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.L)
        self.policy = nn.Linear(cfg.d, cfg.n_families)
        self.value = nn.Linear(cfg.d, 1)
        self.retrieval = nn.Linear(cfg.d, 128)

    def forward(self, ids: Tensor, mask: Tensor) -> dict[str, Tensor]:
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        # key_padding_mask: True positions are masked (padding)
        kpm = ~mask  # mask True = valid, kpm True = pad
        x = self.enc(x, src_key_padding_mask=kpm)
        # CLS pooling
        pooled = x[:, 0]
        z = self.retrieval(pooled)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "pooled": pooled,
            "policy_logits": self.policy(pooled),
            "value": self.value(pooled).squeeze(-1),  # logit
            "retrieval_query": z,
        }


def collate_text(
    states: list[ProofState], cfg: TextConfig
) -> tuple[Tensor, Tensor]:
    ids_list = [tokenize_bytes(pretty_print_state(P), cfg.max_len) for P in states]
    L = max(len(x) for x in ids_list)
    L = min(L, cfg.max_len)
    ids = torch.zeros((len(ids_list), L), dtype=torch.long)
    mask = torch.zeros((len(ids_list), L), dtype=torch.bool)
    for i, x in enumerate(ids_list):
        x = x[:L]
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, : len(x)] = True
    return ids, mask
