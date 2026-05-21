"""Metrics for Tasks A/B/C."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Task A: tactic-family classification
# ---------------------------------------------------------------------------


def task_a_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    top1 = (pred == target).float().mean().item()
    k = min(5, probs.shape[1])
    _, topk_idx = probs.topk(k, dim=-1)
    top5 = (topk_idx == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
    ce = torch.nn.functional.cross_entropy(logits, target).item()
    return {"top1": top1, "top5": top5, "ce": ce}


# ---------------------------------------------------------------------------
# Task B: premise retrieval
# ---------------------------------------------------------------------------


def task_b_metrics(scores: Tensor, target_premise_idx: Tensor) -> dict[str, float]:
    """scores: [B, N_premises]; target: [B] index of the correct premise."""
    # Recall@k and MRR
    B = scores.shape[0]
    sorted_idx = scores.argsort(dim=-1, descending=True)
    rank = (sorted_idx == target_premise_idx.unsqueeze(-1)).float().argmax(dim=-1)
    # if target not present at all (shouldn't happen here), argmax returns 0
    in_top10 = (rank < 10).float().mean().item()
    in_top50 = (rank < 50).float().mean().item()
    mrr = (1.0 / (rank.float() + 1)).mean().item()
    return {"recall@10": in_top10, "recall@50": in_top50, "mrr": mrr}


# ---------------------------------------------------------------------------
# Task C: value prediction
# ---------------------------------------------------------------------------


def task_c_metrics(probs: Tensor, target: Tensor) -> dict[str, float]:
    """probs: [B] in [0,1]; target: [B] in {0,1}."""
    pred = (probs > 0.5).float()
    acc = (pred == target).float().mean().item()
    # AUC via Mann-Whitney
    p = probs.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    if len(np.unique(t)) < 2:
        auc = 0.5
    else:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(t, p))
    # 15-bin ECE
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < 14 else p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc_b = t[mask].mean()
        ece += (mask.sum() / len(p)) * abs(conf - acc_b)
    return {"auc": auc, "acc": acc, "ece": float(ece)}
