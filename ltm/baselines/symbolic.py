"""M0 / M0b: symbolic affordance-count baselines (§7.3).

A feature vector of per-class affordance-cell counts plus basic syntactic
features (goal head, number of hypotheses, presence of equalities, etc.), fed
to either logistic regression or a small MLP.

If this baseline matches the model on Task A, then the model's contribution
lies in Tasks B/C and in the structurally complex slices of Task A.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..proof_state import ProofState
from ..rlic import RLIC, CellLabel, AFFORDANCE_LABELS


# We use a fixed bucket per cell label + a few syntactic features.
_LABEL_BUCKETS = [
    int(CellLabel.VARIABLE),
    int(CellLabel.LOCAL_HYP),
    int(CellLabel.GOAL),
    int(CellLabel.CONSTANT),
    int(CellLabel.SUBTERM),
    int(CellLabel.HAS_TYPE),
    int(CellLabel.OCCURS_IN),
    int(CellLabel.PROVES),
    int(CellLabel.BINDS),
    int(CellLabel.APP_N),
    int(CellLabel.EQ_LIT),
    int(CellLabel.REWRITE_OPP),
    int(CellLabel.APPLY_OPP),
    int(CellLabel.INDUCTION_OPP),
    int(CellLabel.CONSTRUCTOR_OPP),
]
N_BUCKETS = len(_LABEL_BUCKETS)
N_SYNTACTIC = 5  # n_hyps, n_eq_hyps, n_proof_hyps, goal_arity, goal_is_eq


def featurise(K: RLIC, P: ProofState) -> np.ndarray:
    counts = np.zeros(N_BUCKETS, dtype=np.float32)
    for c in K.cells:
        for i, lab in enumerate(_LABEL_BUCKETS):
            if c.label == lab:
                counts[i] += 1
                break

    syntactic = np.zeros(N_SYNTACTIC, dtype=np.float32)
    syntactic[0] = len(P.gamma)
    syntactic[1] = sum(1 for h in P.gamma if h.typ.op == "eq")
    syntactic[2] = sum(1 for h in P.gamma if h.is_proof)
    if P.goal is not None:
        syntactic[3] = len(P.goal.children) if P.goal.op == "app" else 0
        syntactic[4] = 1.0 if P.goal.op == "eq" else 0.0

    return np.concatenate([counts, syntactic])


class SymbolicMLP(nn.Module):
    """M0b: small MLP on the symbolic feature vector."""

    def __init__(self, n_families: int, n_premise_buckets: int = 4096, hidden: int = 64):
        super().__init__()
        in_dim = N_BUCKETS + N_SYNTACTIC
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.policy = nn.Linear(hidden, n_families)
        self.value = nn.Linear(hidden, 1)
        self.retrieval = nn.Linear(hidden, 128)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        z = self.retrieval(h)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "pooled": h,
            "policy_logits": self.policy(h),
            # logit; train loop applies sigmoid for metrics, BCEWithLogits for loss
            "value": self.value(h).squeeze(-1),
            "retrieval_query": z,
        }
