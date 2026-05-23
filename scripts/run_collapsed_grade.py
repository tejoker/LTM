"""Train the collapsed-grade M5 with rich transport (M3+) ablation.

This is the "M3 + side direction" model: hyperedges (no graded levels) +
three role-conditioned attention directions (↑ node→HE, ↓ HE→node,
↔ node↔node via shared HE). Parameter-matched to M5 at d=384.

Per the paper (§7.8), this ablation isolates whether the cellular *grading*
itself contributes beyond rich role-conditioned transport over typed
hyperedges.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.baselines.hypergraph import HyperConfig, HypergraphModel, make_collapsed_grade_rich
from ltm.data.dataset import load_dataset
from ltm.train import TrainConfig, train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_full")
    p.add_argument("--out-dir", default="artifacts/results/phase3_grading")
    p.add_argument("--d", type=int, default=384,
                   help="Param-matched to M5 (36.04M) at d=384 -> 35.92M, Δ=-0.3%")
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_ds = load_dataset(f"{args.data_dir}/train.pkl")
    val_ds = load_dataset(f"{args.data_dir}/val.pkl")
    n_families = len(train_ds.family_vocab)

    cfg = HyperConfig(
        d=args.d, L=args.L, n_families=n_families,
        use_roles=True, use_hyperedge_state=True, use_side=True,
    )
    model = make_collapsed_grade_rich(cfg)

    train_cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup,
        log_every=max(50, args.steps // 50),
        bf16=(args.device == "cuda"),
        n_premise_buckets=len(train_ds.premise_vocab),
    )

    seed_suffix = f"_s{args.seed}" if args.seed != 0 else ""
    label = f"collapsed_grade_rich{seed_suffix}"
    print(f"\n=== M3+ collapsed-grade + rich transport (seed {args.seed}, d={args.d}, L={args.L}) ===")
    t0 = time.perf_counter()
    res = train_model(
        model, train_ds, val_ds, train_cfg, args.device,
        use_struct=False, label=label,
        log_path=out_dir / f"{label}.json",
    )
    print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
    f = res.final_metrics
    print(f"  top1={f['top1']:.4f}  top5={f['top5']:.4f}  C_AUC={f['C_auc']:.4f}  "
          f"ms/state={f['ms_per_state']:.2f}")


if __name__ == "__main__":
    main()
