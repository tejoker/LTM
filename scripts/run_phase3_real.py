"""Phase 3 ablations A1-A4 on M5 (real LeanDojo data).

A5 strict-sheaf placeholder requires parser-imposed path identifications
(future work) and is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.data.dataset import load_dataset
from ltm.models.rlic_model import RLICConfig, RLICModel
from ltm.train import TrainConfig, train_model


ABLATIONS = [
    ("M5_full",            {}),
    ("A1_no_roles",        {"use_role_labels": False}),
    ("A2_freeze_transport",{"freeze_transport": True}),
    ("A3_no_pe",           {}),  # PE not implemented; baseline marker only
    ("A4_no_side",         {"use_side": False}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_filtered")
    p.add_argument("--out-dir", default="artifacts/results/phase3_real")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_ds = load_dataset(f"{args.data_dir}/train.pkl")
    val_ds = load_dataset(f"{args.data_dir}/val.pkl")
    n_families = len(train_ds.family_vocab)

    base = dict(n_families=n_families, d=args.d, L=args.L)
    train_cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_steps=200,
        log_every=max(50, args.steps // 50),
        bf16=(args.device == "cuda"),
        n_premise_buckets=len(train_ds.premise_vocab),
    )

    runs = []
    for name, overrides in ABLATIONS:
        print(f"\n=== {name} : {overrides} ===")
        t0 = time.perf_counter()
        rc = RLICConfig(**{**base, **overrides})
        model = RLICModel(rc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          use_struct=False, label=name,
                          log_path=out_dir / f"{name}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append((name, res))

    print("\n=== Phase 3 ablations (real LeanDojo subset) ===")
    cols = ["ablation", "params(M)", "top1", "top5", "B_mrr", "C_auc", "ms/state"]
    print("  ".join(f"{c:>20}" for c in cols))
    for name, r in runs:
        f = r.final_metrics
        print("  ".join([
            f"{name:>20}",
            f"{r.param_count/1e6:>18.2f}M",
            f"{f['top1']:>20.3f}",
            f"{f['top5']:>20.3f}",
            f"{f['B_mrr']:>20.3f}",
            f"{f['C_auc']:>20.3f}",
            f"{f['ms_per_state']:>18.2f}",
        ]))

    print("\n=== Sliced Task A top-1 (gap on rewrite-heavy + multi-arg app per §7.7) ===")
    slice_names = sorted({n for _, r in runs for n in r.sliced_metrics})
    if slice_names:
        header = ["ablation"] + slice_names
        print("  ".join(f"{c:>20}" for c in header))
        for name, r in runs:
            row = [name]
            for sn in slice_names:
                row.append(f"{r.sliced_metrics.get(sn, {}).get('top1', float('nan')):.3f}")
            print("  ".join(f"{c:>20}" for c in row))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"runs": {name: {"param_count": r.param_count, "final": r.final_metrics,
                              "sliced": r.sliced_metrics} for name, r in runs}},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
