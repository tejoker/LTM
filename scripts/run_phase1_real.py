"""Phase 1 head-to-head on real LeanDojo proof states.

LTM-Tiny config (§7.4): d=256, L=4, H=4. M3 param-matched at d=448.

Models: M0b symbolic, M1 text, M2 weak, M3 strong (matched), M4 rtgnn,
M5a structural-only RLIC, M5 full RLIC. Single seed for v1.
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

from ltm.baselines.hypergraph import HyperConfig, HypergraphModel, make_weak
from ltm.baselines.role_typed_gnn import GNNConfig, RoleTypedGNN
from ltm.baselines.symbolic import SymbolicMLP, featurise
from ltm.baselines.text_transformer import TextConfig, TextEncoder, collate_text
from ltm.data.dataset import load_dataset
from ltm.data.leandojo_adapter import TACTIC_FAMILIES as REAL_TACTIC_FAMILIES
from ltm.models.rlic_model import RLICConfig, RLICModel
from ltm.train import TrainConfig, train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_filtered")
    p.add_argument("--out-dir", default="artifacts/results/phase1_real")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--m3-d", type=int, default=448,
                   help="M3 hypergraph d, param-matched to M5 at d=256/L=4")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", default="symbolic,text,weak,strong,rtgnn,rlic_struct,rlic_full")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    seed_suffix = f"_s{args.seed}" if args.seed != 0 else ""

    print(f"Loading datasets from {args.data_dir}...")
    train_ds = load_dataset(f"{args.data_dir}/train.pkl")
    val_ds = load_dataset(f"{args.data_dir}/val.pkl")
    print(f"  train={len(train_ds)} val={len(val_ds)}")

    n_families = len(train_ds.family_vocab)
    print(f"  family vocab size: {n_families}  ({train_ds.family_vocab})")

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

    requested = set(args.models.split(","))
    runs = []

    def text_collate_fn(states):
        return collate_text(states, TextConfig(d=args.d, L=args.L, n_families=n_families))

    def sym_collate_fn(records):
        feats = np.stack([featurise(r.K_afford, r.P_ref) for r in records])
        return torch.tensor(feats, dtype=torch.float32)

    if "symbolic" in requested:
        print("\n=== M0b: Symbolic MLP ===")
        t0 = time.perf_counter()
        model = SymbolicMLP(n_families=n_families)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          symbolic_collate=sym_collate_fn,
                          label="symbolic",
                          log_path=out_dir / f"symbolic{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("symbolic", res))

    if "text" in requested:
        print("\n=== M1: Text Transformer ===")
        t0 = time.perf_counter()
        tcfg = TextConfig(d=args.d, L=args.L, n_families=n_families)
        model = TextEncoder(tcfg)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          text_collate=text_collate_fn,
                          label="text",
                          log_path=out_dir / f"text{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("text", res))

    if "weak" in requested:
        print("\n=== M2: Weak hypergraph ===")
        t0 = time.perf_counter()
        hc = HyperConfig(d=args.d, L=args.L, n_families=n_families,
                          use_roles=False, use_hyperedge_state=False)
        model = make_weak(hc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="weak", log_path=out_dir / f"weak{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("weak", res))

    if "strong" in requested:
        print(f"\n=== M3: Strong role-aware hypergraph (d={args.m3_d}, param-matched) ===")
        t0 = time.perf_counter()
        hc = HyperConfig(d=args.m3_d, L=args.L, n_families=n_families,
                          use_roles=True, use_hyperedge_state=True)
        model = HypergraphModel(hc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="strong", log_path=out_dir / f"strong{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("strong", res))

    if "rtgnn" in requested:
        print("\n=== M4: Role-typed GNN ===")
        t0 = time.perf_counter()
        gc = GNNConfig(d=args.d, L=args.L, n_families=n_families)
        model = RoleTypedGNN(gc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="rtgnn", log_path=out_dir / f"rtgnn{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("rtgnn", res))

    if "rlic_struct" in requested:
        print("\n=== M5a: Structural-only RLIC ===")
        t0 = time.perf_counter()
        rc = RLICConfig(d=args.d, L=args.L, n_families=n_families)
        model = RLICModel(rc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          use_struct=True,
                          label="rlic_struct", log_path=out_dir / f"rlic_struct{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("rlic_struct", res))

    if "rlic_full" in requested:
        print("\n=== M5: Full RLIC ===")
        t0 = time.perf_counter()
        rc = RLICConfig(d=args.d, L=args.L, n_families=n_families)
        model = RLICModel(rc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          use_struct=False,
                          label="rlic_full", log_path=out_dir / f"rlic_full{seed_suffix}.json")
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s  params: {res.param_count/1e6:.2f}M")
        runs.append(("rlic_full", res))

    # Headline table
    print("\n=== Phase 1 headline (real LeanDojo subset) ===")
    cols = ["model", "params(M)", "top1", "top5", "B_mrr", "C_auc", "ms/state"]
    print("  ".join(f"{c:>14}" for c in cols))
    for name, r in runs:
        f = r.final_metrics
        print("  ".join([
            f"{name:>14}",
            f"{r.param_count/1e6:>12.2f}M",
            f"{f['top1']:>14.3f}",
            f"{f['top5']:>14.3f}",
            f"{f['B_mrr']:>14.3f}",
            f"{f['C_auc']:>14.3f}",
            f"{f['ms_per_state']:>12.2f}",
        ]))
    print("\n=== Sliced Task A top-1 ===")
    slice_names = sorted({n for _, r in runs for n in r.sliced_metrics})
    if slice_names:
        header = ["model"] + slice_names
        print("  ".join(f"{c:>18}" for c in header))
        for name, r in runs:
            row = [name]
            for sn in slice_names:
                row.append(f"{r.sliced_metrics.get(sn, {}).get('top1', float('nan')):.3f}")
            print("  ".join(f"{c:>18}" for c in row))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "runs": {
                    name: {"param_count": r.param_count, "final": r.final_metrics,
                           "sliced": r.sliced_metrics}
                    for name, r in runs
                },
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
