"""End-to-end smoke test on synthetic data.

Trains every M0–M5 model for a small number of steps and prints metrics.
This verifies the entire pipeline (parser → encoding → model → loss → eval).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.baselines.hypergraph import HyperConfig, make_strong_role_aware, make_weak
from ltm.baselines.role_typed_gnn import GNNConfig, RoleTypedGNN
from ltm.baselines.symbolic import SymbolicMLP, featurise
from ltm.baselines.text_transformer import TextConfig, TextEncoder, collate_text
from ltm.data.dataset import ProofStateDataset
from ltm.data.synthetic import TACTIC_FAMILIES, make_dataset
from ltm.models.rlic_model import RLICConfig, RLICModel
from ltm.parser import build_afford
from ltm.probes import run_all_probes
from ltm.proof_state import ParserConfig
from ltm.train import TrainConfig, train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slice", type=int, default=80)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--models", default="symbolic,text,weak,strong,rtgnn,rlic_struct,rlic_full")
    p.add_argument("--out", default="artifacts/results/smoke")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0); np.random.seed(0)

    cfg = ParserConfig()
    print("=== Building synthetic dataset ===")
    all_states = make_dataset(n_per_slice=args.n_per_slice)
    # 80/10/10 split by index
    n = len(all_states)
    split_a, split_b = int(0.8 * n), int(0.9 * n)
    train_states = all_states[:split_a]
    val_states = all_states[split_a:split_b]
    test_states = all_states[split_b:]
    print(f"  train={len(train_states)} val={len(val_states)} test={len(test_states)}")

    print("=== Algorithm 4 NoLeakageProbes ===")
    states_with_A = [(P, [{"family": "rewrite", "premise": "h"}, {"family": "apply"}], "h")
                     for P in train_states[:60]]
    probe_res = run_all_probes(train_states[:60], states_with_A, cfg)
    probe_summary = {}
    for name, r in probe_res.items():
        probe_summary[name] = r.pass_rate
        print(f"  {name}: {r.n_passed}/{r.n_checked} ({r.pass_rate:.2%})")
    if any(r.pass_rate < 1.0 for r in probe_res.values()):
        print("WARNING: probe drift detected — investigate before launching long runs")

    # premise vocab: collect from data + a dummy pad slot
    premise_vocab = ("<pad>", "h")

    print("=== Parsing dataset ===")
    train_ds = ProofStateDataset(train_states, cfg, TACTIC_FAMILIES, premise_vocab)
    val_ds = ProofStateDataset(val_states, cfg, TACTIC_FAMILIES, premise_vocab)
    test_ds = ProofStateDataset(test_states, cfg, TACTIC_FAMILIES, premise_vocab)
    print(f"  parsed {len(train_ds)} train + {len(val_ds)} val + {len(test_ds)} test")

    train_cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(50, args.steps // 8),
        log_every=max(1, args.steps // 8),
        bf16=(args.device == "cuda"),
    )

    requested = set(args.models.split(","))

    def text_collate_fn(states):
        return collate_text(states, TextConfig())

    def sym_collate_fn(records):
        feats = np.stack([featurise(r.K_afford, r.P_ref) for r in records])
        return torch.tensor(feats, dtype=torch.float32)

    n_families = len(TACTIC_FAMILIES)
    runs = []

    if "symbolic" in requested:
        print("\n=== M0b: Symbolic MLP ===")
        model = SymbolicMLP(n_families=n_families)
        res = train_model(
            model, train_ds, val_ds, train_cfg, args.device,
            symbolic_collate=sym_collate_fn,
            label="symbolic",
            log_path=out_dir / "symbolic.json",
        )
        runs.append(("symbolic", res))

    if "text" in requested:
        print("\n=== M1: Text Transformer ===")
        tcfg = TextConfig(d=args.d, L=args.L, n_families=n_families)
        model = TextEncoder(tcfg)
        res = train_model(
            model, train_ds, val_ds, train_cfg, args.device,
            text_collate=text_collate_fn,
            label="text",
            log_path=out_dir / "text.json",
        )
        runs.append(("text", res))

    if "weak" in requested:
        print("\n=== M2: Weak hypergraph ===")
        hc = HyperConfig(d=args.d, L=args.L, n_families=n_families,
                         use_roles=False, use_hyperedge_state=False)
        model = make_weak(hc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="weak", log_path=out_dir / "weak.json")
        runs.append(("weak", res))

    if "strong" in requested:
        print("\n=== M3: Strong role-aware hypergraph ===")
        hc = HyperConfig(d=args.d, L=args.L, n_families=n_families,
                         use_roles=True, use_hyperedge_state=True)
        model = make_strong_role_aware(hc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="strong", log_path=out_dir / "strong.json")
        runs.append(("strong", res))

    if "rtgnn" in requested:
        print("\n=== M4: Role-typed GNN (grade-0/1 only) ===")
        gc = GNNConfig(d=args.d, L=args.L, n_families=n_families)
        model = RoleTypedGNN(gc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          label="rtgnn", log_path=out_dir / "rtgnn.json")
        runs.append(("rtgnn", res))

    if "rlic_struct" in requested:
        print("\n=== M5a: Structural-only RLIC (K_struct fed to full layer) ===")
        rc = RLICConfig(d=args.d, L=args.L, n_families=n_families)
        model = RLICModel(rc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          use_struct=True,
                          label="rlic_struct", log_path=out_dir / "rlic_struct.json")
        runs.append(("rlic_struct", res))

    if "rlic_full" in requested:
        print("\n=== M5: Full RLIC (K_afford + role-conditioned transport) ===")
        rc = RLICConfig(d=args.d, L=args.L, n_families=n_families)
        model = RLICModel(rc)
        res = train_model(model, train_ds, val_ds, train_cfg, args.device,
                          use_struct=False,
                          label="rlic_full", log_path=out_dir / "rlic_full.json")
        runs.append(("rlic_full", res))

    # Headline table
    print("\n=== Headline results ===")
    cols = ["model", "params(M)", "top1", "top5", "B_mrr", "C_auc", "ms/state"]
    print("  ".join(f"{c:>12}" for c in cols))
    for name, r in runs:
        f = r.final_metrics
        print("  ".join([
            f"{name:>12}",
            f"{r.param_count/1e6:>10.2f}M",
            f"{f['top1']:>12.3f}",
            f"{f['top5']:>12.3f}",
            f"{f['B_mrr']:>12.3f}",
            f"{f['C_auc']:>12.3f}",
            f"{f['ms_per_state']:>10.2f}",
        ]))
    print("\n=== Sliced (Task A top-1) ===")
    slice_names = sorted({n for _, r in runs for n in r.sliced_metrics})
    header = ["model"] + slice_names
    print("  ".join(f"{c:>16}" for c in header))
    for name, r in runs:
        row = [name]
        for sn in slice_names:
            row.append(f"{r.sliced_metrics.get(sn, {}).get('top1', float('nan')):.3f}")
        print("  ".join(f"{c:>16}" for c in row))

    with open(out_dir / "summary.json", "w") as f:
        import json
        json.dump(
            {
                "probes": probe_summary,
                "runs": {
                    name: {
                        "param_count": r.param_count,
                        "final": r.final_metrics,
                        "sliced": r.sliced_metrics,
                    }
                    for name, r in runs
                },
            },
            f, indent=2,
        )
    print(f"\nWrote: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
