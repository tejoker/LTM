"""Ablations A1–A5 on M5 (full RLIC).

A1: collapse all roles into one
A2: freeze ρ^↑, ρ^↓ to identity
A3: drop incidence-spectral PEs (PEs not implemented in this prototype, so this
    is a no-op marker)
A4: drop same-grade neighbourhood N↔
A5: strict cellular-sheaf variant (path identifications) — placeholder, would
    require parser-imposed path identifications in the incidence category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.data.dataset import ProofStateDataset
from ltm.data.synthetic import TACTIC_FAMILIES, make_dataset
from ltm.models.rlic_model import RLICConfig, RLICModel
from ltm.proof_state import ParserConfig
from ltm.train import TrainConfig, train_model


def run_ablation(name: str, rc_overrides: dict, train_ds, val_ds, train_cfg, device, out_dir):
    print(f"\n=== Ablation {name}: {rc_overrides} ===")
    rc = RLICConfig(**rc_overrides)
    model = RLICModel(rc)
    res = train_model(
        model, train_ds, val_ds, train_cfg, device,
        use_struct=False, label=name,
        log_path=out_dir / f"{name}.json",
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-slice", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/results/ablations")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    cfg = ParserConfig()
    all_states = make_dataset(n_per_slice=args.n_per_slice)
    n = len(all_states)
    split_a, split_b = int(0.8 * n), int(0.9 * n)
    train_states = all_states[:split_a]
    val_states = all_states[split_a:split_b]
    train_ds = ProofStateDataset(train_states, cfg, TACTIC_FAMILIES, ("<pad>", "h"))
    val_ds = ProofStateDataset(val_states, cfg, TACTIC_FAMILIES, ("<pad>", "h"))

    train_cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(50, args.steps // 8),
        bf16=(args.device == "cuda"),
    )

    base = dict(n_families=len(TACTIC_FAMILIES), d=args.d, L=args.L)

    runs = []
    # baseline M5
    runs.append(("M5_full", run_ablation("M5_full", base, train_ds, val_ds, train_cfg, args.device, out_dir)))
    # A1: collapse roles
    runs.append(("A1_no_roles", run_ablation("A1_no_roles", {**base, "use_role_labels": False},
                                              train_ds, val_ds, train_cfg, args.device, out_dir)))
    # A2: freeze transport
    runs.append(("A2_freeze_transport", run_ablation("A2_freeze_transport", {**base, "freeze_transport": True},
                                                      train_ds, val_ds, train_cfg, args.device, out_dir)))
    # A3: drop spectral PEs (no-op marker — record param count + result for the row)
    runs.append(("A3_no_pe", run_ablation("A3_no_pe", base,
                                           train_ds, val_ds, train_cfg, args.device, out_dir)))
    # A4: drop same-grade neighbourhood
    runs.append(("A4_no_side", run_ablation("A4_no_side", {**base, "use_side": False},
                                             train_ds, val_ds, train_cfg, args.device, out_dir)))
    # A5: strict sheaf variant — placeholder; full implementation requires
    # parser-imposed path identifications. We record an explicit gap.
    print("\n=== A5_strict_sheaf: placeholder (requires parser-imposed path identifications); skipping. ===")

    print("\n=== Ablation results ===")
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


if __name__ == "__main__":
    main()
