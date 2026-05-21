"""Find a HyperConfig.d for M3 that matches M5's param count at given (d, L)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.baselines.hypergraph import HyperConfig, HypergraphModel
from ltm.models.rlic_model import RLICConfig, RLICModel


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--n-families", type=int, default=9)
    args = p.parse_args()

    rc = RLICConfig(d=args.d, L=args.L, n_families=args.n_families)
    m5 = RLICModel(rc)
    target = n_params(m5)
    print(f"M5 @ d={args.d} L={args.L}: {target/1e6:.2f}M params")

    print(f"\nSweeping HyperConfig.d to find param-match for M3 (use_roles=True, use_hyperedge_state=True):")
    results = []
    for d3 in range(64, 512, 16):
        hc = HyperConfig(d=d3, L=args.L, n_families=args.n_families,
                          use_roles=True, use_hyperedge_state=True)
        m3 = HypergraphModel(hc)
        np_ = n_params(m3)
        results.append((d3, np_))
        delta = (np_ - target) / target
        marker = "  <-- closest above" if np_ >= target and (not results[:-1] or results[-2][1] < target) else ""
        print(f"  d={d3:>4}: {np_/1e6:>7.2f}M  (Δ={delta:+.2%}){marker}")
        if np_ > 1.3 * target:
            break

    best = min(results, key=lambda r: abs(r[1] - target))
    print(f"\nBest match: d={best[0]} -> {best[1]/1e6:.2f}M (target {target/1e6:.2f}M, Δ={(best[1]-target)/target:+.2%})")


if __name__ == "__main__":
    main()
