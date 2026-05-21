"""Algorithm 4 NoLeakageProbes on real LeanDojo proof states.

Paper §7.6: probe pass-rate is necessary, not sufficient, and must be co-reported
alongside task metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.data.dataset import load_dataset
from ltm.probes import run_all_probes
from ltm.proof_state import ParserConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to parsed pickle")
    p.add_argument("--n", type=int, default=200,
                   help="Number of states to probe (probes are O(n) parsings each)")
    p.add_argument("--out", default="artifacts/results/probes_real.json")
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    print(f"Loaded {len(ds.records)} records from {args.dataset}")

    states = [r.P_ref for r in ds.records[: args.n]]
    print(f"Probing first {len(states)} states...")
    states_with_A = [
        (P,
         [{"family": "rewrite", "premise": P.next_tactic_premises[0] if P.next_tactic_premises else ""},
          {"family": "apply"}],
         P.next_tactic_premises[0] if P.next_tactic_premises else "")
        for P in states[:50]  # smaller for probe 3 since it's O(states * |A|)
    ]
    cfg = ParserConfig()
    res = run_all_probes(states, states_with_A, cfg)
    summary = {}
    for name, r in res.items():
        summary[name] = {"pass": r.n_passed, "total": r.n_checked,
                          "pass_rate": r.pass_rate, "failing": r.failing[:5]}
        print(f"  {name}: {r.n_passed}/{r.n_checked} ({r.pass_rate:.2%})")
        if r.failing:
            for f in r.failing[:3]:
                print(f"    fail: {f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
