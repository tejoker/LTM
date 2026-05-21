"""Compile final results tables (paper §7 headline + sliced + latency + params).

Reads the per-model JSON outputs from artifacts/results/ and prints the
tables that go into the empirical section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-dir", default="artifacts/results/smoke")
    p.add_argument("--abl-dir", default="artifacts/results/ablations")
    args = p.parse_args()

    smoke_dir = Path(args.smoke_dir)
    abl_dir = Path(args.abl_dir)

    rows = []
    for jp in sorted(smoke_dir.glob("*.json")):
        if jp.name == "summary.json":
            continue
        d = load(jp)
        rows.append(("Phase 1", d["label"], d["param_count"], d["final"], d["sliced"], d["elapsed_s"]))
    for jp in sorted(abl_dir.glob("*.json")):
        d = load(jp)
        rows.append(("Phase 3", d["label"], d["param_count"], d["final"], d["sliced"], d["elapsed_s"]))

    print("="*100)
    print("Headline table (Task A top-1, Task B MRR, Task C AUC, params, ms/state)")
    print("="*100)
    print(f"{'phase':<10} {'model':<22} {'params(M)':>10}  {'top1':>6}  {'top5':>6}  {'B_mrr':>6}  {'C_auc':>6}  {'ms':>7}  {'wall(s)':>8}")
    for phase, lbl, np_, final, sliced, elapsed in rows:
        print(f"{phase:<10} {lbl:<22} {np_/1e6:>9.2f}M  {final['top1']:>6.3f}  {final['top5']:>6.3f}  {final['B_mrr']:>6.3f}  {final['C_auc']:>6.3f}  {final['ms_per_state']:>6.2f}  {elapsed:>8.1f}")

    print()
    print("="*100)
    print("Sliced Task A top-1 (paper §7.5)")
    print("="*100)
    slices = sorted({s for _, _, _, _, sl, _ in rows for s in sl.keys()})
    header = f"{'phase':<10} {'model':<22}  " + "  ".join(f"{s:>18}" for s in slices)
    print(header)
    for phase, lbl, _, _, sliced, _ in rows:
        cells = []
        for s in slices:
            v = sliced.get(s, {}).get("top1", float("nan"))
            cells.append(f"{v:>18.3f}")
        print(f"{phase:<10} {lbl:<22}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
