"""Compile Phase 1 (head-to-head) + Phase 3 (ablations) real-data results.

Reads:
  artifacts/results/phase1_real/*.json
  artifacts/results/phase3_real/*.json
  artifacts/results/probes_real.json

Emits the headline + sliced + ablation tables and writes a final
RESULTS_REAL.md in the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def load_run_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def headline_row(name: str, d: dict) -> str:
    f = d["final"]
    return (f"| {name} | {d['param_count']/1e6:.2f}M | "
            f"{f['top1']:.3f} | {f['top5']:.3f} | "
            f"{f['B_mrr']:.3f} | {f['B_recall@10']:.3f} | "
            f"{f['C_auc']:.3f} | {f['ms_per_state']:.2f} |")


def main():
    root = Path("artifacts/results")
    phase1 = root / "phase1_real"
    phase3 = root / "phase3_real"

    probes = load_run_json(root / "probes_real.json") or {}

    # Phase 1 runs
    p1_runs = {}
    for p in sorted(phase1.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = load_run_json(p)
        if d:
            p1_runs[d["label"]] = d

    p3_runs = {}
    for p in sorted(phase3.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = load_run_json(p)
        if d:
            p3_runs[d["label"]] = d

    out_md = []
    out_md.append("# LTM-Tiny: Real-Data Empirical Results\n")
    out_md.append("Run on a filtered LeanDojo Benchmark 4 subset "
                  "(algebra / logic / numerics, paper §7.1) at LTM-Tiny config "
                  "(d=256, L=4, H=4). Single seed; M3 hypergraph param-matched "
                  "to M5 at d=448. Hardware: RTX 4060 Ti 16 GB, BF16.\n")
    out_md.append("## Algorithm 4 No-Leakage Probes (necessary condition)\n")
    out_md.append("| probe | pass / total | pass-rate |")
    out_md.append("|---|---|---|")
    for k in ("tactic_swap", "permuted_context", "heldout_lemma"):
        if k in probes:
            r = probes[k]
            out_md.append(f"| {k} | {r['pass']} / {r['total']} | {r['pass_rate']:.2%} |")
    out_md.append("")

    if p1_runs:
        out_md.append("## Phase 1 — Head-to-Head\n")
        out_md.append("| model | params | top-1 | top-5 | B MRR | B R@10 | C AUC | ms/state |")
        out_md.append("|---|---|---|---|---|---|---|---|")
        order = ["symbolic", "text", "weak", "strong", "rtgnn", "rlic_struct", "rlic_full"]
        for k in order:
            if k in p1_runs:
                out_md.append(headline_row(k, p1_runs[k]))
        out_md.append("")
        # Sliced
        all_slices = sorted({s for d in p1_runs.values() for s in d.get("sliced", {})})
        if all_slices:
            out_md.append("### Sliced top-1 (§7.5)\n")
            out_md.append("| model | " + " | ".join(all_slices) + " |")
            out_md.append("|" + "---|" * (len(all_slices) + 1))
            for k in order:
                if k not in p1_runs:
                    continue
                row = [k]
                for s in all_slices:
                    v = p1_runs[k].get("sliced", {}).get(s, {}).get("top1")
                    row.append(f"{v:.3f}" if v is not None else "—")
                out_md.append("| " + " | ".join(row) + " |")
            out_md.append("")

    if p3_runs:
        out_md.append("## Phase 3 — Ablations on M5\n")
        out_md.append("| ablation | params | top-1 | top-5 | B MRR | B R@10 | C AUC | ms/state |")
        out_md.append("|---|---|---|---|---|---|---|---|")
        for k in ["M5_full", "A1_no_roles", "A2_freeze_transport", "A3_no_pe", "A4_no_side"]:
            if k in p3_runs:
                out_md.append(headline_row(k, p3_runs[k]))
        out_md.append("")

    out_md.append("## Caveats\n")
    out_md.append(dedent("""
        - **Single seed, single subset (15k train / 1.5k val / 1.5k test).** Paper §7.4 calls for 3 seeds on the M3↔M5 head-to-head; not exercised here.
        - **Coarse expression parser.** [ltm/data/leandojo_adapter.py](ltm/data/leandojo_adapter.py) uses a heuristic top-level-paren parser, not `lean_dojo.parse_goals`. Some type tags will be `UNKNOWN` and some operators degrade to generic application. This adds noise across *all* models uniformly but is a real source of error vs. an AST-grade adapter.
        - **Task C target is degenerate** on this dataset. Every observed state is reachable by *some* recorded tactic, so the value target is constant 1 in train and val. C_AUC = 0.5 by construction. The paper anticipates this; a real Task C target requires negative state mining.
        - **Steps = 2500** (~5 epochs). Paper §7.4 specifies 80k. Results should be read as *learning signal*, not as the converged ceiling.
    """).strip())

    text = "\n".join(out_md)
    with open("RESULTS_REAL.md", "w") as f:
        f.write(text)
    print(text)
    print("\nWrote: RESULTS_REAL.md")


if __name__ == "__main__":
    main()
