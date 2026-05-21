"""Clean LTM-Tiny full run (paper §7.4 config).

Sequences:
  - Phase 1   : M0b, M1, M2, M3 (param-matched), M4, M5a, M5  — single seed
  - Phase 1b  : M3 and M5 with two extra seeds                — variance bars
  - Phase 3   : M5 ablations A1, A2, A3, A4                   — single seed

Each run is one Python subprocess that writes its own JSON. Resume-aware:
runs whose output JSON already exists are skipped — so this orchestrator can
be re-launched after a crash without losing progress.

Default config (paper §7.4):
  d=256, L=4, H=4, batch=32 (effective 32, grad_accum=1), AdamW lr=3e-4,
  cosine to 1e-5, 800 warmup steps, 80k total steps.
M3 is param-matched at d=448.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


@dataclass
class Run:
    label: str           # short id, used for the JSON name
    phase: str           # "phase1" | "phase1b" | "phase3"
    runner: str          # which script to invoke
    args: list[str]      # command-line args

    @property
    def out_path(self) -> Path:
        return ROOT / "artifacts" / "results" / self.phase / f"{self.label}.json"


def runs_for_phase(args) -> list[Run]:
    """Build the full list of runs (Phase 1 + 1b + 3)."""
    base_args = [
        "--data-dir", args.data_dir,
        "--d", str(args.d),
        "--L", str(args.L),
        "--m3-d", str(args.m3_d),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--grad-accum", "1",
    ]
    runs: list[Run] = []

    # Phase 1 — all seven models, single seed=0
    for model in ["symbolic", "text", "weak", "strong", "rtgnn", "rlic_struct", "rlic_full"]:
        runs.append(Run(
            label=f"{model}_s0",
            phase="phase1",
            runner="scripts/run_phase1_real.py",
            args=base_args + [
                "--models", model,
                "--seed", "0",
                "--out-dir", "artifacts/results/phase1",
            ],
        ))

    # Phase 1b — M3 and M5 with seeds 1, 2 (for variance bars)
    for seed in (1, 2):
        for model in ("strong", "rlic_full"):
            runs.append(Run(
                label=f"{model}_s{seed}",
                phase="phase1b",
                runner="scripts/run_phase1_real.py",
                args=base_args + [
                    "--models", model,
                    "--seed", str(seed),
                    "--out-dir", "artifacts/results/phase1b",
                ],
            ))

    # Phase 3 — M5 ablations (single seed=0)
    # The Phase 3 driver runs all 5 in one process; we keep it monolithic so the
    # ablation comparison shares a process / cache.
    runs.append(Run(
        label="ablations_s0",
        phase="phase3",
        runner="scripts/run_phase3_real.py",
        args=[
            "--data-dir", args.data_dir,
            "--d", str(args.d),
            "--L", str(args.L),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--grad-accum", "1",
            "--seed", "0",
            "--out-dir", "artifacts/results/phase3",
        ],
    ))

    return runs


def already_done(run: Run) -> bool:
    """Phase-1 / 1b runs: check for the model+seed JSON.
    Phase 3: check that all five ablation JSONs exist."""
    if run.phase in ("phase1", "phase1b"):
        model = run.args[run.args.index("--models") + 1]
        seed = int(run.args[run.args.index("--seed") + 1])
        suffix = f"_s{seed}" if seed != 0 else ""
        out_dir = ROOT / run.args[run.args.index("--out-dir") + 1]
        return (out_dir / f"{model}{suffix}.json").exists()
    if run.phase == "phase3":
        out_dir = ROOT / run.args[run.args.index("--out-dir") + 1]
        wanted = ["M5_full", "A1_no_roles", "A2_freeze_transport",
                  "A3_no_pe", "A4_no_side"]
        return all((out_dir / f"{n}.json").exists() for n in wanted)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_full")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--m3-d", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--only", default=None,
                   help="Comma-separated phases to run (phase1,phase1b,phase3)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs = runs_for_phase(args)
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if r.phase in wanted]

    print(f"Plan: {len(runs)} runs")
    for r in runs:
        status = "SKIP (already done)" if already_done(r) else "RUN"
        print(f"  [{r.phase:>7}] {r.label:<20} {status}")
    if args.dry_run:
        return

    overall_t0 = time.perf_counter()
    for i, r in enumerate(runs, 1):
        r.out_path.parent.mkdir(parents=True, exist_ok=True)
        if already_done(r):
            print(f"\n[{i}/{len(runs)}] SKIP {r.label} (already done)")
            continue
        print(f"\n[{i}/{len(runs)}] === RUN {r.label} ({r.phase}) ===")
        cmd = [PY, "-u", r.runner] + r.args
        print("  cmd:", " ".join(cmd))
        log_path = ROOT / "artifacts" / "results" / r.phase / f"{r.label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode}) after {dt:.0f}s — log: {log_path}")
        else:
            print(f"  done in {dt/60:.1f} min  (cumulative {((time.perf_counter()-overall_t0)/3600):.2f} h)")

    print(f"\nFull sweep complete in {(time.perf_counter()-overall_t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
