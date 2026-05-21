# LTM-Tiny: Role-Labelled Incidence Complexes for Lean Proof States

Implementation of the §7 empirical protocol from `Papier_de_recherche_partagé (16).pdf`.

## Repo layout

```
ltm/
  rlic.py                 Role-labelled incidence complex data structures (§3.1)
  proof_state.py          ProofState + ParserConfig Π (§3.6, §5.5)
  parser.py               Algorithms 1–3 (BuildStruct, BuildAfford, BuildCandidateAction)
                          + canonical truncation (§5.5)
  probes.py               Algorithm 4 NoLeakageProbes (tactic-swap, permuted-context,
                          held-out-lemma)
  models/
    rlic_layer.py         §5 architecture: cell encoder, role-conditioned attention
                          (↓, ↑, ↔), layer update, class-canonical pool
    rlic_model.py         M5 full RLIC + ablation toggles (A1, A2, A4)
    heads.py              Policy / Value / Retrieval heads (§5.6, Mode 1)
  baselines/
    hypergraph.py         M2 weak + M3 strong role-aware hypergraph
    role_typed_gnn.py     M4 role-typed GNN over grade-0/grade-1 only
    symbolic.py           M0 affordance-count features + M0b symbolic MLP
    text_transformer.py   M1 byte-level text Transformer over pretty-printed state
  data/
    synthetic.py          Synthetic ProofState generator for pipeline verification
    leandojo_adapter.py   LeanDojo TacticState → ProofState (scaffold; untested live)
    dataset.py            ProofStateDataset, structural slice categorisation (§7.5)
    encoding.py           RLIC → CellBatch tensor encoding
  eval/
    metrics.py            Top-k / MRR / Recall@k / AUC / ECE
  train.py                Uniform training & evaluation loop (Tasks A/B/C)
scripts/
  run_smoke.py            Train all 7 models on synthetic + run Algorithm 4 probes
  run_ablations.py        Phase 3 ablations A1–A4 on M5 (A5 placeholder)
  compile_results.py      Headline + sliced + latency + param-count table
artifacts/
  results/                Per-model JSON outputs
```

## Hardware

Verified on **NVIDIA GeForce RTX 4060 Ti 16 GB**, compute 8.9 (Ada Lovelace),
driver 580.142, CUDA toolkit 12.4, PyTorch 2.6.0+cu124, BF16 mixed precision.

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy pyarrow scikit-learn tqdm pytest lean-dojo
```

## Phase 0: smoke + probes

```bash
.venv/bin/python scripts/run_smoke.py \
  --n-per-slice 200 --steps 800 --batch-size 16 --d 128 --L 2
```

Runs Algorithm 4 NoLeakageProbes (must report 100% pass-rate before training),
then trains M0–M5 on synthetic data and dumps per-model JSON to
`artifacts/results/smoke/`.

## Phase 3: ablations

```bash
.venv/bin/python scripts/run_ablations.py \
  --n-per-slice 200 --steps 600 --batch-size 16 --d 128 --L 2
```

A1 collapse roles, A2 freeze transport, A3 drop PEs (no-op marker), A4 drop
same-grade neighbourhood. A5 strict-sheaf placeholder requires parser-imposed
path identifications in `ltm/parser.py` (future work).

## Phase 1/2: real-data runs (BLOCKED on data)

Status: **scaffolded but not run.** The synthetic generator saturates all
models at 100% top-1, so it can verify the pipeline but cannot discriminate
models — exactly the situation §7 anticipates for any non-trivial baseline.

Two paths to unblock:

1. **LeanDojo pre-traced benchmark.** Download `leandojo_benchmark_4` from
   <https://leandojo.org/>, then convert to ProofStates with
   `ltm.data.leandojo_adapter.from_jsonl`. The adapter currently uses a coarse
   text parse; for production, integrate LeanDojo's parsed AST via
   `lean_dojo.parse_goals`.

2. **Live mathlib4 tracing.** Install elan + Lean toolchain, then
   `lean_dojo.trace(LeanGitRepo("leanprover-community/mathlib4", rev))`. Needs
   ~50 GB free disk + several hours; the current 4060 Ti host has 65 GB free,
   which is too tight.

LTM-Tiny config for full runs (paper §7.4): `d=256, L=4, H=4`, budgets
`(|K_0|, |K_1|, |K_≥2|) = (256, 512, 64)`, batch 16 + grad accum 2 (→ effective
32), AdamW lr 3e-4 cosine, 80k steps. Wall-clock estimate per single seed
on this 4060 Ti: ~10–14 GPU-hours per model. Phase 1 head-to-head
(M0/M0b, M1, M3, M5 single seed): ~50 GPU-hours. Phase 2 (M2, M4, M5a + 2
extra seeds for M3 and M5): ~80 GPU-hours. Phase 3 ablations: ~50 GPU-hours.

**Parameter-count match for the head-to-head M3 vs M5 is load-bearing**
(paper §7.3). The current code emits ~36M params for M3 and ~41M for M5; tune
`HyperConfig.d` upward for M3 before the real headline run.

## Compile results

```bash
.venv/bin/python scripts/compile_results.py
```

Prints the headline + sliced tables.
