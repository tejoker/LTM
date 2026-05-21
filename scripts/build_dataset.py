"""Build a filtered ProofStateDataset from LeanDojo Benchmark 4 and cache it.

Filters by file_path prefix (paper §7.1: algebra, logic, basic numerics) and
caps proof length. Caches the result as a pickle (Arrow-mmap can be done later
if the dataset grows past RAM).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.data.dataset import ProofStateDataset, precompute_encodings, save_dataset
from ltm.data.leandojo_adapter import (
    DEFAULT_FILE_PREFIXES, TACTIC_FAMILIES, build_premise_vocab,
    build_premise_vocab_from_corpus, family_distribution,
    from_split_file, synthesize_value_negatives,
)
from ltm.proof_state import ParserConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-root", required=True,
                   help="Path to mathlib4_<commit>/ inside the unpacked benchmark")
    p.add_argument("--split", default="random", choices=["random", "novel_premises"])
    p.add_argument("--out-dir", default="artifacts/parses/mathlib_filtered")
    p.add_argument("--max-train", type=int, default=20000)
    p.add_argument("--max-val", type=int, default=2000)
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--proof-max-length", type=int, default=30)
    p.add_argument("--goal-max-chars", type=int, default=1024)
    p.add_argument("--no-filter", action="store_true",
                   help="Skip file-prefix filter (use whole mathlib)")
    p.add_argument("--corpus-jsonl", default=None,
                   help="Path to corpus.jsonl for global Task B premise pool")
    p.add_argument("--premise-vocab-size", type=int, default=16384)
    p.add_argument("--value-negatives", type=float, default=0.3,
                   help="Fraction of synthesized Task C negatives to add per split")
    args = p.parse_args()

    bench = Path(args.bench_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefixes = () if args.no_filter else DEFAULT_FILE_PREFIXES
    cfg = ParserConfig()

    splits = {
        "train": (bench / args.split / "train.json", args.max_train),
        "val":   (bench / args.split / "val.json",   args.max_val),
        "test":  (bench / args.split / "test.json",  args.max_test),
    }
    all_states: dict[str, list] = {}
    for name, (path, cap) in splits.items():
        print(f"[{name}] parsing {path}  (cap={cap})")
        t0 = time.perf_counter()
        states = from_split_file(
            path,
            file_prefixes=prefixes,
            max_states=cap,
            goal_max_chars=args.goal_max_chars,
            proof_max_length=args.proof_max_length,
        )
        dt = time.perf_counter() - t0
        print(f"  {len(states)} ProofStates in {dt:.1f}s")
        print(f"  families: {family_distribution(states)}")
        all_states[name] = states

    # build premise vocab — prefer the corpus (paper-spec global negatives)
    print("\nBuilding premise vocab...")
    if args.corpus_jsonl:
        print(f"  using global corpus at {args.corpus_jsonl}")
        premise_vocab = build_premise_vocab_from_corpus(
            args.corpus_jsonl, max_size=args.premise_vocab_size,
        )
    else:
        print("  using in-train premise frequencies")
        premise_vocab = build_premise_vocab(all_states["train"], max_size=4096)
    print(f"  premise vocab size: {len(premise_vocab)}")

    # synthesize Task C negatives (goal-swap heuristic) per split
    if args.value_negatives > 0:
        print(f"\nSynthesizing Task C negatives at fraction={args.value_negatives}...")
        for name in all_states:
            negs = synthesize_value_negatives(
                all_states[name], rng_seed=hash(name) & 0xFFFF,
                fraction=args.value_negatives,
            )
            print(f"  {name}: +{len(negs)} negative states")
            all_states[name] = all_states[name] + negs

    for name, states in all_states.items():
        print(f"[{name}] building RLIC parses...")
        t0 = time.perf_counter()
        ds = ProofStateDataset(states, cfg, TACTIC_FAMILIES, premise_vocab)
        dt = time.perf_counter() - t0
        print(f"  parsed {len(ds)} records in {dt:.1f}s")
        t0 = time.perf_counter()
        precompute_encodings(ds, struct=True, afford=True)
        print(f"  pre-encoded local tensors in {time.perf_counter()-t0:.1f}s")
        save_path = out / f"{name}.pkl"
        save_dataset(ds, save_path)
        print(f"  wrote {save_path} ({save_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
