"""Add pre-encoded tensors to existing dataset pickles in-place.

Run once after build_dataset.py if it was built without precompute_encodings.
This is what we'd otherwise pay every training-run launch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.data.dataset import load_dataset, precompute_encodings, save_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_full")
    args = p.parse_args()
    d = Path(args.data_dir)
    for split in ("train", "val", "test"):
        path = d / f"{split}.pkl"
        if not path.exists():
            print(f"  {path} missing, skip")
            continue
        print(f"[{split}] loading {path}")
        t0 = time.perf_counter()
        ds = load_dataset(path)
        print(f"  loaded {len(ds.records)} records in {time.perf_counter()-t0:.1f}s")
        # skip if already encoded
        if ds.records and ds.records[0].enc_afford is not None and ds.records[0].enc_struct is not None:
            print(f"  already pre-encoded; skipping")
            continue
        t0 = time.perf_counter()
        precompute_encodings(ds, struct=True, afford=True)
        print(f"  encoded in {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        save_dataset(ds, path)
        print(f"  saved {path} ({path.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
