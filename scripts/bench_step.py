"""Per-step wall-clock benchmark across (M3, M5) at the §7.4 config.

Measures steady-state ms/step after a warmup. Reports throughput in
states/second and projects 80k-step wall-clock per model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltm.baselines.hypergraph import HyperConfig, HypergraphModel
from ltm.data.dataset import load_dataset, make_collate_fn, precompute_encodings
from ltm.models.heads import PremiseEncoder
from ltm.models.rlic_model import RLICConfig, RLICModel
from ltm.train import TrainConfig, forward_and_loss
from torch.utils.data import DataLoader


def bench(model, label, train_ds, batch_size, n_warm, n_meas, device, cfg):
    model.to(device)
    premise_enc = PremiseEncoder(cfg.n_premise_buckets, 128).to(device)
    import ltm.train as _t
    setattr(_t, "_RETRIEVAL_NEG_K", cfg.n_retrieval_negatives)
    if cfg.compile_model and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=make_collate_fn(use_struct=False),
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    params = list(model.parameters()) + list(premise_enc.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, fused=device.type == "cuda")
    model.train(); premise_enc.train()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)
    it = iter(loader)
    # warmup (triggers compile)
    for i in range(n_warm):
        try:
            batch, extras = next(it)
        except StopIteration:
            it = iter(loader); batch, extras = next(it)
        with autocast:
            loss, _ = forward_and_loss(model, premise_enc, batch, extras, device)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_meas):
        try:
            batch, extras = next(it)
        except StopIteration:
            it = iter(loader); batch, extras = next(it)
        with autocast:
            loss, _ = forward_and_loss(model, premise_enc, batch, extras, device)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = 1000 * dt / n_meas
    s_per_state = dt / (n_meas * batch_size)
    proj_80k_hr = (80000 * ms / 1000) / 3600
    print(f"  {label}: {ms:.1f} ms/step  ({1/s_per_state:.0f} states/s)  "
          f"→ 80k steps ≈ {proj_80k_hr:.2f} h")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/parses/mathlib_filtered")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--m3-d", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-warm", type=int, default=30)
    p.add_argument("--n-meas", type=int, default=80)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = load_dataset(f"{args.data_dir}/train.pkl")
    print(f"Loaded {len(train_ds)} records; pre-encoding...")
    t0 = time.perf_counter()
    precompute_encodings(train_ds, struct=False, afford=True)
    print(f"  pre-encoded in {time.perf_counter()-t0:.1f}s")

    n_families = len(train_ds.family_vocab)
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=3e-4,
        compile_model=not args.no_compile,
        num_workers=args.num_workers,
        precompute_encodings=False,  # already done above
    )

    print(f"\n=== Benching at batch={args.batch_size}, d={args.d}, L={args.L} "
          f"compile={cfg.compile_model} workers={cfg.num_workers} ===")

    print("\nM5 full RLIC:")
    m5 = RLICModel(RLICConfig(d=args.d, L=args.L, n_families=n_families))
    bench(m5, "M5", train_ds, args.batch_size, args.n_warm, args.n_meas, device, cfg)

    print("\nM3 strong hypergraph (param-matched):")
    m3 = HypergraphModel(HyperConfig(d=args.m3_d, L=args.L, n_families=n_families,
                                      use_roles=True, use_hyperedge_state=True))
    bench(m3, "M3", train_ds, args.batch_size, args.n_warm, args.n_meas, device, cfg)


if __name__ == "__main__":
    main()
