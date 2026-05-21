"""Training & evaluation driver, uniform across all M0–M5 models.

Multitask loss with equal weights on Tasks A/B/C (paper §7.2):
    L = CE_policy + InfoNCE_retrieval + BCE_value
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .data.dataset import ProofStateDataset, SLICE_NAMES, make_collate_fn, precompute_encodings
from .data.encoding import CellBatch
from .eval.metrics import task_a_metrics, task_b_metrics, task_c_metrics
from .models.heads import PremiseEncoder


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    total_steps: int = 5000
    batch_size: int = 16
    grad_accum: int = 2
    log_every: int = 50
    eval_every: int = 500
    bf16: bool = True
    grad_clip: float = 1.0
    n_premise_buckets: int = 4096
    # Task B: number of random global negatives sampled per step for InfoNCE.
    # 0 = in-batch negatives only (legacy behaviour); paper §7.2 uses global.
    n_retrieval_negatives: int = 512
    # torch.compile: scatter-heavy graphs (especially M5 with 3 attention
    # directions) actually run slower under compile because of cudagraph
    # recompilation on each new edge-count. Eager wins at batch=32 on this
    # model. Kept off by default; turn on for fixed-shape baselines.
    compile_model: bool = False
    # DataLoader workers; >0 needs pickleable records (no RLIC reference if pre-encoded)
    num_workers: int = 0
    # Pre-encode RLICs into per-record CPU tensors before training
    precompute_encodings: bool = True


def cosine_schedule(step: int, warmup: int, total: int, base_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    import math
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def _move_to(batch: CellBatch, device) -> CellBatch:
    return batch.to(device)


def forward_and_loss(
    model: nn.Module,
    premise_enc: PremiseEncoder,
    batch: CellBatch,
    extras: dict,
    device,
    is_text: bool = False,
    text_ids: Tensor | None = None,
    text_mask: Tensor | None = None,
    is_symbolic: bool = False,
    symbolic_x: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if is_text:
        out = model(text_ids.to(device), text_mask.to(device))
        family_idx = batch.family_idx.to(device)
        value_target = extras["value_target"].to(device)
        premise_id = extras["premise_id"].to(device)
        batch_idx_dev = None
    elif is_symbolic:
        out = model(symbolic_x.to(device))
        family_idx = batch.family_idx.to(device)
        value_target = extras["value_target"].to(device)
        premise_id = extras["premise_id"].to(device)
    else:
        batch = _move_to(batch, device)
        out = model(batch)
        family_idx = batch.family_idx
        value_target = extras["value_target"].to(device)
        premise_id = extras["premise_id"].to(device)

    # Task A: cross-entropy
    loss_a = F.cross_entropy(out["policy_logits"], family_idx)
    # Task C: BCE-with-logits (autocast-safe). out["value"] is a logit.
    loss_c = F.binary_cross_entropy_with_logits(out["value"], value_target)

    # Task B: InfoNCE against ψ(premise). Positives = the recorded premise of
    # this state. Negatives = the other states' premises (in-batch) plus
    # ``n_retrieval_negatives`` random premise ids sampled globally from the
    # whole premise vocabulary, per paper §7.2.
    n_neg = globals().get("_RETRIEVAL_NEG_K", 0)
    psi_pos = premise_enc(premise_id)         # [B, embed_dim]
    phi = out["retrieval_query"]              # [B, embed_dim]
    if n_neg > 0:
        n_vocab = premise_enc.emb.num_embeddings
        neg_ids = torch.randint(1, n_vocab, (n_neg,), device=device)
        psi_neg = premise_enc(neg_ids)         # [n_neg, embed_dim]
        psi_all = torch.cat([psi_pos, psi_neg], dim=0)
    else:
        psi_all = psi_pos
    logits = phi @ psi_all.t() * 10.0
    targets = torch.arange(phi.shape[0], device=device)
    loss_b = F.cross_entropy(logits, targets)

    loss = loss_a + loss_b + loss_c
    return loss, {
        "policy_logits": out["policy_logits"].detach(),
        # convert logit to probability for downstream metrics
        "value": torch.sigmoid(out["value"]).detach(),
        "retrieval_query": phi.detach(),
        "premise_emb": psi_pos.detach(),
        "family_idx": family_idx.detach(),
        "value_target": value_target.detach(),
        "loss_a": loss_a.detach(), "loss_b": loss_b.detach(), "loss_c": loss_c.detach(),
    }


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)
    sliced_metrics: dict = field(default_factory=dict)
    param_count: int = 0
    inference_ms_per_state: float = 0.0


def evaluate(
    model: nn.Module,
    premise_enc: PremiseEncoder,
    loader: DataLoader,
    device,
    cfg: TrainConfig,
    text_collate: Callable | None = None,
    symbolic_collate: Callable | None = None,
) -> tuple[dict, dict]:
    """Returns (aggregate metrics, sliced metrics per slice id)."""
    model.eval()
    premise_enc.eval()
    all_logits, all_targets = [], []
    all_phi, all_premise = [], []
    all_value_pred, all_value_target = [], []
    all_slice = []
    n = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch, extras in loader:
            if text_collate is not None:
                ids, mask = text_collate([r.P_ref for r in extras["records"]])
                _, info = forward_and_loss(
                    model, premise_enc, batch, extras, device,
                    is_text=True, text_ids=ids, text_mask=mask,
                )
                slice_idx = batch.slice_idx
            elif symbolic_collate is not None:
                x = symbolic_collate(extras["records"])
                _, info = forward_and_loss(
                    model, premise_enc, batch, extras, device,
                    is_symbolic=True, symbolic_x=x,
                )
                slice_idx = batch.slice_idx
            else:
                _, info = forward_and_loss(model, premise_enc, batch, extras, device)
                slice_idx = batch.slice_idx
            all_logits.append(info["policy_logits"].cpu())
            all_targets.append(info["family_idx"].cpu())
            all_phi.append(info["retrieval_query"].cpu())
            all_premise.append(info["premise_emb"].cpu())
            all_value_pred.append(info["value"].cpu())
            all_value_target.append(info["value_target"].cpu())
            all_slice.append(slice_idx)
            n += info["family_idx"].shape[0]
    elapsed = time.perf_counter() - t0
    ms_per = 1000.0 * elapsed / max(1, n)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    phi = torch.cat(all_phi); psi = torch.cat(all_premise)
    scores = phi @ psi.t()
    target_idx = torch.arange(scores.shape[0])
    value_pred = torch.cat(all_value_pred); value_t = torch.cat(all_value_target)
    slices = torch.cat(all_slice)

    agg = {
        **task_a_metrics(logits, targets),
        **{f"B_{k}": v for k, v in task_b_metrics(scores, target_idx).items()},
        **{f"C_{k}": v for k, v in task_c_metrics(value_pred, value_t).items()},
        "ms_per_state": ms_per,
    }
    sliced = {}
    for s in range(len(SLICE_NAMES)):
        m = slices == s
        if m.sum() < 5:
            continue
        a = task_a_metrics(logits[m], targets[m])
        # B retrieval slice: only over rows in this slice
        sub_scores = phi[m] @ psi.t()
        sub_target = torch.arange(scores.shape[0])[m]
        b = task_b_metrics(sub_scores, sub_target)
        c = task_c_metrics(value_pred[m], value_t[m])
        sliced[SLICE_NAMES[s]] = {
            "n": int(m.sum().item()), **a,
            **{f"B_{k}": v for k, v in b.items()},
            **{f"C_{k}": v for k, v in c.items()},
        }
    model.train(); premise_enc.train()
    return agg, sliced


def train_model(
    model: nn.Module,
    train_ds: ProofStateDataset,
    val_ds: ProofStateDataset,
    cfg: TrainConfig,
    device: str | torch.device,
    use_struct: bool = False,
    text_collate: Callable | None = None,
    symbolic_collate: Callable | None = None,
    label: str = "model",
    log_path: Path | None = None,
) -> TrainResult:
    device = torch.device(device)
    model.to(device)
    premise_enc = PremiseEncoder(cfg.n_premise_buckets, 128).to(device)
    # publish the global-negatives K for forward_and_loss to pick up
    globals()["_RETRIEVAL_NEG_K"] = cfg.n_retrieval_negatives

    # Pre-encode RLICs once for fast collate (massive CPU-side speedup)
    if cfg.precompute_encodings:
        need_struct = use_struct
        precompute_encodings(train_ds, struct=need_struct, afford=not need_struct)
        precompute_encodings(val_ds, struct=need_struct, afford=not need_struct)

    # torch.compile: on this scatter-heavy graph, cudagraphs in
    # "reduce-overhead" mode actually serialise per-dynamic-shape, slowing
    # things down. Default mode helps marginally on stable shapes but the
    # overhead of recompilation per varying graph size kills the win. Empirics
    # at batch=32 say eager is fastest; we keep the toggle for the text/
    # symbolic baselines and for future fixed-shape paths.
    if cfg.compile_model and device.type == "cuda" and text_collate is None and symbolic_collate is None:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
        except Exception as e:
            print(f"  torch.compile disabled: {e}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=make_collate_fn(use_struct=use_struct),
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=make_collate_fn(use_struct=use_struct),
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    params = list(model.parameters()) + list(premise_enc.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    # Fused AdamW: ~5-10% faster on Ada/Ampere
    opt_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    opt = torch.optim.AdamW(params, **opt_kwargs)

    history: list[dict] = []
    step = 0
    model.train(); premise_enc.train()
    use_bf16 = cfg.bf16 and device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    grad_accum = cfg.grad_accum
    opt.zero_grad(set_to_none=True)
    t_start = time.perf_counter()
    done = False
    while not done:
        for batch, extras in train_loader:
            lr = cosine_schedule(step, cfg.warmup_steps, cfg.total_steps, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            with autocast_ctx:
                if text_collate is not None:
                    ids, mask = text_collate([r.P_ref for r in extras["records"]])
                    loss, info = forward_and_loss(
                        model, premise_enc, batch, extras, device,
                        is_text=True, text_ids=ids, text_mask=mask,
                    )
                elif symbolic_collate is not None:
                    x = symbolic_collate(extras["records"])
                    loss, info = forward_and_loss(
                        model, premise_enc, batch, extras, device,
                        is_symbolic=True, symbolic_x=x,
                    )
                else:
                    loss, info = forward_and_loss(
                        model, premise_enc, batch, extras, device,
                    )
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True)
            if step % cfg.log_every == 0:
                history.append({
                    "step": step, "lr": lr,
                    "loss": loss.item(),
                    "loss_a": info["loss_a"].item(),
                    "loss_b": info["loss_b"].item(),
                    "loss_c": info["loss_c"].item(),
                })
            step += 1
            if step >= cfg.total_steps:
                done = True
                break
    elapsed = time.perf_counter() - t_start

    agg, sliced = evaluate(
        model, premise_enc, val_loader, device, cfg,
        text_collate=text_collate, symbolic_collate=symbolic_collate,
    )
    res = TrainResult(
        history=history,
        final_metrics=agg,
        sliced_metrics=sliced,
        param_count=n_params,
        inference_ms_per_state=agg["ms_per_state"],
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(
                {
                    "label": label,
                    "param_count": n_params,
                    "elapsed_s": elapsed,
                    "final": agg,
                    "sliced": sliced,
                    "history": history,
                },
                f, indent=2,
            )
    return res
