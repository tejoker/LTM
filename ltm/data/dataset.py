"""Dataset abstraction: ProofState → parsed (K_struct, K_afford) → tensor batch.

Caches parses on disk as a single Arrow file to amortise the parser cost
(which dominates wall-clock for any real LeanDojo run).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..data.encoding import collate, encode_rlic_local
from ..parser import build_afford, build_struct, canonical_truncate
from ..proof_state import ParserConfig, ProofState
from ..rlic import RLIC


# Structural slice ids (matches §7.5 sliced eval)
SLICE_NAMES = ("rewrite_heavy", "multi_arg_app", "constructor_split", "induction", "residual")


def slice_of(K: RLIC) -> int:
    """Categorise an RLIC into one of the five §7.5 slices.

    Priority order matches the paper's wording: if more than one applies, pick
    the most specific.
    """
    from ..rlic import CellLabel
    labels = {c.label for c in K.cells if c.grade >= 2}
    # multi-arg application?
    max_n = 0
    for c in K.cells:
        if c.label == int(CellLabel.APP_N):
            args = sum(1 for _, r in c.boundary if r.role.name == "ARGUMENT")
            max_n = max(max_n, args)
    if int(CellLabel.REWRITE_OPP) in labels:
        return 0
    if max_n >= 3:
        return 1
    if int(CellLabel.CONSTRUCTOR_OPP) in labels:
        return 2
    if int(CellLabel.INDUCTION_OPP) in labels:
        return 3
    return 4


@dataclass
class ParsedRecord:
    K_afford: RLIC
    K_struct: RLIC
    family_id: int
    slice_id: int
    value_target: int  # 0 or 1
    premise_id: int    # for retrieval; 0 if not applicable
    P_ref: ProofState  # kept for text baseline & probes
    # Pre-encoded local CPU tensors, set lazily by precompute_encodings() for
    # fast collate. Two variants: afford (default) and struct (M5a path).
    enc_afford: dict | None = None
    enc_struct: dict | None = None


class ProofStateDataset(Dataset):
    """In-memory parsed dataset. For the synthetic / small mathlib runs this
    fits in RAM trivially; bigger runs should swap in a memory-mapped Arrow
    backend."""

    def __init__(
        self,
        states: list[ProofState],
        cfg: ParserConfig,
        family_vocab: tuple[str, ...],
        premise_vocab: tuple[str, ...],
        truncate: bool = True,
    ):
        self.cfg = cfg
        self.family_vocab = family_vocab
        self.premise_vocab = premise_vocab
        self.records: list[ParsedRecord] = []
        fam_to_id = {f: i for i, f in enumerate(family_vocab)}
        prem_to_id = {p: i for i, p in enumerate(premise_vocab)}
        for P in states:
            K_a = build_afford(P, cfg)
            K_s = build_struct(P, cfg)
            if truncate:
                K_a = canonical_truncate(K_a, cfg)
                K_s = canonical_truncate(K_s, cfg)
            fam = fam_to_id.get(P.next_tactic_family, 0)
            sl = slice_of(K_a)
            # premise: first one, hashed into vocab bucket
            prem_id = 0
            if P.next_tactic_premises:
                prem_id = prem_to_id.get(P.next_tactic_premises[0], 0)
            self.records.append(
                ParsedRecord(
                    K_afford=K_a,
                    K_struct=K_s,
                    family_id=fam,
                    slice_id=sl,
                    value_target=int(P.is_solved_by_some_tactic),
                    premise_id=prem_id,
                    P_ref=P,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ParsedRecord:
        return self.records[idx]


def make_collate_fn(use_struct: bool = False):
    """Returns a collate fn that produces a CellBatch from a list of records.

    Uses pre-encoded local tensors (``record.enc_afford`` / ``enc_struct``) if
    present, otherwise falls back to encoding the RLIC at collate time.
    """
    def _collate(records: list[ParsedRecord]):
        items = []
        for r in records:
            if use_struct:
                obj = r.enc_struct if r.enc_struct is not None else r.K_struct
            else:
                obj = r.enc_afford if r.enc_afford is not None else r.K_afford
            items.append((obj, r.family_id, r.slice_id))
        batch = collate(items)
        value_target = torch.tensor([r.value_target for r in records], dtype=torch.float32)
        premise_id = torch.tensor([r.premise_id for r in records], dtype=torch.long)
        return batch, {
            "value_target": value_target,
            "premise_id": premise_id,
            "records": records,
        }
    return _collate


def precompute_encodings(ds: "ProofStateDataset", *, struct: bool = True,
                          afford: bool = True) -> None:
    """Populate ``enc_afford`` / ``enc_struct`` on every record. One-time cost.

    For a real LTM-Tiny run this is the single biggest CPU-side speedup: it
    converts per-batch ``encode_rlic`` work into per-batch concatenation.
    """
    for r in ds.records:
        if afford and r.enc_afford is None:
            r.enc_afford = encode_rlic_local(r.K_afford)
        if struct and r.enc_struct is None:
            r.enc_struct = encode_rlic_local(r.K_struct)


def save_dataset(ds: ProofStateDataset, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "records": ds.records,
                "family_vocab": ds.family_vocab,
                "premise_vocab": ds.premise_vocab,
                "cfg": ds.cfg,
            },
            f,
        )


def load_dataset(path: str | Path) -> ProofStateDataset:
    with open(path, "rb") as f:
        data = pickle.load(f)
    ds = ProofStateDataset.__new__(ProofStateDataset)
    ds.records = data["records"]
    ds.family_vocab = data["family_vocab"]
    ds.premise_vocab = data["premise_vocab"]
    ds.cfg = data["cfg"]
    return ds
