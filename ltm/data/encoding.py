"""Convert RLIC cells into the tensor batches consumed by the §5 layer.

Per Theorem 6.3(i) and (iii): embeddings depend only on (ℓ, τ, g); all weights
are indexed by class-and-role signatures, never cell identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rlic import RLIC, Role, RoleTag


# (label, type_tag, grade) → id; managed per-batch (small enough that we don't
# need a global vocabulary)
@dataclass
class CellBatch:
    """A batch of RLICs flattened for vectorised computation."""

    label: torch.Tensor      # [N] long
    type_tag: torch.Tensor   # [N] long
    grade: torch.Tensor      # [N] long

    # Boundary edges (face → coface): from face cell to coface cell, with role
    # `down_src` is the face cid, `down_dst` is the coface cid (the cell whose
    # ∂ contains face).
    down_src: torch.Tensor   # [E_d] long — face cell global id
    down_dst: torch.Tensor   # [E_d] long — coface cell global id
    down_role: torch.Tensor  # [E_d] long — role.as_int()

    # Coboundary edges (coface → face): reverse direction with same role.
    up_src: torch.Tensor     # [E_u] coface
    up_dst: torch.Tensor     # [E_u] face
    up_role: torch.Tensor    # [E_u]

    # Same-grade adjacency: (σ', η, r, r') pairs from §5.2
    side_src: torch.Tensor   # [E_s] σ'
    side_dst: torch.Tensor   # [E_s] σ
    side_role_self: torch.Tensor   # [E_s] r
    side_role_other: torch.Tensor  # [E_s] r'

    # Batch index per cell (for pooling and sliced eval)
    batch_idx: torch.Tensor  # [N]
    n_graphs: int

    # Per-graph metadata (for sliced eval / Task A label / etc.)
    family_idx: torch.Tensor # [B] long, tactic-family target id
    slice_idx: torch.Tensor  # [B] long, structural-slice id

    def to(self, device) -> "CellBatch":
        return CellBatch(
            **{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in self.__dict__.items()}
        )


def encode_rlic_local(K: RLIC) -> dict:
    """Encode a single RLIC into *local* CPU tensors (indices not yet offset).

    This is the heavy work; do it once per record at dataset load time and
    cache the result on the record. The collate step then just shifts indices
    and concatenates — a fraction of the original CPU cost per batch.
    """
    n = len(K.cells)
    label = torch.tensor([c.label for c in K.cells], dtype=torch.long)
    type_tag = torch.tensor([c.type_tag for c in K.cells], dtype=torch.long)
    grade = torch.tensor([c.grade for c in K.cells], dtype=torch.long)

    down_src_l, down_dst_l, down_role_l = [], [], []
    for c in K.cells:
        for face_cid, role in c.boundary:
            down_src_l.append(face_cid)
            down_dst_l.append(c.cid)
            down_role_l.append(role.as_int())
    down_src = torch.tensor(down_src_l, dtype=torch.long)
    down_dst = torch.tensor(down_dst_l, dtype=torch.long)
    down_role = torch.tensor(down_role_l, dtype=torch.long)

    # Same-grade adjacency
    side_src_l, side_dst_l, side_role_self_l, side_role_other_l = [], [], [], []
    coface_by_face: dict[int, list[tuple[int, int]]] = {}
    for c in K.cells:
        for face_cid, role in c.boundary:
            coface_by_face.setdefault(face_cid, []).append((c.cid, role.as_int()))
    for face_cid, cofaces in coface_by_face.items():
        by_grade: dict[int, list[tuple[int, int]]] = {}
        for ccid, rid in cofaces:
            by_grade.setdefault(K.cells[ccid].grade, []).append((ccid, rid))
        for g, group in by_grade.items():
            for i, (a, ra) in enumerate(group):
                for j, (b, rb) in enumerate(group):
                    if a == b:
                        continue
                    side_src_l.append(b); side_dst_l.append(a)
                    side_role_self_l.append(ra); side_role_other_l.append(rb)
    side_src = torch.tensor(side_src_l, dtype=torch.long)
    side_dst = torch.tensor(side_dst_l, dtype=torch.long)
    side_role_self = torch.tensor(side_role_self_l, dtype=torch.long)
    side_role_other = torch.tensor(side_role_other_l, dtype=torch.long)

    return {
        "label": label, "type_tag": type_tag, "grade": grade,
        "down_src": down_src, "down_dst": down_dst, "down_role": down_role,
        "side_src": side_src, "side_dst": side_dst,
        "side_role_self": side_role_self, "side_role_other": side_role_other,
        "n_cells": n,
    }


def collate(
    items: list[tuple,]  # see ProofStateDataset.__getitem__: passes ParsedRecord-equivalent
) -> CellBatch:
    """Collate pre-encoded records into a CellBatch.

    Each item must be either:
      - (record_with_encoded_attr, family_id, slice_id) where the record has
        an ``encoded`` attribute carrying the local tensors from
        ``encode_rlic_local``;
      - or (RLIC, family_id, slice_id) — fallback that pays the local encode
        cost at collate time (legacy path).
    """
    labels, types, grades = [], [], []
    downs_src, downs_dst, downs_role = [], [], []
    sides_src, sides_dst, sides_role_self, sides_role_other = [], [], [], []
    batch_idx, family_idx, slice_idx = [], [], []

    offset = 0
    for b, (obj, fam, sl) in enumerate(items):
        enc = obj if isinstance(obj, dict) else encode_rlic_local(obj)
        n = enc["n_cells"]
        labels.append(enc["label"]); types.append(enc["type_tag"]); grades.append(enc["grade"])
        downs_src.append(enc["down_src"] + offset)
        downs_dst.append(enc["down_dst"] + offset)
        downs_role.append(enc["down_role"])
        sides_src.append(enc["side_src"] + offset)
        sides_dst.append(enc["side_dst"] + offset)
        sides_role_self.append(enc["side_role_self"])
        sides_role_other.append(enc["side_role_other"])
        batch_idx.append(torch.full((n,), b, dtype=torch.long))
        family_idx.append(fam); slice_idx.append(sl)
        offset += n

    def cat(parts):
        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(parts)

    down_src = cat(downs_src); down_dst = cat(downs_dst); down_role = cat(downs_role)
    # coboundary = reverse direction with same role
    up_src = down_dst.clone(); up_dst = down_src.clone(); up_role = down_role.clone()

    return CellBatch(
        label=cat(labels), type_tag=cat(types), grade=cat(grades),
        down_src=down_src, down_dst=down_dst, down_role=down_role,
        up_src=up_src, up_dst=up_dst, up_role=up_role,
        side_src=cat(sides_src), side_dst=cat(sides_dst),
        side_role_self=cat(sides_role_self), side_role_other=cat(sides_role_other),
        batch_idx=cat(batch_idx),
        family_idx=torch.tensor(family_idx, dtype=torch.long),
        slice_idx=torch.tensor(slice_idx, dtype=torch.long),
        n_graphs=len(items),
    )
