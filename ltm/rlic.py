"""Role-Labelled Incidence Complex (RLIC) data structures.

Implements Definition 3.1 of the paper:
    K = (K_0 ⊔ K_1 ⊔ ..., ∂, ℓ, τ)
- cell set finite, graded by g(σ) ∈ Z_{≥0}
- ℓ : cells → L (cell labels)
- τ : cells → T (type/foundation annotations)
- ∂(σ) ∈ Multi(K_{<g(σ)} × R), the role-labelled boundary multiset

We use integer ids for cells and integer ids for (role, label, type) ontology entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

# ---------------------------------------------------------------------------
# Ontologies (Definition 3.4 role set, plus cell-label and type vocabularies)
# ---------------------------------------------------------------------------


class Role(IntEnum):
    """Role set R (Definition 3.4)."""

    SOURCE = 0
    TARGET = 1
    FUNCTION = 2
    ARGUMENT = 3  # argument_k indexed below via (Role.ARGUMENT, k)
    GOAL = 4
    EQUALITY = 5
    OCCURRENCE = 6
    BINDER = 7
    BOUND = 8
    PREMISE = 9
    CONCLUSION = 10
    TERM = 11
    TYPE = 12
    PROOF = 13
    PROPOSITION = 14
    INDUCTIVE = 15
    BASE_CASE = 16
    STEP_CASE = 17
    WITNESS = 18


@dataclass(frozen=True)
class RoleTag:
    """A role with optional integer index (for argument_k)."""

    role: Role
    index: int = 0  # used only for ARGUMENT_k

    def as_int(self, max_arg: int = 16) -> int:
        """Canonical integer id for (role, index) for embedding tables."""
        base = int(self.role) * (max_arg + 1)
        return base + (self.index if self.role is Role.ARGUMENT else 0)


# Cell labels (ℓ). Open vocabulary; the parser maintains a string→id map.
# Frequent ones reserved here.
class CellLabel(IntEnum):
    # grade-0 atoms
    VARIABLE = 0
    LOCAL_HYP = 1
    GOAL = 2
    CONSTANT = 3
    SUBTERM = 4
    METAVAR = 5
    LEMMA_NAME = 6
    TYPE_EXPR = 7
    UNIVERSE = 8
    # grade-1 typed binary incidences
    HAS_TYPE = 100
    OCCURS_IN = 101
    PROVES = 102
    BINDS = 103
    # structural higher-grade (grade ≥ 2)
    APP_N = 200      # f a_1 ... a_n  (n encoded in cell attrs)
    EQ_LIT = 201     # equality literal x = y (structural)
    # affordance higher-grade (grade ≥ 2), emitted by P-independent generators
    REWRITE_OPP = 300
    APPLY_OPP = 301
    INDUCTION_OPP = 302
    CONSTRUCTOR_OPP = 303


# Type/foundation annotations (τ). Coarse for the prototype.
class TypeTag(IntEnum):
    UNKNOWN = 0
    PROP = 1
    TYPE_T = 2
    NAT = 3
    INT = 4
    LIST = 5
    FUNC = 6
    INDUCTIVE = 7


# ---------------------------------------------------------------------------
# The RLIC itself
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One cell σ of the RLIC."""

    cid: int                  # canonical id within K
    grade: int                # g(σ) ∈ Z_{≥0}
    label: int                # ℓ(σ) — id in label vocabulary (CellLabel or extension)
    type_tag: int             # τ(σ) — id in type vocabulary
    name: str | None = None   # raw label/text if applicable (subject to §5.5 policy)
    boundary: list[tuple[int, RoleTag]] = field(default_factory=list)
    # boundary entries (face_cid, role) — for grade-0 cells this is empty.
    # multiset semantics: repeated entries allowed.

    @property
    def class_sig(self) -> tuple[int, int, int]:
        """(ℓ(σ), τ(σ), g(σ)) — the "class" of the cell (§3.4)."""
        return (self.label, self.type_tag, self.grade)


@dataclass
class RLIC:
    """A role-labelled incidence complex over a fixed ontology."""

    cells: list[Cell] = field(default_factory=list)
    # provenance
    source: str = ""           # e.g. theorem name + state index

    # ---- builders ------------------------------------------------------

    def add_cell(
        self,
        grade: int,
        label: int,
        type_tag: int = int(TypeTag.UNKNOWN),
        name: str | None = None,
        boundary: Iterable[tuple[int, RoleTag]] = (),
    ) -> int:
        cid = len(self.cells)
        self.cells.append(
            Cell(
                cid=cid,
                grade=grade,
                label=label,
                type_tag=type_tag,
                name=name,
                boundary=list(boundary),
            )
        )
        return cid

    # ---- views ---------------------------------------------------------

    def by_grade(self, g: int) -> list[Cell]:
        return [c for c in self.cells if c.grade == g]

    @property
    def K0(self) -> list[Cell]:
        return self.by_grade(0)

    @property
    def K1(self) -> list[Cell]:
        return self.by_grade(1)

    def K_ge(self, g: int) -> list[Cell]:
        return [c for c in self.cells if c.grade >= g]

    def coboundary(self, cid: int) -> list[tuple[int, RoleTag]]:
        """For σ with id `cid`, return [(ω, r) : (σ,r) ∈ ∂ω]."""
        out: list[tuple[int, RoleTag]] = []
        for c in self.cells:
            for face_cid, role in c.boundary:
                if face_cid == cid:
                    out.append((c.cid, role))
        return out

    # ---- sub-complex extraction (§3.4 four-complex hierarchy) ---------

    def restrict_to_struct(self) -> "RLIC":
        """Return K_struct(P): keep grade-0, grade-1, and *structural* higher-grade cells.

        Structural higher-grade labels are those that encode multi-way state-side
        structure (e.g. APP_N, EQ_LIT). Affordance labels (REWRITE_OPP, APPLY_OPP, ...)
        are dropped.
        """
        keep = {int(CellLabel.APP_N), int(CellLabel.EQ_LIT)}
        out = RLIC(source=self.source + ":struct")
        old_to_new: dict[int, int] = {}
        for c in self.cells:
            if c.grade <= 1 or c.label in keep:
                new = out.add_cell(c.grade, c.label, c.type_tag, c.name)
                old_to_new[c.cid] = new
        # remap boundaries, dropping any that point to discarded cells
        for c in self.cells:
            if c.cid not in old_to_new:
                continue
            nc = out.cells[old_to_new[c.cid]]
            for face_cid, role in c.boundary:
                if face_cid in old_to_new:
                    nc.boundary.append((old_to_new[face_cid], role))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


STRUCTURAL_LABELS = frozenset({int(CellLabel.APP_N), int(CellLabel.EQ_LIT)})
AFFORDANCE_LABELS = frozenset(
    {
        int(CellLabel.REWRITE_OPP),
        int(CellLabel.APPLY_OPP),
        int(CellLabel.INDUCTION_OPP),
        int(CellLabel.CONSTRUCTOR_OPP),
    }
)


def is_structural(cell: Cell) -> bool:
    return cell.grade <= 1 or cell.label in STRUCTURAL_LABELS


def is_affordance(cell: Cell) -> bool:
    return cell.grade >= 2 and cell.label in AFFORDANCE_LABELS
