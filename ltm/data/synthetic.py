"""Synthetic Lean-like proof states for end-to-end pipeline smoke testing.

These are *not* real mathlib proofs. They're tiny, structurally realistic states
that exercise rewrite-opps, apply-opps, induction-opps and constructor-opps.
A LeanDojo adapter will later replace this generator with real extractions.
"""

from __future__ import annotations

import random

from ..proof_state import Expr, Hyp, ProofState
from ..rlic import TypeTag


def nat_type() -> Expr:
    return Expr(op="type", name="Nat", type_tag=int(TypeTag.NAT))


def prop_type() -> Expr:
    return Expr(op="type", name="Prop", type_tag=int(TypeTag.PROP))


def var(name: str, ty: int = int(TypeTag.NAT)) -> Expr:
    return Expr(op="var", name=name, type_tag=ty)


def const(name: str, ty: int = int(TypeTag.PROP)) -> Expr:
    return Expr(op="const", name=name, type_tag=ty)


def app(f: Expr, *args: Expr, ty: int = int(TypeTag.UNKNOWN)) -> Expr:
    return Expr(op="app", name="", children=(f,) + tuple(args), type_tag=ty)


def eq(lhs: Expr, rhs: Expr) -> Expr:
    return Expr(op="eq", name="", children=(lhs, rhs), type_tag=int(TypeTag.PROP))


# ---------------------------------------------------------------------------
# Generators per structural category (rewrite / app / constructor / induction)
# ---------------------------------------------------------------------------


def make_rewrite_state(idx: int) -> ProofState:
    """h : x = y ⊢ y = x — classical rewrite-heavy state."""
    x, y = var("x"), var("y")
    return ProofState(
        theorem=f"synth.rewrite.{idx}",
        state_idx=0,
        gamma=[
            Hyp(name="x", typ=nat_type()),
            Hyp(name="y", typ=nat_type()),
            Hyp(name="h", typ=eq(x, y), is_proof=True),
        ],
        goal=eq(y, x),
        namespace="synth.rewrite",
        next_tactic_family="rewrite",
        next_tactic_premises=("h",),
    )


def make_app_state(idx: int) -> ProofState:
    """h : f x y z = w ⊢ g x y = w — multi-argument application slice."""
    f = const("f"); g = const("g"); w = var("w")
    x, y, z = var("x"), var("y"), var("z")
    return ProofState(
        theorem=f"synth.app.{idx}",
        state_idx=0,
        gamma=[
            Hyp(name="x", typ=nat_type()),
            Hyp(name="y", typ=nat_type()),
            Hyp(name="z", typ=nat_type()),
            Hyp(name="w", typ=nat_type()),
            Hyp(name="h", typ=eq(app(f, x, y, z), w), is_proof=True),
        ],
        goal=eq(app(g, x, y), w),
        namespace="synth.app",
        next_tactic_family="apply",
    )


def make_constructor_state(idx: int) -> ProofState:
    """⊢ P ∧ Q — constructor-split slice."""
    P_c = const("P"); Q_c = const("Q")
    return ProofState(
        theorem=f"synth.cons.{idx}",
        state_idx=0,
        gamma=[],
        goal=app(const("And"), P_c, Q_c),
        namespace="synth.constructor",
        next_tactic_family="constructor",
    )


def make_induction_state(idx: int) -> ProofState:
    """n : Nat ⊢ P n — induction slice."""
    n = var("n")
    return ProofState(
        theorem=f"synth.ind.{idx}",
        state_idx=0,
        gamma=[Hyp(name="n", typ=nat_type())],
        goal=app(const("P"), n),
        namespace="synth.induction",
        next_tactic_family="induction",
    )


def make_residual_state(idx: int) -> ProofState:
    """h : P ⊢ P — pure apply, no rewrite, no constructor."""
    P_c = const("P")
    return ProofState(
        theorem=f"synth.res.{idx}",
        state_idx=0,
        gamma=[Hyp(name="h", typ=P_c, is_proof=True)],
        goal=P_c,
        namespace="synth.residual",
        next_tactic_family="exact",
        next_tactic_premises=("h",),
    )


def make_unsolvable_state(idx: int, base: ProofState) -> ProofState:
    """A state in which is_solved_by_some_tactic is False.

    Constructed by adding a confounding hypothesis but stripping the relevant
    premise, so the natural tactic family for this slice no longer applies.
    """
    new_gamma = [h for h in base.gamma if not h.is_proof]
    return ProofState(
        theorem=base.theorem.replace("synth.", "synth.unsolv."),
        state_idx=base.state_idx,
        gamma=new_gamma,
        goal=base.goal,
        namespace=base.namespace + ".unsolv",
        next_tactic_family=base.next_tactic_family,
        next_tactic_premises=(),
        is_solved_by_some_tactic=False,
    )


def make_dataset(n_per_slice: int = 200, seed: int = 0, unsolv_frac: float = 0.3) -> list[ProofState]:
    rng = random.Random(seed)
    out: list[ProofState] = []
    builders = [
        make_rewrite_state, make_app_state, make_constructor_state,
        make_induction_state, make_residual_state,
    ]
    for i in range(n_per_slice):
        for b in builders:
            P = b(i)
            if rng.random() < unsolv_frac:
                P = make_unsolvable_state(i, P)
            out.append(P)
    rng.shuffle(out)
    return out


# tactic-family vocabulary used by Task A targets in the synthetic set
TACTIC_FAMILIES = ("rewrite", "apply", "constructor", "induction", "exact")
