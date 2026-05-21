"""Parser P ↦ K_afford(P) implementing Algorithms 1–3 from §3.6.

Compositional no-leakage (Lemma 6.2): each generator G_i is a deterministic
function of (P, Π) using only queries allowed by Condition 3.6(ii). We respect
the computational boundary:
  (a) syntactic inspection of parse tree, names, binders, hypotheses, goal;
  (b) Lean kernel typing queries at fixed transparency level Θ ∈ Π;
  (c) syntactic-only unification (first-order matching) for affordance patterns;
  (d) typeclass instance lookup if explicitly enabled in Π, at bounded depth.

This module implements (a) and (c) on the synthetic Expr schema. (b) and (d)
hooks are stubbed and exposed for the LeanDojo adapter.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .proof_state import Expr, Hyp, ParserConfig, ProofState
from .rlic import (
    AFFORDANCE_LABELS,
    Cell,
    CellLabel,
    RLIC,
    Role,
    RoleTag,
    TypeTag,
)


# ---------------------------------------------------------------------------
# Algorithm 1: BuildStruct(P, Π)
# ---------------------------------------------------------------------------


class StructBuilder:
    """Emit K_struct(P): grade-0 atoms, grade-1 typed binary incidences, and
    structural higher-grade cells (APP_N, EQ_LIT). No affordance cells.
    """

    def __init__(self, cfg: ParserConfig):
        self.cfg = cfg
        self.K = RLIC()
        # cache for atom de-duplication: keyed by (label_id, canonical_name)
        self._atom_by_key: dict[tuple[int, str], int] = {}
        # subterm dedup by (op, name, child_cids tuple)
        self._subterm_by_key: dict[tuple, int] = {}
        # type-expression dedup
        self._type_by_repr: dict[str, int] = {}

    # ---- atom emitters ------------------------------------------------

    def emit_var(self, raw_name: str) -> int:
        key_name = self.cfg.canonical_var_name(raw_name)
        return self._dedup_atom(int(CellLabel.VARIABLE), key_name, raw_name)

    def emit_hyp(self, h: Hyp) -> int:
        key_name = self.cfg.canonical_hyp_name(h.name)
        cid = self._dedup_atom(int(CellLabel.LOCAL_HYP), key_name, h.name)
        # also emit type expression and link with "has-type" grade-1 cell
        tcid = self.emit_type_expr(h.typ)
        self.emit_has_type(cid, tcid)
        if h.is_proof:
            # h proves its type; record proves(h, type)
            self.emit_proves(cid, tcid)
        return cid

    def emit_goal(self, g: Expr) -> int:
        gcid = self._dedup_atom(int(CellLabel.GOAL), "<goal>", "<goal>")
        # treat goal as carrying a proposition: link via has-type to its prop expr
        tcid = self.emit_type_expr(g)
        self.emit_has_type(gcid, tcid)
        # also recurse into the goal to emit subterm structure
        self.walk_expr(g)
        return gcid

    def emit_constant(self, name: str) -> int:
        kn = name if self.cfg.preserve_constant_names else "<const>"
        return self._dedup_atom(int(CellLabel.CONSTANT), kn, name)

    def emit_subterm(self, e: Expr) -> int:
        """Emit a grade-0 subterm cell (deduped). Recurses to children first."""
        child_ids = tuple(self.emit_subterm(c) for c in e.children)
        key = (e.op, e.name, child_ids)
        if key in self._subterm_by_key:
            return self._subterm_by_key[key]
        cid = self.K.add_cell(
            grade=0,
            label=int(CellLabel.SUBTERM),
            type_tag=e.type_tag,
            name=e.name,
        )
        self._subterm_by_key[key] = cid
        # add structural higher-grade if applicable
        if e.op == "app":
            self.emit_app(cid, child_ids)
        elif e.op == "eq":
            self.emit_eq_lit(cid, child_ids)
        elif e.op == "binder":
            # binder beta with body b — record "binds" grade-1
            # children = [bound_type, body]
            if len(child_ids) == 2:
                self.emit_binds(cid, child_ids[1])
        return cid

    def emit_type_expr(self, t: Expr) -> int:
        """Type expressions go through subterm dedup but use TYPE_EXPR label."""
        rep = _repr_expr(t)
        if rep in self._type_by_repr:
            return self._type_by_repr[rep]
        cid = self.K.add_cell(
            grade=0,
            label=int(CellLabel.TYPE_EXPR),
            type_tag=t.type_tag,
            name=rep,
        )
        self._type_by_repr[rep] = cid
        return cid

    # ---- grade-1 emitters --------------------------------------------

    def emit_has_type(self, term_cid: int, type_cid: int) -> int:
        return self.K.add_cell(
            grade=1,
            label=int(CellLabel.HAS_TYPE),
            type_tag=int(TypeTag.UNKNOWN),
            boundary=[
                (term_cid, RoleTag(Role.TERM)),
                (type_cid, RoleTag(Role.TYPE)),
            ],
        )

    def emit_occurs_in(self, var_cid: int, expr_cid: int) -> int:
        return self.K.add_cell(
            grade=1,
            label=int(CellLabel.OCCURS_IN),
            boundary=[
                (var_cid, RoleTag(Role.OCCURRENCE)),
                (expr_cid, RoleTag(Role.PROPOSITION)),
            ],
        )

    def emit_proves(self, hyp_cid: int, prop_cid: int) -> int:
        return self.K.add_cell(
            grade=1,
            label=int(CellLabel.PROVES),
            boundary=[
                (hyp_cid, RoleTag(Role.PROOF)),
                (prop_cid, RoleTag(Role.PROPOSITION)),
            ],
        )

    def emit_binds(self, binder_cid: int, body_cid: int) -> int:
        return self.K.add_cell(
            grade=1,
            label=int(CellLabel.BINDS),
            boundary=[
                (binder_cid, RoleTag(Role.BINDER)),
                (body_cid, RoleTag(Role.BOUND)),
            ],
        )

    # ---- structural higher-grade emitters -----------------------------

    def emit_app(self, result_cid: int, child_ids: tuple[int, ...]) -> int:
        """Grade-2 application app-(n) cell."""
        if not child_ids:
            return -1
        f_cid, *arg_cids = child_ids
        boundary = [(f_cid, RoleTag(Role.FUNCTION))]
        for k, a in enumerate(arg_cids, start=1):
            boundary.append((a, RoleTag(Role.ARGUMENT, index=k)))
        boundary.append((result_cid, RoleTag(Role.CONCLUSION)))
        return self.K.add_cell(
            grade=2,
            label=int(CellLabel.APP_N),
            type_tag=int(TypeTag.UNKNOWN),
            name=f"app-{len(arg_cids)}",
            boundary=boundary,
        )

    def emit_eq_lit(self, eq_cid: int, child_ids: tuple[int, ...]) -> int:
        """Grade-2 eq-lit cell: x = y with (lhs, source), (rhs, target)."""
        if len(child_ids) != 2:
            return -1
        lhs, rhs = child_ids
        return self.K.add_cell(
            grade=2,
            label=int(CellLabel.EQ_LIT),
            boundary=[
                (lhs, RoleTag(Role.SOURCE)),
                (rhs, RoleTag(Role.TARGET)),
                (eq_cid, RoleTag(Role.FUNCTION)),
            ],
        )

    # ---- driver -------------------------------------------------------

    def build(self, P: ProofState) -> RLIC:
        # grade-0 cells for hypotheses (which emit types and grade-1 has-type)
        for h in P.gamma:
            self.emit_hyp(h)
        # goal cell + grade-1 has-type from goal to its prop expr
        if P.goal is not None:
            self.emit_goal(P.goal)
        # variables occurring in the goal: emit occurs-in grade-1 cells
        if P.goal is not None:
            goal_atom = self._atom_by_key.get(
                (int(CellLabel.GOAL), "<goal>"), None
            )
            for sub in P.goal.walk():
                if sub.op == "var":
                    vc = self.emit_var(sub.name)
                    if goal_atom is not None:
                        self.emit_occurs_in(vc, goal_atom)
        # constants referenced in goal & hypotheses
        for e in self._iter_exprs(P):
            for sub in e.walk():
                if sub.op == "const":
                    self.emit_constant(sub.name)
        # subterms up to depth bound
        for e in self._iter_exprs(P):
            self._walk_bounded(e, self.cfg.depth_bound)
        # done
        self.K.source = f"{P.theorem}#{P.state_idx}:struct"
        return self.K

    # ---- helpers ------------------------------------------------------

    def walk_expr(self, e: Expr) -> None:
        self.emit_subterm(e)

    def _walk_bounded(self, e: Expr, depth: int) -> None:
        if depth <= 0:
            return
        self.emit_subterm(e)
        for c in e.children:
            self._walk_bounded(c, depth - 1)

    def _iter_exprs(self, P: ProofState) -> Iterable[Expr]:
        for h in P.gamma:
            yield h.typ
        if P.goal is not None:
            yield P.goal

    def _dedup_atom(self, label: int, key_name: str, raw_name: str) -> int:
        key = (label, key_name)
        if key in self._atom_by_key:
            return self._atom_by_key[key]
        cid = self.K.add_cell(
            grade=0,
            label=label,
            type_tag=int(TypeTag.UNKNOWN),
            name=raw_name if self._should_preserve(label) else key_name,
        )
        self._atom_by_key[key] = cid
        return cid

    def _should_preserve(self, label: int) -> bool:
        if label == int(CellLabel.CONSTANT):
            return self.cfg.preserve_constant_names
        if label == int(CellLabel.LOCAL_HYP):
            return self.cfg.preserve_hyp_names
        if label == int(CellLabel.VARIABLE):
            return self.cfg.preserve_bound_var_names
        return True


def _repr_expr(e: Expr) -> str:
    """Stable string repr of an expression, for type-expr dedup."""
    if e.op in ("var", "const", "type"):
        return f"{e.op}:{e.name}"
    return f"{e.op}({','.join(_repr_expr(c) for c in e.children)})"


def build_struct(P: ProofState, cfg: ParserConfig) -> RLIC:
    return StructBuilder(cfg).build(P)


# ---------------------------------------------------------------------------
# Algorithm 2: BuildAfford(P, Π)
# ---------------------------------------------------------------------------


def build_afford(P: ProofState, cfg: ParserConfig) -> RLIC:
    """Take K_struct and add P-independent affordance generators' output.

    Each generator G_i is a deterministic function of (P, Π) using only
    queries allowed by Condition 3.6(ii). Pure-syntactic (first-order matching)
    here; kernel-typing queries are future hooks.
    """
    K = build_struct(P, cfg)
    # Collect equality hypotheses and goal-side occurrence map for rewrite-opp
    if cfg.enable_rewrite_opp:
        _emit_rewrite_opps(P, K, cfg)
    if cfg.enable_constructor_opp:
        _emit_constructor_opp(P, K)
    if cfg.enable_induction_opp:
        _emit_induction_opps(P, K)
    if cfg.enable_apply_opp:
        _emit_apply_opps(P, K, cfg)
    K.source = f"{P.theorem}#{P.state_idx}:afford"
    return K


def _equality_hyps(P: ProofState) -> list[tuple[Hyp, Expr, Expr]]:
    out = []
    for h in P.gamma:
        if h.typ.op == "eq" and len(h.typ.children) == 2:
            out.append((h, h.typ.children[0], h.typ.children[1]))
    return out


def _occurrences(expr: Expr, target: Expr) -> int:
    """Count syntactic occurrences of target in expr (first-order matching)."""
    if _repr_expr(expr) == _repr_expr(target):
        return 1
    return sum(_occurrences(c, target) for c in expr.children)


def _find_cid(K: RLIC, label: int, key: str) -> int | None:
    for c in K.cells:
        if c.label == label and c.name == key:
            return c.cid
    return None


def _emit_rewrite_opps(P: ProofState, K: RLIC, cfg: ParserConfig) -> None:
    if P.goal is None:
        return
    goal_cid = _find_cid(K, int(CellLabel.GOAL), "<goal>")
    if goal_cid is None:
        return
    for h, lhs, rhs in _equality_hyps(P):
        n_lhs = _occurrences(P.goal, lhs)
        n_rhs = _occurrences(P.goal, rhs)
        if n_lhs == 0 and n_rhs == 0:
            continue
        # locate the equality hypothesis cell
        hyp_key = cfg.canonical_hyp_name(h.name)
        hyp_cid = _find_cid(K, int(CellLabel.LOCAL_HYP), hyp_key)
        if hyp_cid is None:
            continue
        # one occurrence cell per side that appears
        for _ in range(n_lhs + n_rhs):
            occ_cid = K.add_cell(
                grade=0,
                label=int(CellLabel.SUBTERM),
                name="<occ>",
            )
            K.add_cell(
                grade=2,
                label=int(CellLabel.REWRITE_OPP),
                type_tag=int(TypeTag.PROP),
                boundary=[
                    (hyp_cid, RoleTag(Role.EQUALITY)),
                    (occ_cid, RoleTag(Role.OCCURRENCE)),
                    (goal_cid, RoleTag(Role.GOAL)),
                ],
            )


def _emit_constructor_opp(P: ProofState, K: RLIC) -> None:
    if P.goal is None:
        return
    head = P.goal
    # the head of an iff/and/or/exists is constructive
    if head.op == "const" and head.name in {"And", "Or", "Iff", "Exists"}:
        return _emit_constructor_for(K, head)
    if head.op == "app" and head.children:
        f = head.children[0]
        if f.op == "const" and f.name in {"And", "Or", "Iff", "Exists"}:
            return _emit_constructor_for(K, f)


def _emit_constructor_for(K: RLIC, head: Expr) -> None:
    goal_cid = _find_cid(K, int(CellLabel.GOAL), "<goal>")
    if goal_cid is None:
        return
    K.add_cell(
        grade=2,
        label=int(CellLabel.CONSTRUCTOR_OPP),
        type_tag=int(TypeTag.PROP),
        name=head.name,
        boundary=[(goal_cid, RoleTag(Role.GOAL))],
    )


def _emit_induction_opps(P: ProofState, K: RLIC) -> None:
    if P.goal is None:
        return
    goal_cid = _find_cid(K, int(CellLabel.GOAL), "<goal>")
    if goal_cid is None:
        return
    # any free variable whose type tag matches the inductive set in the goal
    free_vars = {sub.name for sub in P.goal.walk() if sub.op == "var"}
    inductive_types = {int(TypeTag.NAT), int(TypeTag.LIST), int(TypeTag.INDUCTIVE)}
    for h in P.gamma:
        if h.is_proof:
            continue
        if h.name in free_vars and h.typ.type_tag in inductive_types:
            var_cid = _find_cid(K, int(CellLabel.VARIABLE), "<bound>")
            if var_cid is None:
                continue
            K.add_cell(
                grade=2,
                label=int(CellLabel.INDUCTION_OPP),
                type_tag=int(TypeTag.PROP),
                boundary=[
                    (var_cid, RoleTag(Role.INDUCTIVE)),
                    (goal_cid, RoleTag(Role.GOAL)),
                ],
            )


def _emit_apply_opps(P: ProofState, K: RLIC, cfg: ParserConfig) -> None:
    """For each (h, g) where head(type(h)) matches head(g) via first-order matching."""
    if P.goal is None:
        return
    goal_cid = _find_cid(K, int(CellLabel.GOAL), "<goal>")
    if goal_cid is None:
        return
    goal_head = _head_of(P.goal)
    if goal_head is None:
        return
    for h in P.gamma:
        if not h.is_proof:
            continue
        # if h's type is an arrow / forall, peel binders
        ty = _conclusion_of(h.typ)
        h_head = _head_of(ty)
        if h_head is None:
            continue
        if h_head == goal_head:
            hyp_key = cfg.canonical_hyp_name(h.name)
            hyp_cid = _find_cid(K, int(CellLabel.LOCAL_HYP), hyp_key)
            if hyp_cid is None:
                continue
            K.add_cell(
                grade=2,
                label=int(CellLabel.APPLY_OPP),
                type_tag=int(TypeTag.PROP),
                boundary=[
                    (hyp_cid, RoleTag(Role.PREMISE)),
                    (goal_cid, RoleTag(Role.CONCLUSION)),
                ],
            )


def _head_of(e: Expr) -> str | None:
    if e.op == "const":
        return e.name
    if e.op == "app" and e.children:
        return _head_of(e.children[0])
    return None


def _conclusion_of(e: Expr) -> Expr:
    if e.op == "binder" and e.name in {"forall", "lambda"} and len(e.children) == 2:
        return _conclusion_of(e.children[1])
    return e


# ---------------------------------------------------------------------------
# Algorithm 3: BuildCandidateAction(P, A, Π)
# ---------------------------------------------------------------------------


def build_candidate_action(
    P: ProofState,
    A: list[dict],
    cfg: ParserConfig,
) -> RLIC:
    """K_a(P, A): K_afford(P) plus one candidate-action cell per a ∈ A.

    Each `a` is a dict like {"family": "rewrite", "premise": "lemma_name"} or
    {"family": "intro"}. Boundaries lie entirely in K_afford(P).
    """
    K = build_afford(P, cfg)
    for a in A:
        family = a.get("family", "")
        # signature data ≠ tactic-family target label (Mode 2 caveat in §5.6)
        sig_label = int(CellLabel.APPLY_OPP)  # reuse code for now
        # boundary: cells already in K_afford referenced by a (e.g. the premise)
        boundary: list[tuple[int, RoleTag]] = []
        premise = a.get("premise")
        if premise:
            pc = _find_cid(K, int(CellLabel.CONSTANT), premise)
            if pc is not None:
                boundary.append((pc, RoleTag(Role.PREMISE)))
        K.add_cell(
            grade=2,
            label=sig_label,
            name=f"cand:{family}",
            boundary=boundary,
        )
    return K


# ---------------------------------------------------------------------------
# Canonical truncation (§5.5): used to enforce budget after parser emits cells.
# ---------------------------------------------------------------------------


def canonical_truncate(K: RLIC, cfg: ParserConfig) -> RLIC:
    """Discard cells beyond budget using a canonical, label-and-incidence-only policy.

    Sorts cells per grade-bucket by a stable key derived from (label, type_tag,
    boundary-class-signature), and keeps the first N. Isomorphic complexes
    truncate to isomorphic complexes (Theorem 6.3 condition (iv)).
    """
    buckets: dict[str, list[Cell]] = defaultdict(list)
    for c in K.cells:
        if c.grade == 0:
            buckets["K0"].append(c)
        elif c.grade == 1:
            buckets["K1"].append(c)
        else:
            buckets["Kge2"].append(c)

    def _key(c: Cell) -> tuple:
        # incidence signature: sorted multiset of (face.label, face.type_tag, role_int)
        sig = tuple(
            sorted(
                (
                    K.cells[fc].label,
                    K.cells[fc].type_tag,
                    role.as_int(),
                )
                for fc, role in c.boundary
            )
        )
        return (c.label, c.type_tag, sig)

    budgets = {
        "K0": cfg.budget_K0,
        "K1": cfg.budget_K1,
        "Kge2": cfg.budget_Kge2,
    }
    keep_ids: set[int] = set()
    for k, cells in buckets.items():
        cells.sort(key=_key)
        for c in cells[: budgets[k]]:
            keep_ids.add(c.cid)

    # rebuild RLIC, remapping ids and dropping boundary entries that reference
    # truncated cells.
    out = RLIC(source=K.source + ":trunc")
    old_to_new: dict[int, int] = {}
    for c in K.cells:
        if c.cid in keep_ids:
            nc = out.add_cell(
                grade=c.grade,
                label=c.label,
                type_tag=c.type_tag,
                name=c.name,
            )
            old_to_new[c.cid] = nc
    for c in K.cells:
        if c.cid not in old_to_new:
            continue
        nc = out.cells[old_to_new[c.cid]]
        for fc, role in c.boundary:
            if fc in old_to_new:
                nc.boundary.append((old_to_new[fc], role))
    return out
