"""Proof state P schema fed into the parser.

This is a minimal Lean-like data model: just enough to run Algorithms 1–4
without committing to a full Lean elaboration. A LeanDojo adapter (later) will
populate these structures from real mathlib4 extractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Expression tree (the "term" object)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """A simple Lean-ish expression.

    op categories:
      - "var"   : payload is a string name (or de Bruijn key)
      - "const" : payload is a constant/lemma name
      - "app"   : children are the function then its args
      - "eq"    : binary equality, children = [lhs, rhs]
      - "binder": payload is "forall" or "lambda"; children = [bound_type, body]
                  bound variable name in `name`
      - "type"  : a type expression (e.g. "Nat", "List Nat")
    """

    op: str
    name: str = ""
    children: tuple["Expr", ...] = ()
    type_tag: int = 0  # τ tag id; coarse

    def __post_init__(self):
        # Normalise children to tuple for hashability
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))

    # convenience
    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


# ---------------------------------------------------------------------------
# Local context entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyp:
    name: str          # local hypothesis name (subject to §5.5 policy)
    typ: Expr          # its type / proposition
    is_proof: bool = False  # True if τ marks it as Prop-valued


# ---------------------------------------------------------------------------
# Proof state
# ---------------------------------------------------------------------------


@dataclass
class ProofState:
    """A single goal proof state P."""

    theorem: str = ""           # global identifier (e.g. mathlib lemma name)
    state_idx: int = 0          # position in the proof (0-based)
    gamma: list[Hyp] = field(default_factory=list)  # local context Γ
    goal: Expr | None = None    # goal proposition

    # provenance for splits / leakage probes
    namespace: str = ""
    file: str = ""

    # ground-truth (used by Tasks A/B/C; never read by parser)
    next_tactic_family: str = ""    # for Task A
    next_tactic_premises: tuple[str, ...] = ()  # for Task B (premise names)
    is_solved_by_some_tactic: bool = True       # for Task C target


# ---------------------------------------------------------------------------
# Parser configuration Π (§3.6 / §5.5)
# ---------------------------------------------------------------------------


@dataclass
class ParserConfig:
    """Π — label/role ontology, depth bounds, affordance generators, truncation."""

    depth_bound: int = 5
    # cell budgets (§7.4)
    budget_K0: int = 256
    budget_K1: int = 512
    budget_Kge2: int = 64

    # label preservation policy (§5.5)
    preserve_bound_var_names: bool = False     # α-equivalence
    preserve_hyp_names: bool = False           # configurable via Π
    preserve_constant_names: bool = True       # semantic identity
    preserve_inductive_names: bool = True
    preserve_namespace: bool = False           # leakage risk; off by default
    preserve_metavar_names: bool = False       # artefact
    preserve_universe_names: bool = False

    # which affordance generators run
    enable_rewrite_opp: bool = True
    enable_apply_opp: bool = True
    enable_induction_opp: bool = True
    enable_constructor_opp: bool = True

    # transparency level Θ for typeclass / kernel queries
    transparency_level: int = 0

    def canonical_var_name(self, raw: str) -> str:
        """Canonical key for bound variables under α-renaming.

        Default: collapse to a single placeholder; a Π extension can swap in
        de Bruijn indices instead.
        """
        if self.preserve_bound_var_names:
            return raw
        return "<bound>"

    def canonical_hyp_name(self, raw: str) -> str:
        if self.preserve_hyp_names:
            return raw
        return "<hyp>"
