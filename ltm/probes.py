"""Algorithm 4: NoLeakageProbes(parser, dataset).

Necessary, not sufficient, evidence for no leakage (Condition 3.6). Lemma 6.2
gives the formal contract; these probes are a runtime drift check.

Probe 1: Tactic-swap. K_afford(P) does not depend on the next human tactic.
Probe 2: Permuted-context. K_afford is invariant under bound-variable renaming.
Probe 3: Held-out lemma in retrieval shortlist. K_afford(K_a) doesn't differ on
         the K_afford portion when the human lemma is removed from A.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .parser import build_afford, build_candidate_action
from .proof_state import Expr, Hyp, ParserConfig, ProofState
from .rlic import RLIC


# ---------------------------------------------------------------------------
# Canonical RLIC comparison (modulo cell ids; respects label/type/grade/role)
# ---------------------------------------------------------------------------


def canonical_signature(K: RLIC) -> tuple:
    """A canonical, isomorphism-invariant signature of K.

    Coarse but sufficient for drift detection: sorted multiset of cell
    signatures, where each cell signature is (label, type_tag, grade, boundary
    multiset under canonical re-key).

    We compute a fixed-point colouring (Weisfeiler-Lehman-style) so isomorphic
    complexes produce identical signatures.
    """
    # initial colour = (label, type_tag, grade)
    colour = {c.cid: (c.label, c.type_tag, c.grade) for c in K.cells}
    for _ in range(4):
        new_colour: dict[int, tuple] = {}
        for c in K.cells:
            nb = tuple(
                sorted(
                    (colour[fc], role.as_int())
                    for fc, role in c.boundary
                    if fc in colour
                )
            )
            new_colour[c.cid] = (colour[c.cid], nb)
        # compress to ints to keep stable
        codes: dict[tuple, int] = {}
        compressed = {}
        for cid, key in new_colour.items():
            if key not in codes:
                codes[key] = len(codes)
            compressed[cid] = codes[key]
        if compressed == colour:
            colour = compressed
            break
        colour = compressed
    sig = tuple(sorted(colour.values()))
    return sig


# ---------------------------------------------------------------------------
# Bound-variable α-renaming
# ---------------------------------------------------------------------------


def _rename_in_expr(e: Expr, mapping: dict[str, str]) -> Expr:
    if e.op == "var" and e.name in mapping:
        return Expr(op="var", name=mapping[e.name], type_tag=e.type_tag)
    return Expr(
        op=e.op,
        name=mapping.get(e.name, e.name) if e.op == "binder" else e.name,
        children=tuple(_rename_in_expr(c, mapping) for c in e.children),
        type_tag=e.type_tag,
    )


def alpha_rename(P: ProofState, rng: random.Random) -> ProofState:
    """Random α-equivalent renaming of bound variables and hypothesis names.

    Constants, inductive type names, and the goal structure are preserved.
    """
    # collect renameables
    names = set()
    for h in P.gamma:
        names.add(h.name)
    if P.goal is not None:
        for sub in P.goal.walk():
            if sub.op == "var":
                names.add(sub.name)
            if sub.op == "binder" and sub.name:
                names.add(sub.name)
    pool = list(names)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(pool, shuffled))

    new_gamma = []
    for h in P.gamma:
        new_gamma.append(
            Hyp(
                name=mapping.get(h.name, h.name),
                typ=_rename_in_expr(h.typ, mapping),
                is_proof=h.is_proof,
            )
        )
    new_goal = _rename_in_expr(P.goal, mapping) if P.goal is not None else None
    return ProofState(
        theorem=P.theorem,
        state_idx=P.state_idx,
        gamma=new_gamma,
        goal=new_goal,
        namespace=P.namespace,
        file=P.file,
        next_tactic_family=P.next_tactic_family,
        next_tactic_premises=P.next_tactic_premises,
        is_solved_by_some_tactic=P.is_solved_by_some_tactic,
    )


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    name: str
    n_checked: int
    n_passed: int
    failing: list[str]

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_checked if self.n_checked else 1.0


def probe_tactic_swap(
    states: list[ProofState],
    cfg: ParserConfig,
    alt_families: tuple[str, ...] = ("rewrite", "apply", "intro", "induction", "rfl"),
) -> ProbeResult:
    """Probe 1: K_afford(P) computed with the recorded next tactic and with each
    alternative tactic must coincide. (build_afford ignores P.next_tactic_*, so
    this probe also acts as a smoke test for that property.)
    """
    failing: list[str] = []
    n_pass = 0
    for P in states:
        K_human = build_afford(P, cfg)
        sig_human = canonical_signature(K_human)
        ok = True
        for fam in alt_families:
            if fam == P.next_tactic_family:
                continue
            P_alt = ProofState(**{**P.__dict__, "next_tactic_family": fam})
            K_alt = build_afford(P_alt, cfg)
            if canonical_signature(K_alt) != sig_human:
                ok = False
                failing.append(f"{P.theorem}#{P.state_idx} differs under tactic={fam}")
                break
        if ok:
            n_pass += 1
    return ProbeResult("tactic_swap", len(states), n_pass, failing[:10])


def probe_permuted_context(
    states: list[ProofState],
    cfg: ParserConfig,
    seed: int = 0,
) -> ProbeResult:
    """Probe 2: K_afford(σ(P)) ≃ σ_*(K_afford(P)) under α-renaming."""
    rng = random.Random(seed)
    failing: list[str] = []
    n_pass = 0
    for P in states:
        K = build_afford(P, cfg)
        P_r = alpha_rename(P, rng)
        K_r = build_afford(P_r, cfg)
        # Under canonical_hyp_name + canonical_var_name policy, K and K_r should
        # have the same canonical signature.
        if canonical_signature(K) == canonical_signature(K_r):
            n_pass += 1
        else:
            failing.append(f"{P.theorem}#{P.state_idx}")
    return ProbeResult("permuted_context", len(states), n_pass, failing[:10])


def probe_heldout_lemma(
    states_with_actions: list[tuple[ProofState, list[dict], str]],
    cfg: ParserConfig,
) -> ProbeResult:
    """Probe 3: Removing the human lemma from A leaves K_afford unchanged and
    only the candidate-action cells differ.
    """
    failing: list[str] = []
    n_pass = 0
    for P, A, human_lemma in states_with_actions:
        A_minus = [a for a in A if a.get("premise") != human_lemma]
        Ka = build_candidate_action(P, A, cfg)
        Ka2 = build_candidate_action(P, A_minus, cfg)
        # K_afford portion = cells that are not candidate-action cells
        # (in our impl, candidate cells use APPLY_OPP label with name "cand:*")
        afford_sig = lambda K: canonical_signature(
            _filter_out_candidates(K)
        )
        if afford_sig(Ka) == afford_sig(Ka2):
            n_pass += 1
        else:
            failing.append(f"{P.theorem}#{P.state_idx}")
    return ProbeResult("heldout_lemma", len(states_with_actions), n_pass, failing[:10])


def _filter_out_candidates(K: RLIC) -> RLIC:
    out = RLIC(source=K.source + ":no-cand")
    keep_ids: dict[int, int] = {}
    for c in K.cells:
        if c.name and isinstance(c.name, str) and c.name.startswith("cand:"):
            continue
        nid = out.add_cell(
            grade=c.grade, label=c.label, type_tag=c.type_tag, name=c.name
        )
        keep_ids[c.cid] = nid
    for c in K.cells:
        if c.cid in keep_ids:
            nc = out.cells[keep_ids[c.cid]]
            for fc, role in c.boundary:
                if fc in keep_ids:
                    nc.boundary.append((keep_ids[fc], role))
    return out


def run_all_probes(
    states: list[ProofState],
    states_with_actions: list[tuple[ProofState, list[dict], str]],
    cfg: ParserConfig,
) -> dict[str, ProbeResult]:
    return {
        "tactic_swap": probe_tactic_swap(states, cfg),
        "permuted_context": probe_permuted_context(states, cfg),
        "heldout_lemma": probe_heldout_lemma(states_with_actions, cfg),
    }
