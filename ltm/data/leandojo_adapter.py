"""LeanDojo Benchmark 4 adapter.

Schema (per random/{train,val,test}.json, top-level JSON array):
    [{
       "url": str, "commit": str, "file_path": str, "full_name": str,
       "theorem_statement": str,
       "start": [line, col], "end": [line, col],
       "traced_tactics": [
            {"tactic": str, "annotated_tactic": [...],
             "state_before": str, "state_after": str},
            ...
       ]
    }, ...]

Each traced_tactic gives one ProofState: hypotheses + goal from `state_before`,
tactic family from `tactic`, premises from the tactic's argument list, and
value target = 1 iff state_after != "no goals" (heuristic) — i.e. the proof
made progress; 0 if state_after == "no goals" (the proof finishes here, so the
"is_solved_by_some_tactic" flag is True), or some other interpretation.

For the paper §7.2 Task C target ("the recorded proof has a successful tactic
at this state"), we set value_target = 1 for every observed state, since by
construction the human proof did succeed; value diversity must come from
*negative* states (e.g. mined unsolvable states or randomised goals). For the
v1 evaluation we report Task C alongside Tasks A/B but note that the value
target is degenerate on this dataset.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..proof_state import Expr, Hyp, ProofState
from ..rlic import TypeTag


# ---------------------------------------------------------------------------
# Tactic-family vocabulary (§7.2 Task A)
# ---------------------------------------------------------------------------


FAMILY_MAP = {
    # rewrite-like
    "rw": "rewrite", "rewrite": "rewrite", "rwa": "rewrite",
    "simp_rw": "rewrite", "erw": "rewrite",
    # apply-like (proof term)
    "apply": "apply", "exact": "apply", "refine": "apply", "exact_mod_cast": "apply",
    "convert": "apply", "convert_to": "apply",
    # intro / extensionality
    "intro": "intro", "intros": "intro", "rintro": "intro", "ext": "intro",
    # case analysis / induction
    "cases": "induction", "rcases": "induction", "induction": "induction",
    "obtain": "induction", "split": "induction", "by_cases": "induction",
    # structural / constructors
    "constructor": "constructor", "use": "constructor", "left": "constructor",
    "right": "constructor",
    # finishing / simp-driven
    "rfl": "rfl",
    "simp": "simp", "simpa": "simp", "norm_num": "simp", "dsimp": "simp",
    "decide": "simp", "omega": "simp", "ring": "simp", "linarith": "simp",
    "field_simp": "simp", "abel": "simp",
    # have / let / show — auxiliary
    "have": "have", "let": "have", "show": "have", "suffices": "have",
    "set": "have", "specialize": "have", "change": "have",
    # other
}

TACTIC_FAMILIES = (
    "rewrite", "apply", "intro", "induction", "constructor",
    "rfl", "simp", "have", "other",
)


def tactic_family(tactic: str) -> str:
    tactic = tactic.lstrip()
    m = re.match(r"^([A-Za-z_][\w']*)", tactic)
    if not m:
        return "other"
    return FAMILY_MAP.get(m.group(1), "other")


_IDENT_RE = re.compile(r"^[A-Za-z_][\w'.]*")


def extract_premises(tactic: str) -> tuple[str, ...]:
    """Pull identifier-like tokens that look like lemma references.

    Pragmatic heuristic: tokens that start with uppercase or contain a dot are
    likely globals; lowercase one-word tokens are usually local hypotheses.
    """
    tac = tactic
    # strip brackets / parens / commas to expose tokens
    tac = re.sub(r"[\[\],\(\)]", " ", tac)
    parts = tac.split()
    if not parts:
        return ()
    out = []
    for tok in parts[1:]:
        if not _IDENT_RE.match(tok):
            continue
        if tok in {"at", "with", "using", "_", "←", "→", ":"}:
            continue
        # likely a global lemma: dotted path OR starts capital
        if "." in tok or tok[0].isupper():
            out.append(tok)
    return tuple(out)


# ---------------------------------------------------------------------------
# Pretty-state parser (handles real Lean 4 pp output)
# ---------------------------------------------------------------------------


HYP_RE = re.compile(r"^([\w'⊢↑⊥↓→∀∃⟨⟩.✝⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+(?:\s+[\w'.]+)*)\s*:\s*(.+)$")
GOAL_RE = re.compile(r"^⊢\s*(.+)$")


# Coarse type-tag heuristics on the type-text fragment
def coarse_type_tag(text: str) -> int:
    t = text.strip()
    if t == "Prop" or t.startswith("Prop"):
        return int(TypeTag.PROP)
    if re.match(r"^(Nat|ℕ)\b", t):
        return int(TypeTag.NAT)
    if re.match(r"^(Int|ℤ)\b", t):
        return int(TypeTag.INT)
    if t.startswith("List"):
        return int(TypeTag.LIST)
    if re.match(r"^Type\b", t):
        return int(TypeTag.TYPE_T)
    if "→" in t:
        return int(TypeTag.FUNC)
    # heuristic: a proposition often contains a binary relation symbol
    if re.search(r"[=≤≥<>≠∈∉⊆⊂]", t) or t.startswith(("∀", "∃")):
        return int(TypeTag.PROP)
    return int(TypeTag.UNKNOWN)


def _parse_expr(text: str, depth: int = 0) -> Expr:
    """Coarse parenthesised parse: head + args; recognises =, ∧, ∨, →.

    Not a real Lean parser — keeps semantics light for the prototype.
    """
    text = text.strip()
    if depth > 6 or not text:
        return Expr(op="const", name=text or "_", type_tag=coarse_type_tag(text))

    # strip outer parens
    while text.startswith("(") and text.endswith(")") and _paren_balanced(text[1:-1]):
        text = text[1:-1].strip()
        if not text:
            return Expr(op="const", name="_")

    # binary operators at depth 0
    for op_str, op_name in (
        (" ↔ ", "iff"), (" → ", "arrow"),
        (" ∧ ", "and"), (" ∨ ", "or"),
        (" = ", "eq"), (" ≠ ", "ne"),
        (" ≤ ", "le"), (" ≥ ", "ge"),
        (" < ", "lt"), (" > ", "gt"),
        (" + ", "add"), (" - ", "sub"),
        (" * ", "mul"), (" / ", "div"),
    ):
        i = _find_top_level(text, op_str)
        if i >= 0:
            lhs = text[:i].strip()
            rhs = text[i + len(op_str):].strip()
            if op_name == "eq":
                return Expr(op="eq", name="",
                            children=(_parse_expr(lhs, depth+1), _parse_expr(rhs, depth+1)),
                            type_tag=int(TypeTag.PROP))
            head = Expr(op="const", name=op_name, type_tag=int(TypeTag.PROP if op_name in {"iff","and","or","ne","le","ge","lt","gt","arrow"} else TypeTag.UNKNOWN))
            return Expr(op="app", name="",
                        children=(head, _parse_expr(lhs, depth+1), _parse_expr(rhs, depth+1)),
                        type_tag=int(TypeTag.PROP) if op_name in {"iff","and","or","ne","le","ge","lt","gt"} else int(TypeTag.UNKNOWN))

    # quantifiers / binders
    if text.startswith("∀") or text.startswith("∃"):
        kw = "forall" if text.startswith("∀") else "exists"
        body_text = text[1:].strip()
        # split on first ","
        comma = _find_top_level(body_text, ",")
        if comma >= 0:
            head_text = body_text[:comma].strip().lstrip("(").rstrip(")")
            tail = body_text[comma+1:].strip()
            return Expr(op="binder", name=kw,
                        children=(_parse_expr(head_text, depth+1), _parse_expr(tail, depth+1)),
                        type_tag=int(TypeTag.PROP))

    # application: split on whitespace at top level
    tokens = _split_top_level(text)
    if len(tokens) == 1:
        t = tokens[0]
        if not t:
            return Expr(op="const", name="_")
        if t[0].islower() or t[0] == "_":
            return Expr(op="var", name=t, type_tag=coarse_type_tag(t))
        return Expr(op="const", name=t, type_tag=coarse_type_tag(t))
    head, *args = tokens
    head_e = (Expr(op="const", name=head, type_tag=coarse_type_tag(head))
              if head and head[0].isupper() else
              Expr(op="var", name=head, type_tag=coarse_type_tag(head)))
    arg_es = tuple(_parse_expr(a, depth+1) for a in args)
    return Expr(op="app", name="", children=(head_e,) + arg_es,
                type_tag=int(TypeTag.UNKNOWN))


def _paren_balanced(s: str) -> bool:
    n = 0
    for ch in s:
        if ch == "(":
            n += 1
        elif ch == ")":
            n -= 1
            if n < 0:
                return False
    return n == 0


def _find_top_level(text: str, needle: str) -> int:
    """Find the leftmost occurrence of `needle` not enclosed by () or {}."""
    depth = 0
    i = 0
    L = len(needle)
    while i <= len(text) - L:
        ch = text[i]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and text[i:i+L] == needle:
            return i
        i += 1
    return -1


def _split_top_level(text: str) -> list[str]:
    """Split on whitespace at parenthesis-depth 0."""
    out = []
    depth = 0
    cur = []
    for ch in text:
        if ch in "({[":
            depth += 1; cur.append(ch)
        elif ch in ")}]":
            depth -= 1; cur.append(ch)
        elif ch.isspace() and depth == 0:
            if cur:
                out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def parse_tactic_state(text: str) -> tuple[list[Hyp], Expr | None]:
    """Parse a LeanDojo TacticState pp text into hyps + goal.

    Uses ``lean_dojo.parse_goals`` for the hyp/goal split (AST-grade) and our
    own recursive expression parser for the term structure. ``parse_goals``
    handles multi-line declarations and instance synthesis variables (``inst✝``)
    cleanly; if it fails on a state we fall back to the regex split.
    """
    if text is None or text.strip() == "no goals":
        return [], None
    try:
        from lean_dojo import parse_goals as _ld_parse_goals
        goals = _ld_parse_goals(text)
    except Exception:
        return _parse_tactic_state_regex(text)
    if not goals:
        return [], None
    g = goals[0]  # focus on the first sub-goal
    hyps: list[Hyp] = []
    for d in g.assumptions:
        ty_text = d.lean_type or ""
        ty_expr = _with_type_tag(_parse_expr(ty_text), coarse_type_tag(ty_text))
        is_proof = (ty_expr.type_tag == int(TypeTag.PROP))
        hyps.append(Hyp(name=d.ident, typ=ty_expr, is_proof=is_proof))
    goal_text = g.conclusion or ""
    goal_expr = _with_type_tag(_parse_expr(goal_text), coarse_type_tag(goal_text))
    return hyps, goal_expr


def _with_type_tag(e: Expr, tt: int) -> Expr:
    """Propagate the outer type-text's coarse tag onto the parsed Expr."""
    if tt == int(TypeTag.UNKNOWN) or tt == e.type_tag:
        return e
    return Expr(op=e.op, name=e.name, children=e.children, type_tag=tt)


def _parse_tactic_state_regex(text: str) -> tuple[list[Hyp], Expr | None]:
    """Fallback regex-based parser (pre-parse_goals path), used only when
    lean_dojo.parse_goals raises on a state."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    hyps: list[Hyp] = []
    goal: Expr | None = None
    merged: list[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        if not stripped:
            continue
        if stripped.startswith("⊢") or HYP_RE.match(stripped):
            merged.append(stripped)
        elif merged:
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)
    for ln in merged:
        m = GOAL_RE.match(ln)
        if m:
            goal = _parse_expr(m.group(1))
            continue
        m = HYP_RE.match(ln)
        if m:
            names_text, ty = m.group(1), m.group(2)
            names = names_text.split()
            ty_expr = _parse_expr(ty)
            is_proof = (coarse_type_tag(ty) == int(TypeTag.PROP))
            for nm in names:
                hyps.append(Hyp(name=nm, typ=ty_expr, is_proof=is_proof))
    return hyps, goal


# ---------------------------------------------------------------------------
# Driver: from a LeanDojo Benchmark 4 split file → list[ProofState]
# ---------------------------------------------------------------------------


# Paper §7.1: algebra, logic, basic numerical libraries.
DEFAULT_FILE_PREFIXES = (
    "Mathlib/Algebra/",
    "Mathlib/Logic/",
    "Mathlib/Data/Nat/",
    "Mathlib/Data/Int/",
    "Mathlib/Data/Rat/",
    "Mathlib/Data/Real/",
    "Mathlib/Order/",
)


def _namespace_of(full_name: str | None) -> str:
    if not full_name:
        return ""
    return full_name.split(".")[0]


def from_split_file(
    path: str | Path,
    *,
    file_prefixes: tuple[str, ...] = DEFAULT_FILE_PREFIXES,
    max_theorems: int | None = None,
    max_states: int | None = None,
    goal_max_chars: int = 1024,
    proof_max_length: int = 30,
) -> list[ProofState]:
    """Stream a LeanDojo split file, filter, and yield ProofStates."""
    with open(path) as f:
        data = json.load(f)
    out: list[ProofState] = []
    theorem_count = 0
    for thm in data:
        fp = thm.get("file_path") or ""
        if file_prefixes and not any(fp.startswith(pf) for pf in file_prefixes):
            continue
        traced = thm.get("traced_tactics", [])
        if not traced:
            continue
        if proof_max_length and len(traced) > proof_max_length:
            continue
        for idx, tt in enumerate(traced):
            sb = tt.get("state_before") or ""
            if not sb or sb.strip() == "no goals":
                continue
            if len(sb) > goal_max_chars * 4:
                continue
            hyps, goal = parse_tactic_state(sb)
            if goal is None or len(goal.children) > 20:
                continue
            tactic = tt.get("tactic") or ""
            fam = tactic_family(tactic)
            premises = extract_premises(tactic)
            state_after = tt.get("state_after") or ""
            P = ProofState(
                theorem=thm.get("full_name") or "",
                state_idx=idx,
                gamma=hyps,
                goal=goal,
                namespace=_namespace_of(thm.get("full_name")),
                file=fp,
                next_tactic_family=fam,
                next_tactic_premises=premises,
                is_solved_by_some_tactic=True,  # by construction; see module docstring
            )
            out.append(P)
            if max_states and len(out) >= max_states:
                return out
        theorem_count += 1
        if max_theorems and theorem_count >= max_theorems:
            break
    return out


def family_distribution(states: list[ProofState]) -> Counter:
    return Counter(s.next_tactic_family for s in states)


def build_premise_vocab(states: list[ProofState], max_size: int = 4096) -> tuple[str, ...]:
    """Hashed/frequency-based premise vocab from training states' premises.

    Used only when the global corpus is unavailable.
    """
    c = Counter()
    for s in states:
        for p in s.next_tactic_premises:
            c[p] += 1
    common = [p for p, _ in c.most_common(max_size - 1)]
    return ("<pad>",) + tuple(common)


def build_premise_vocab_from_corpus(
    corpus_path: str | Path,
    *,
    name_field: str = "name",
    max_size: int = 16384,
) -> tuple[str, ...]:
    """Build the *global* premise pool from LeanDojo's corpus.jsonl.

    The corpus lists every public lemma/theorem/definition in the traced repo.
    For Task B retrieval (§7.2) this gives the realistic negative pool the
    paper specifies, instead of in-batch negatives only.
    """
    import json
    seen = set()
    out = ["<pad>"]
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # corpus records contain either `name` or `full_name`; also support
            # `premises` lists (some corpus formats nest them per file).
            for k in (name_field, "full_name", "name"):
                v = rec.get(k)
                if isinstance(v, str) and v and v not in seen:
                    seen.add(v)
                    out.append(v)
            # nested premises
            ps = rec.get("premises", [])
            if isinstance(ps, list):
                for p in ps:
                    if isinstance(p, dict):
                        n = p.get("full_name") or p.get("name")
                        if isinstance(n, str) and n and n not in seen:
                            seen.add(n); out.append(n)
            if len(out) >= max_size:
                break
    return tuple(out[:max_size])


def synthesize_value_negatives(
    states: list[ProofState],
    *,
    rng_seed: int = 0,
    fraction: float = 0.3,
) -> list[ProofState]:
    """Synthesize unsolvable counterparts by swapping goals across theorems.

    Heuristic: for a chosen state P, replace its goal with the goal of a
    randomly selected *other* theorem. The hypotheses no longer match the goal,
    so no tactic in the original proof of P would apply. These negatives carry
    ``is_solved_by_some_tactic=False`` and are unambiguously distinguishable
    from the originals only by the cell-class incidence structure of the new
    state.
    """
    import random
    rng = random.Random(rng_seed)
    pool = [P for P in states if P.goal is not None]
    if len(pool) < 2:
        return []
    out: list[ProofState] = []
    for P in states:
        if rng.random() >= fraction:
            continue
        # pick a donor goal from a *different* theorem
        for _ in range(10):
            donor = rng.choice(pool)
            if donor.theorem != P.theorem and donor.goal is not None:
                break
        else:
            continue
        out.append(
            ProofState(
                theorem=P.theorem + ".value_neg",
                state_idx=P.state_idx,
                gamma=list(P.gamma),
                goal=donor.goal,
                namespace=P.namespace + ".value_neg",
                file=P.file,
                next_tactic_family="other",
                next_tactic_premises=(),
                is_solved_by_some_tactic=False,
            )
        )
    return out
