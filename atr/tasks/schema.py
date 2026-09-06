"""Task and score definitions."""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict

# Task families. Keep these stable: every report, every data-mix knob and every
# RL curriculum weight is keyed on them.
# The benchmark is MuSiQue-style multi-hop retrieval: the single tool is BM25
# `search` over a candidate passage set, and the model chains passages hop by
# hop to extract the answer.
TASK_TYPES = [
    "musique_2hop",    # single BM25 search resolves a 2-link chain
    "musique_3hop",    # chain 3: two dependent searches resolve a 3-link chain
    "musique_4hop",    # chain 4: three dependent searches resolve a 4-link chain
    "unanswerable",    # the information genuinely is not in the corpus; must say so
    "no_tool",         # answerable from the model's own knowledge; tools would be wrong
]


@dataclass
class Task:
    task_id: str
    seed: int
    prompt: str
    task_type: str
    difficulty: int                       # 0 easy .. 3 hard, ~= dependent hops
    gold: dict                            # {"kind": "numeric|text|any_of|none", "value": ...}
    tier: int = 0                         # tool-chain length: how many tool calls the
                                          # reference solution needs (0 = no tool needed).
                                          # The (A) curriculum tier -- see TASK_TIERS in
                                          # generator.py. Mirrors oracle_plan length for
                                          # tool families.
    required_tools: list[str] = field(default_factory=list)   # must all appear
    required_any: list[list[str]] = field(default_factory=list)  # each group: >=1 must appear
    forbidden_tools: list[str] = field(default_factory=list)  # must never appear
    oracle_plan: list[dict] = field(default_factory=list)     # reference calls
    oracle_answer: str = ""
    route: list[str] = field(default_factory=list)            # _REL keys of the chain;
                                                              # the real diversity axis (see
                                                              # rejection._shape). Empty for
                                                              # families with no chain.
    expect_side_effect: dict | None = None                    # e.g. {"to": ..., "must_contain": [...]}
    notes: str = ""
    # Optional explicit 20-passage candidate set (real MuSiQue train tasks).
    # When non-empty this REPLACES the seeded synthetic world for the episode:
    # the task owns its documents instead of deriving them from build_world(seed).
    # Shape mirrors the judge probe: [{"doc_id", "title", "text", "facts", "links"}].
    documents: list[dict] = field(default_factory=list)

    @property
    def needs_tool(self) -> bool:
        return self.task_type not in ("no_tool",)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(**d)


@dataclass
class ScoreCard:
    task_id: str
    task_type: str
    difficulty: int
    # --- headline ---
    success: bool = False
    # --- per-objective sub-metrics (this is the diagnostic that drives the next data batch) ---
    format_strict: bool = False        # every turn parsed with zero repairs
    format_loose: bool = False         # every turn parsed at all
    necessity_ok: bool = False         # used tools iff tools were needed
    selection_ok: bool = False         # all required tools called, no forbidden ones
    args_ok: bool = False              # the decisive call carried the right arguments
    recovery_ok: bool | None = None    # None when no error was encountered
    final_correct: bool = False
    final_f1: float = 0.0              # SQuAD/MuSiQue token-F1 vs the gold string;
                                       # the OFFICIAL judge metric, and what the RL
                                       # reward optimises. `final_correct` stays a
                                       # boolean purely for internal diagnostics.
    side_effect_ok: bool | None = None
    # --- efficiency / diagnostics ---
    num_steps: int = 0
    num_calls: int = 0
    num_tool_errors: int = 0
    oracle_calls: int = 0
    stop_reason: str = ""
    failure_mode: str = ""             # single most informative reason it failed
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# answer matching
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# token-level F1 (the official judge metric)
# ---------------------------------------------------------------------------
# The judge scores SQuAD/MuSiQue-style token F1 against a short gold string, not
# exact match. `match_answer` above stays as-is because it drives `success`, the
# internal boolean diagnostic; F1 is what the reward optimises, so the two must
# not be conflated. Anything scored here has to reproduce the official
# normaliser exactly, or every reward we compute is measuring our own dialect.
_PUNCT = set(string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _norm_tokens(s: str) -> list[str]:
    """The SQuAD normaliser: lower -> drop punctuation -> drop articles -> split.

    Punctuation is DELETED rather than replaced with a space, which is what the
    official implementation does and is load-bearing here: it makes "3,456,000"
    and "3456000" the same single token instead of three."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    return s.split()          # .split() with no arg also collapses whitespace


def _f1_tokens(gold_toks: list[str], pred_toks: list[str]) -> float:
    if not gold_toks or not pred_toks:
        return 0.0
    # MULTISET intersection: SQuAD F1 counts token multiplicity, so a prediction
    # that repeats a gold token does not get credit for it twice. Using set
    # intersection here silently inflates precision on repetitive answers.
    n_same = sum((Counter(gold_toks) & Counter(pred_toks)).values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _gold_strings(gold: dict | str | None) -> list[str]:
    """Candidate gold surface forms. `any_of` scores against its best member;
    `all_of` scores against the concatenation (the answer must carry them all);
    `none` yields nothing -- abstention is not a string-overlap question and the
    verifier scores it separately."""
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold]
    kind = gold.get("kind", "text")
    value = gold.get("value")
    if kind == "none" or value is None:
        return []
    if kind == "any_of":
        return [str(v) for v in value]
    if kind == "all_of":
        return [" ".join(str(v) for v in value)]
    return [str(value)]


def answer_f1(gold: dict | str | None, pred: str | None) -> float:
    """Token-level F1 of `pred` against `gold`, in [0.0, 1.0].

    `gold` is either a raw gold string or a Task.gold dict. An empty or missing
    prediction scores 0.0 -- deliberately, even against an empty gold, so that
    silence can never be farmed for reward."""
    if not pred or not str(pred).strip():
        return 0.0
    pred_toks = _norm_tokens(str(pred))
    best = 0.0
    for g in _gold_strings(gold):
        best = max(best, _f1_tokens(_norm_tokens(g), pred_toks))
    return best


NEGATIVE_MARKERS = [
    "not available", "no information", "cannot be determined", "can't be determined",
    "not found", "does not exist", "doesn't exist", "no such", "unable to find",
    "not enough information", "insufficient information", "no record", "not in the",
    "unavailable", "cannot answer", "can't answer", "no data",
]


def match_answer(gold: dict, answer: str | None, strict: bool = False) -> tuple[bool, str]:
    """Return (correct, reason).

    strict=True closes the number-shotgunning loophole: more than 2 distinct
    numbers in the reply counts as a miss even when one of them matches."""
    if answer is None:
        return False, "no_answer"
    kind = gold.get("kind", "text")

    if kind == "numeric":
        want = float(gold["value"])
        tol = float(gold.get("tol", max(0.01, abs(want) * 0.005)))
        nums = extract_numbers(answer)
        if not nums:
            return False, "no_number_in_answer"
        if any(abs(n - want) <= tol for n in nums):
            distinct = {round(n, 6) for n in nums}
            if strict and len(distinct) > 2:
                return False, "shotgun_strict"
            if len(distinct) > 4:
                return True, "matched_but_verbose"
            return True, "ok"
        return False, "wrong_number"

    if kind == "text":
        want = norm_text(str(gold["value"]))
        got = norm_text(answer)
        if not want:
            return False, "empty_gold"
        return (want in got, "ok" if want in got else "text_mismatch")

    if kind == "any_of":
        got = norm_text(answer)
        for v in gold["value"]:
            if norm_text(str(v)) in got:
                return True, "ok"
        return False, "text_mismatch"

    if kind == "all_of":
        got = norm_text(answer)
        missing = [v for v in gold["value"] if norm_text(str(v)) not in got]
        return (not missing, "ok" if not missing else f"missing:{missing[:3]}")

    if kind == "none":  # correct behaviour is to report that it cannot be answered
        got = (answer or "").lower()
        return (any(m in got for m in NEGATIVE_MARKERS), "ok" if any(m in got for m in NEGATIVE_MARKERS)
                else "did_not_abstain")

    raise ValueError(f"unknown gold kind {kind!r}")
