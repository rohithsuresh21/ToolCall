"""
LLM passage naturalization (facts untouched).

Part A: vary the fixed per-kind templated prose in `atr.tools.world._passage()`
into varied, natural Wikipedia-style paragraphs -- WITHOUT the LLM ever inventing
or altering a fact.

Two hard invariants, enforced by post-check, never by trust:

1. Fact-presence: every attribute value the original template surfaced must still
   appear in the naturalized text. Missing OR added facts -> reject and retry.
2. Scoring isolation: naturalization only ever replaces `doc.text`. It NEVER
   touches the entity `attrs` dict, `doc.title`, `doc.facts`, or anything
   verifiers.py reads (`traj.call_log`, `traj.final_answer`). So gold-answer
   computation and every scorer are byte-identical whether or not naturalization
   is enabled (proven by tests/test_naturalize.py).

This runs OFFLINE only (scripts/70_naturalize_passages.sh), never inside
build_world() or at train/eval time. An opt-in loader injects cached naturalized
text into build_world() when enabled.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

from ..tasks.schema import norm_text

_MAX_RETRIES = 3


def _format_for_check(value: Any) -> str:
    """Normalised, comparable form of a fact value for a presence check.

    Numbers are comma-formatted the way the template surfaces populations so a
    value like 9_800_000 matches "9,800,000" and the bare "9800000"."""
    if value is None:
        return norm_text("")
    if isinstance(value, bool):
        return norm_text(str(value))
    if isinstance(value, (int, float)):
        return norm_text(f"{int(value):,}") if isinstance(value, int) else norm_text(str(value))
    if isinstance(value, (list, tuple)):
        return norm_text(" ".join(str(x) for x in value))
    return norm_text(str(value))


def _text_presence(haystack: str, needle_normed: str) -> bool:
    """True when the normalised fact value appears in the normalised passage text.

    Falls back to a formatting-robust comparison that strips every non-alphanumeric
    so "1,097,000" / "1 097 000" / "1097000" all match the same population fact."""
    if not needle_normed:
        return True
    h = norm_text(haystack)
    if needle_normed in h:
        return True

    def bare(s: str) -> str:
        return "".join(ch for ch in s if ch.isalnum())

    return bare(needle_normed) in bare(h)


def _surfaced_facts(passage: dict) -> list[str]:
    """The attribute values the ORIGINAL template actually surfaced, as normalized
    strings. Any fact value that appears in the original templated text must
    survive naturalization; facts never surfaced in the first place are exempt."""
    original = norm_text(passage.get("text", ""))
    out: set[str] = set()
    facts = passage.get("facts") or {}
    for v in facts.values():
        n = _format_for_check(v)
        if n and n in original:
            out.add(n)
    # title is always surfaced in prose and must be preserved too
    tn = norm_text(passage.get("title", ""))
    if tn:
        out.add(tn)
    return sorted(out)


def _facts_present(passage: dict, new_text: str) -> bool:
    """Every surfaced fact value must be present in the new text."""
    for n in _surfaced_facts(passage):
        if not _text_presence(new_text, n):
            return False
    return True


def _build_prompt(passage: dict) -> str:
    """Instruction that asks for a varied rewrite forcing the model to keep the facts."""
    attrs = json.dumps(passage.get("facts") or {}, sort_keys=True)
    return (
        "Rewrite the following Wikipedia-style passage in different, natural prose. "
        "Keep the exact same facts for this entity. You MUST use every one of these "
        "fact values somewhere in the new text, and you MUST NOT add any fact that is "
        "not in this attribute list. The entity is named in the CURRENT PASSAGE -- keep "
        "its name. Return only the rewritten passage, no preamble.\n"
        f"FACTS: {attrs}\n"
        f"CURRENT PASSAGE: \"{passage.get('text', '')}\""
    )


def _call_llm(llm_client, prompt: str) -> str:
    """Dispatch to the LLM client. Accepts either a callable(prompt)->str or an
    object exposing .complete(prompt)->str (OpenAI-compatible, like OpenAIBackend)."""
    if callable(llm_client):
        return llm_client(prompt)
    method = getattr(llm_client, "complete", None)
    if method is None:
        raise TypeError("llm_client must be callable or expose .complete(prompt)")
    return method(prompt)


def naturalize_passage(passage: dict, llm_client) -> dict:
    """Return a copy of `passage` whose `text` is naturally varied prose that keeps
    every surfaced fact. On repeated fact-presence failure, falls back to the
    original templated text -- a passage must never ship without valid text.

    Only `text` (and the `title`) may be touched; `facts` and everything else are
    returned unchanged so scoring never reads naturalized prose."""
    out = dict(passage)
    original = passage.get("text", "")
    for _ in range(_MAX_RETRIES):
        new_text = _call_llm(llm_client, _build_prompt(passage)).strip() or original
        if _facts_present(passage, new_text):
            out["text"] = new_text
            return out
    # exhausted retries: ship the known-good templated text rather than a broken one
    out["text"] = original
    out["naturalized"] = False
    return out


def naturalize_passages(world, llm_client) -> dict:
    """Naturalize every document passage in a World in place (returns the world).
    Only `world.documents[..]['text']` is rewritten; `world.entities` and the gold
    answer inputs (attrs) are untouched. Returns the world for chaining."""
    stats = {"docs": len(world.documents), "naturalized": 0, "retries": 0, "fell_back": 0}
    for d in world.documents:
        if not d.get("text"):
            continue
        before = d["text"]
        nd = naturalize_passage(d, llm_client)
        if nd["text"] != before:
            stats["naturalized"] += 1
            if nd.get("naturalized") is False:
                stats["fell_back"] += 1
        d["text"] = nd["text"]
    return stats


# ---------------------------------------------------------------------------
# offline cache + opt-in loader
# ---------------------------------------------------------------------------
def load_naturalized_loader(cache_path: str | pathlib.Path) -> Callable | None:
    """Load a cache of naturalized passage text produced by 70_naturalize_passages.sh.

    Returns a callable (seed:int, doc_id:str) -> str|None that yields cached text
    for a (seed, doc_id) pair, or None when the cache is missing/empty. Pass it to
    build_world(seed, text_loader=...) to use naturalized prose instead of templated.
    """
    p = pathlib.Path(cache_path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    by_seed: dict[str, dict[str, str]] = {}
    for k, v in data.items():
        seed_str, _, doc_id = k.rpartition(":")
        by_seed.setdefault(seed_str, {})[doc_id] = v
    if not by_seed:
        return None

    def loader(seed: int, doc_id: str) -> str | None:
        inner = by_seed.get(str(seed))
        return inner.get(doc_id) if inner else None

    return loader
