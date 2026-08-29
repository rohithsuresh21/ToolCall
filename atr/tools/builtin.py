"""
The stand-in tool set for the single-tool BM25 retrieval contract.

The judge collapsed the tool set to ONE action: BM25 retrieval over a provided
candidate set of passages. There is no calculator, no database, no web_search
and no fetch_page -- the whole task is: given Wikipedia-style candidate
passages, retrieve the relevant ones via BM25 and compose the answer across
hops.

Design intent (this is the part worth copying when the real tools land):

* ONE tool: `search`. It is a genuine BM25 ranker over `world.documents` (the
  candidate set for that episode). It returns full passage texts so a retrieved
  passage can directly supply the next hop's query and, at the leaf, the answer.
* Several queries fail in *readable* ways (empty query, zero hits). Recovering
  from an empty result by rephrasing is a trained skill, not an accident.
* Model-caused failures come back as dict payloads with an "error" key, never
  as raised exceptions (the registry wraps fn into that contract, and ToolError
  here is the registry's signal for such a payload).
"""
from __future__ import annotations

import math
import re
from typing import Any

from .registry import ToolError, ToolRegistry, ToolSpec

_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())]


def _bm25(term_freq: int, doc_len: int, avg_len: float, n_docs: int, df: int) -> float:
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    tf = term_freq * (_K1 + 1) / (term_freq + _K1 * (1 - _B + _B * doc_len / max(avg_len, 1e-9)))
    return idf * tf


def _build_index(world):
    """Lazy per-call index of the candidate set (small sets; simple & correct)."""
    docs = world.documents
    n = len(docs)
    lens = [len(_tokens(d["title"] + " " + d["text"])) for d in docs]
    avg = sum(lens) / max(n, 1)
    # term -> set of doc indices
    postings: dict[str, set[int]] = {}
    for i, d in enumerate(docs):
        for t in set(_tokens(d["title"] + " " + d["text"])):
            postings.setdefault(t, set()).add(i)
    return docs, lens, avg, postings


def _search(world, query: str, top_k: int = 3) -> dict:
    q = _tokens(query)
    if not q:
        raise ToolError("query is empty", kind="empty_query", hint="pass some search terms")
    docs, lens, avg, postings = _build_index(world)
    n = len(docs)

    scores: list[tuple[float, int]] = []
    for i in range(n):
        sc = 0.0
        for t in q:
            tf = 0
            if t in postings:
                tf = sum(1 for tt in _tokens(docs[i]["title"] + " " + docs[i]["text"]) if tt == t)
            df = len(postings.get(t, set()))
            if tf > 0:
                sc += _bm25(tf, lens[i], avg, n, df)
                # boost title matches slightly (title words are strong signals)
                if t in _tokens(docs[i]["title"]):
                    sc *= 1.6
        if sc > 0:
            scores.append((sc, i))
    scores.sort(key=lambda x: (-x[0], docs[x[1]]["doc_id"]))

    k = max(1, min(int(top_k), 10))
    hits = []
    for sc, i in scores[:k]:
        hits.append({
            "doc_id": docs[i]["doc_id"],
            "title": docs[i]["title"],
            "text": docs[i]["text"],
            "score": round(sc, 4),
        })
    # An empty result is a legitimate outcome the model must handle by
    # rephrasing -- not an exception.
    return {"query": query, "num_results": len(hits), "results": hits}


def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(ToolSpec(
        name="search",
        description=("Search the provided candidate passages using BM25 and return the "
                     "most relevant full passages (title + text). Pass targeted keywords to "
                     "pull the passage you need; read its text to find the next link or the answer."),
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "Search terms / keywords."},
            "top_k": {"type": "integer", "description": "Max passages to return (1-10). Default 3."},
        }, "required": ["query"]},
        fn=_search))
    return r
