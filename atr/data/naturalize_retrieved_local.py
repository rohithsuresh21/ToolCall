"""Naturalize ONLY the passages a task's BM25 searches actually retrieve.

Much cheaper than full-world naturalization: a task's oracle plan touches a few
passages, so we spend LLM calls (local Ollama) only where the agent reads prose,
instead of rewriting all ~38 passages per world. One (seed, doc_id) cache entry
per retrieved passage; non-retrieved passages stay templated at build_world time.

Usage:
    python -m atr.data.naturalize_retrieved_local --n 400 --seed-start 0 \
        --model mistral-local:latest --cache artifacts/naturalized_passages.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from ..tasks.generator import generate
from ..tools.builtin import _search
from ..tools.world import build_world
from .naturalize import naturalize_selected
from .naturalize_local import DEFAULT_MODEL, DEFAULT_URL, make_complete


def resolve_retrieved(world, task) -> set[str]:
    """All doc_ids the task's oracle plan would retrieve on the templated world."""
    out: set[str] = set()
    for step in getattr(task, "oracle_plan", []) or []:
        args = step.get("arguments") or {}
        query = args.get("query")
        top_k = args.get("top_k", 3)
        if not query:
            continue
        res = _search(world, query, top_k=top_k)
        for hit in res.get("results") or []:
            if hit.get("doc_id"):
                out.add(hit["doc_id"])
    return out


def run(n: int, seed_start: int, complete, cache_path: str | None, mix: dict | None = None) -> dict:
    loaded = _load(cache_path) if cache_path else {}
    tasks = generate(n, seed_start=seed_start, filter_shortcuts=True, mix=mix)
    total = {"tasks": len(tasks), "docs": 0, "naturalized": 0, "retries": 0, "fell_back": 0}
    per_task_selected: Counter = Counter()
    for t in tasks:
        w = build_world(t.seed)
        sel = resolve_retrieved(w, t)
        per_task_selected[t.task_type] += len(sel)
        # skip (seed, doc_id) already cached by an earlier task sharing this world
        todo = {d for d in sel if f"{t.seed}:{d}" not in loaded}
        if not todo:
            continue
        s = naturalize_selected(w, todo, complete)
        for doc in w.documents:
            if doc.get("doc_id") in todo and doc.get("naturalized") and doc["text"]:
                loaded[f"{t.seed}:{doc['doc_id']}"] = doc["text"]
        total["docs"] += s["docs"]
        total["naturalized"] += s["naturalized"]
        total["retries"] += s.get("retries", 0) or 0
        total["fell_back"] += s["fell_back"]
        if cache_path:
            _write(cache_path, loaded)          # write per task -> resumable
    return {**total, "cache_entries": len(loaded),
            "by_task_retrieved": dict(per_task_selected), "cache": cache_path}


def _load(cache_path: str) -> dict:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(cache_path: str, data: dict) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Naturalize only the passages each task retrieves (Ollama)")
    ap.add_argument("--n", type=int, default=int(os.environ.get("NAT_N", 400)))
    ap.add_argument("--seed-start", type=int, default=int(os.environ.get("NAT_SEED_START", 0)))
    ap.add_argument("--model", default=os.environ.get("NATURALIZE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--cache", default=os.environ.get("CACHE", "artifacts/naturalized_passages.json"))
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", DEFAULT_URL))
    args = ap.parse_args()

    complete = make_complete(args.base_url, args.model)
    result = run(args.n, args.seed_start, complete, args.cache)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
