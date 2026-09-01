"""Free, local naturalization via Ollama (OpenAI-compatible endpoint, no API key).

Runs the same atr.data.naturalize pipeline against a local Ollama server so the
LLM passage rewrite costs nothing and stays on-machine.

Usage:
    python -m atr.data.naturalize_local --seeds 0:2 [--model qwen2.5:7b]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable

from openai import OpenAI

from .naturalize import naturalize_passages
from ..tools.world import build_world, World

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("NATURALIZE_MODEL", "qwen2.5-coder:7b")


def make_complete(base_url: str, model: str) -> Callable[[str], str]:
    """Return a callable that sends one rewrite prompt and returns the model's text."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def complete(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=260,
        )
        return (r.choices[0].message.content or "").strip()

    return complete


def parse_seed_ranges(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.replace(",", " ").split():
        start, _, count = part.partition(":")
        out.append((int(start), int(count or 1)))
    return out


def run(seeds: list[tuple[int, int]], complete: Callable[[str], str], cache: str) -> dict:
    """Naturalize worlds for the given seed ranges with the given LLM client and
    write accepted rewrites to `cache` (keyed "seed:doc_id"). Returns stats."""
    loaded: dict[str, str] = {}
    total = {"docs": 0, "naturalized": 0, "retries": 0, "fell_back": 0}
    for start, count in seeds:
        for seed in range(start, start + count):
            w: World = build_world(seed)
            before = {d["doc_id"]: d["text"] for d in w.documents}
            s = naturalize_passages(w, complete)
            for d in w.documents:
                if d["text"] != before[d["doc_id"]]:
                    loaded[f"{seed}:{d['doc_id']}"] = d["text"]
            total["docs"] += s["docs"]
            total["naturalized"] += s["naturalized"]
            total["retries"] += s.get("retries", 0) or 0
            total["fell_back"] += s["fell_back"]
    if cache:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(loaded, f, ensure_ascii=False, indent=2)
    return {**total, "cache_entries": len(loaded), "cache": cache}


def main() -> None:
    ap = argparse.ArgumentParser(description="Free local naturalization via Ollama")
    ap.add_argument("--seeds", default=os.environ.get("NATURALIZE_SEEDS", "0:2"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=os.environ.get("CACHE", "artifacts/naturalized_passages.json"))
    ap.add_argument("--base-url", default="", help="OpenAI-compatible base URL (default: Ollama)")
    args = ap.parse_args()

    base_url = args.base_url or DEFAULT_URL
    complete = make_complete(base_url, args.model)
    result = run(parse_seed_ranges(args.seeds), complete, args.cache)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
