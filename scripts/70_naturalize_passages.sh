#!/usr/bin/env bash
# Part A: LLM passage naturalization (offline only).
#
# Mints a corpus of worlds, rewrites each passage's prose via an LLM while
# forcing fact-preservation (atr/data/naturalize.py), and writes a cache keyed by
# "seed:doc_id" to an artifact file. This NEVER runs inside build_world() or at
# train/eval time -- the cache is produced here, once, then loaded opt-in via
# build_world(seed, text_loader=load_naturalized_loader(CACHE)).
#
# Env:
#   NATURALIZE_MODEL    OpenAI-compatible model id (default gpt-4.1-mini)
#   NATURALIZE_BASE_URL provider base URL (default $OPENAI_BASE_URL)
#   NATURALIZE_SEEDS    comma/space list of seed ranges as "start:count" (def 0:120)
#   CACHE               output cache path (default artifacts/naturalized_passages.json)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${NATURALIZE_MODEL:-gpt-4.1-mini}"
BASE_URL="${NATURALIZE_BASE_URL:-${OPENAI_BASE_URL:-}}"
RANGES="${NATURALIZE_SEEDS:-0:120}"
CACHE="${CACHE:-artifacts/naturalized_passages.json}"

mkdir -p artifacts
MODEL="$MODEL" BASE_URL="$BASE_URL" RANGES="$RANGES" CACHE="$CACHE" python3 - <<'PY'
import json
import os

from atr.data.naturalize import naturalize_passages
from atr.tools.world import build_world

model = os.environ["MODEL"]
base_url = os.environ.get("BASE_URL") or None
ranges = os.environ["RANGES"].replace(",", " ").split()
cache = os.environ["CACHE"]


def _client():
    from openai import OpenAI
    c = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    def complete(prompt: str) -> str:
        r = c.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=220)
        return r.choices[0].message.content or ""
    return complete


def main():
    llm = _client()
    out: dict[str, str] = {}
    stats = {"checked": 0, "naturalized": 0, "fell_back": 0}
    for rng in ranges:
        start, _, count = rng.partition(":")
        for seed in range(int(start), int(start) + int(count or 1)):
            w = build_world(seed)
            before = {d["doc_id"]: d["text"] for d in w.documents}
            s = naturalize_passages(w, llm)
            for d in w.documents:
                if d["text"] != before[d["doc_id"]]:
                    out[f"{seed}:{d['doc_id']}"] = d["text"]
            stats["checked"] += s["docs"]
            stats["naturalized"] += s["naturalized"]
            stats["fell_back"] += s["fell_back"]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({**stats, "cache": cache, "entries": len(out)}, indent=2))


if __name__ == "__main__":
    main()
PY
