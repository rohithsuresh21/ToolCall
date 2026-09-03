"""Audit a built SFT jsonl for the three defects that the oracle score cannot see.

    python tests/audit_sft.py data/sft.jsonl

Point it at ANY built set -- including one built on a remote GPU box -- to settle
whether it came from the corrected generator. The three axes:

1. PSYCHIC FIRST QUERY. The first search must be writable from the question alone.
   A first query naming a proper noun absent from the prompt means the plan was
   built from the DESTINATION entity of each hop instead of the source, and the
   model is being taught to hallucinate a confident entity name it cannot know.
   This is the pre-4daf979 signature; it was 100% of the old 5850-record build.

2. ANSWER NEVER RETRIEVED. The gold string must appear in the passages the
   trajectory actually got back. When it does not, the record teaches stating an
   answer that was never in context. `eval --backend oracle` scores these 100%
   because MockBackend emits oracle_answer from a lookup table without reading a
   tool result -- which is why this file exists.

3. CONFLICTING QUESTIONS. The same question string under two seeds carries two
   different gold answers, and nothing in the prompt distinguishes them. Those
   pairs are unlearnable; rejection.py drops them via dedupe_by_question.

4. PREFIX LEAKAGE. The gold string appears in the top-k of a call BEFORE the
   terminal read, so the chain can be truncated: a model that stops early is
   still scored right, and that is what it will learn. This is the per-hop form
   of the disconnection shortcut and the single-search filter cannot see it --
   before `_is_prefix_leaky`, 51.9% of generated 4-hop tasks (41.6% at 2 and 3
   hops) were answerable in fewer hops than their label claimed.

Also reported: the answer-diversity of each hop family.

The retrieval checks read the `title` and `text` of each parsed hit, NOT the raw
tool_response string. Substring-matching the whole JSON blob false-positives on
the BM25 `score` float: gold "1949" matches "score": 5.1949 once norm_text
deletes the punctuation, which is how a clean 4-hop set reported one phantom
early hit.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atr.tasks.schema import norm_text  # noqa: E402

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FINAL = re.compile(r"<final_answer>(.*?)</final_answer>", re.S)


def _hit_texts(tool_content: str) -> list[str]:
    """The passage prose a tool_response actually shows the model.

    Only `title` and `text` -- never the raw JSON. doc_ids and the BM25 `score`
    float are in that string too, and norm_text deletes punctuation, so a naive
    substring match reports gold "1949" as retrieved from "score": 5.1949.
    Falls back to the raw string if the payload is not the expected shape, so a
    set built by a different exporter still gets audited rather than skipped."""
    try:
        payload = json.loads(tool_content)
        hits = payload["results"]
    except (ValueError, KeyError, TypeError):
        return [tool_content]
    return [f"{h.get('title', '')} {h.get('text', '')}" for h in hits]


def load(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def audit(path):
    n = 0
    hops = Counter()
    psychic = Counter()
    unretrieved = Counter()
    first_hit = defaultdict(Counter)
    early = Counter()
    answers = defaultdict(Counter)
    by_q: dict[str, set] = defaultdict(set)
    q_count = Counter()
    calls_per_hop = defaultdict(Counter)

    for d in load(path):
        n += 1
        hop = d.get("meta", {}).get("difficulty")
        hops[hop] += 1
        msgs = d["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = [m for m in msgs if m["role"] == "assistant"]
        tool = [m["content"] for m in msgs if m["role"] == "tool"]

        queries = [json.loads(m).get("arguments", {}).get("query", "")
                   for a in asst for m in CALL.findall(a.get("content", ""))]
        calls_per_hop[hop][len(queries)] += 1

        # 1. psychic first query
        if queries:
            caps = [t for t in re.findall(r"[A-Za-z]+", queries[0]) if t[:1].isupper()]
            if any(c.lower() not in user.lower() for c in caps):
                psychic[hop] += 1

        # 2/3. answer retrieval + label conflicts
        fa = FINAL.search(asst[-1].get("content", "")) if asst else None
        ans = fa.group(1).strip() if fa else None
        if ans is None:
            unretrieved[hop] += 1
            continue
        answers[hop][ans] += 1
        q = " ".join(user.split()).lower()
        by_q[q].add(ans)
        q_count[q] += 1
        want = norm_text(ans)
        hit = None
        for i, t in enumerate(tool, start=1):
            if want and any(want in norm_text(x) for x in _hit_texts(t)):
                hit = i
                break
        first_hit[hop][hit] += 1
        if hit is None:
            unretrieved[hop] += 1
        elif hit < len(tool):
            # answerable before the terminal read -> truncatable chain
            early[hop] += 1

    print(f"\n{'='*78}\n{path}   {n} records\n{'='*78}")
    print(f"{'hop':>4} {'n':>6} {'searches':>10} {'psychic 1st q':>16} "
          f"{'answer never retrieved':>24}")
    for h in sorted(hops, key=lambda x: (x is None, x)):
        tot = hops[h]
        ncalls = "/".join(str(k) for k in sorted(calls_per_hop[h]))
        print(f"{str(h):>4} {tot:>6} {ncalls:>10} "
              f"{psychic[h]:>7} {psychic[h]/tot:>7.1%} "
              f"{unretrieved[h]:>13} {unretrieved[h]/tot:>8.1%}")

    print("\nearliest call whose result already contains the gold answer "
          "(None = never; < n_calls means solvable in fewer hops):")
    for h in sorted(first_hit, key=lambda x: (x is None, x)):
        d = dict(sorted(first_hit[h].items(), key=lambda kv: (kv[0] is None, kv[0])))
        print(f"  {h}-hop: {d}")

    dupes = {q: a for q, a in by_q.items() if len(a) > 1}
    affected = sum(q_count[q] for q in dupes)
    print(f"\nquestion-string collisions with CONFLICTING gold answers: "
          f"{len(dupes)}/{len(by_q)} distinct questions")
    print(f"  records affected: {affected}/{n} = {affected/max(n,1):.1%}")
    if dupes:
        worst = max(dupes, key=lambda q: len(dupes[q]))
        print(f"  worst: {len(dupes[worst])} different answers over "
              f"{q_count[worst]} records -- {worst[:88]}")

    print("\nanswer diversity:")
    for h in sorted(answers, key=lambda x: (x is None, x)):
        c = answers[h]
        tot = sum(c.values())
        print(f"  {h}-hop: {len(c)} distinct answers / {tot} records; "
              f"top answer share {c.most_common(1)[0][1]/tot:.1%}")

    clean = (sum(psychic.values()) == 0 and sum(unretrieved.values()) == 0
             and sum(early.values()) == 0 and not dupes)
    print(f"\nVERDICT: {'CLEAN' if clean else 'DEFECTS PRESENT'}")
    return clean


if __name__ == "__main__":
    paths = sys.argv[1:] or ["data/sft.jsonl"]
    ok = all(audit(p) for p in paths)
    sys.exit(0 if ok else 1)
