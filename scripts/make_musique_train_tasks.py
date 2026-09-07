"""Build the GRPO real-task pool from the MuSiQue-Ans TRAIN split, reproducibly.

    python scripts/make_musique_train_tasks.py               # -> data/musique_train_tasks.jsonl

THIS SET MUST NEVER OVERLAP THE JUDGE PROBE. The judge set we score on
(musique_2hop/3hop/4hop) is the 54 public examples in data/judge_tasks.jsonl, a
sample of real MuSiQue; training on any of those 54 (or their exact questions)
is direct benchmark leakage. So before anything is written this script runs the
disjointness check TWICE against data/judge_tasks.jsonl, by two different
fingerprints (question text, evidence-id signature), drops any row that collides
on EITHER, and refuses to write if a collision is found and not dropped. The
names must wind up in the output's commit message / README whenever it is
regenerated.

Rows are sampled deterministically from the full MuSiQue-Ans train split
(bdsaglam/musique, config answerable, split train -- the same repo the judge
set's IDs come from), shuffled with a fixed seed, capped at --max rows. Each
kept row is one ATR task: id, question, answer, hops, task_type, gold, the
example's own 20-passage candidate set as `documents`, and an oracle_plan built
from MuSiQue's own question_decomposition (per-hop sub-questions, in route
order) so GRPO's anchor/progress/reformulate shaping fires on real search
queries too.

Output is sorted by id with sorted JSON keys; two runs on identical input
produce a byte-identical file and the printed sha256 is a real fingerprint
(compare with the judge-set builder, scripts/make_judge_tasks.py).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

HF_DATASET = "bdsaglam/musique"
HF_CONFIG = "answerable"
HF_SPLIT = "train"

DEFAULT_CAP = 4000          # sampled from train (keeps the file cheap to reload)
DEFAULT_JUDGE = "data/judge_tasks.jsonl"
DEFAULT_OUT = "data/musique_train_tasks.jsonl"

# Target hop mix for the sampled real slice. MuSiQue's NATURAL train split is
# ~72/22/6 (2/3/4-hop), which when mixed 20% real into a 40/30/30 synthetic
# batch leaves real 4-hop at ~1.2% of the step -- invisible. Stratify to a
# richer 3/4-hop target instead so the real slice actually exercises the hops
# that fail; the sample never touches the judge probe regardless (see below).
TARGET_HOP_MIX: dict[int, float] = {2: 0.40, 3: 0.30, 4: 0.30}


def _load_hf_rows(limit: int):
    from datasets import load_dataset
    print(f"loading {HF_DATASET} (config={HF_CONFIG}, split={HF_SPLIT}) from the hub")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    return [dict(r) for r in ds]


def _id_number_signature(sample_id: str) -> set[int]:
    """Evidence-id signature: every integer token embedded in a MuSiQue id, e.g.
    '2hop__131611_32392_823060_610794' -> {2, 131611, 32392, 823060, 610794}.
    The per-hop paragraph-support ids are the load-bearing part; the hop prefix
    is included for belt-and-braces (two different hops never share an id, but
    keeping it costs nothing)."""
    toks = [tok for tok in sample_id.replace("__", "_").replace("-", "_").split("_")
            if tok.isdigit()]
    return {int(t) for t in toks}


def _load_judge_fingerprints(path: Path):
    """Read the judge probe ONCE, return the two fingerprints it is later
    checked against twice:
      - question text, normalised (exact-match, strong identity)
      - evidence-id signature sets  (id-shape identity)
    Also keeps the raw rows for reporting."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    qnorm = {_norm_question(r["question"]) for r in rows}
    evids = {frozenset(_id_number_signature(str(r["id"]))) for r in rows}
    return rows, qnorm, evids


def _norm_question(q: str) -> str:
    s = (q or "").lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())


def _check_disjoint(sample_id: str, question: str,
                    judge_qnorm: set[str], judge_evids: set[frozenset]) -> tuple[bool, str]:
    """One disjointness pass. Returns (ok, reason). A row is REJECTED if its
    normalised question matches a judge question OR its evidence-id signature
    intersects a judge signature (a shared supporting paragraph means the same
    underlying example was composed into both splits -- the strongest leak)."""
    qn = _norm_question(question)
    if qn in judge_qnorm:
        return False, f"question text matches a judge probe: {question[:80]}"
    mine = frozenset(_id_number_signature(sample_id))
    for other in judge_evids:
        hit = mine & other
        if hit:
            return False, (f"evidence-id signature {sorted(mine)} intersects "
                           f"judge probe id with {sorted(hit)}")
    return True, ""


def _parse_hops(sample_id: str) -> int:
    m = __import__("re").match(r"^(\d+)hop", str(sample_id))
    return int(m.group(1)) if m else 1


def _build_task(row: dict, idx: int) -> dict:
    hops = _parse_hops(row["id"])
    n_para = len(row["paragraphs"])
    docs = [
        {"doc_id": str(p["idx"]), "title": p["title"], "text": p["paragraph_text"],
         "facts": {}, "links": []}
        for p in row["paragraphs"]
    ]
    plan = [
        {"name": "search", "arguments": {"query": sub["question"]}}
        for sub in (row.get("question_decomposition") or [])
        if (sub.get("question") or "").strip()
    ]
    route = [str(p.get("idx")) for p in row["paragraphs"] if p.get("is_supporting")]
    return {
        "task_id": str(row["id"]),
        "seed": int(hashlib.sha1(str(row["id"]).encode()).hexdigest()[:8], 16),
        "prompt": row["question"],
        "task_type": f"musique_{hops}hop",
        "difficulty": hops,
        "gold": {"kind": "text", "value": str(row["answer"]).strip()},
        "tier": max(0, min(3, hops)) if hops else 0,
        "required_tools": ["search"],
        "required_any": [["search"]],
        "forbidden_tools": [],
        "oracle_plan": plan,
        "oracle_answer": str(row["answer"]).strip(),
        "route": route,
        "expect_side_effect": None,
        "notes": (f"real MuSiQue-Ans train sample {idx}; {hops}-hop; "
                  f"{n_para}-passage candidate set; {len(plan)} oracle searches "
                  f"from MuSiQue question_decomposition"),
        "documents": docs,
    }


def _write_tasks(tasks: list[dict], out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for t in sorted(tasks, key=lambda t: t["task_id"]):
            f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="local jsonl instead of HF hub")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--max", type=int, default=DEFAULT_CAP,
                    help="sampled train rows to keep (0 = all)")
    ap.add_argument("--seed", type=int, default=0xA7B0BAA5)
    args = ap.parse_args()

    judge_path = Path(args.judge)
    if not judge_path.exists():
        raise SystemExit(f"FATAL: judge probe not found at {judge_path}. Run "
                         f"scripts/make_judge_tasks.py first.")
    jrows, jqnorm, jevids = _load_judge_fingerprints(judge_path)
    print(f"judge probe: {len(jrows)} rows (question fingerprints: {len(jqnorm)}, "
          f"evidence-id signatures: {len(jevids)})")

    if args.src:
        with open(args.src, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        rows = _load_hf_rows(args.max)

    print(f"source split: {len(rows)} raw rows")
    n_real = args.max if args.max else len(rows)

    # Stratify per-hop to TARGET_HOP_MIX. Whole-split counts for the stratum
    # pools (the prer-mix gives the names about the natural corpus); the draw is
    # per-hop random without replacement, so the output is reproducible.
    by_hop: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_hop[_parse_hops(row["id"])].append(row)
    print(f"source split hop counts: "
          f"{dict(sorted((h, len(v)) for h, v in by_hop.items()))}")

    rng = random.Random(args.seed)
    cand: list[dict] = []
    for hops, weight in sorted(TARGET_HOP_MIX.items()):
        pool = by_hop.get(hops, [])
        if not pool:
            continue
        want = int(round(n_real * weight))
        rng.shuffle(pool)
        cand.extend(pool[:want])
        print(f"  {hops}-hop: drew {min(want, len(pool))}/{len(pool)}")
    if args.max and len(cand) > args.max:
        cand = cand[:args.max]
    print(f"sampled {len(cand)} rows (seed={args.seed}, max={args.max})")

    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for i in range(len(cand)):
        row = cand[i]
        ok1, why1 = _check_disjoint(str(row["id"]), row["question"], jqnorm, jevids)
        ok2, why2 = _check_disjoint(str(row["id"]), row["question"], jqnorm, jevids)
        if not (ok1 and ok2):
            dropped.append((str(row["id"]), why1 or why2))
            continue
        kept.append(_build_task(row, i))

    print(f"kept {len(kept)} / dropped {len(dropped)}")

    if dropped:
        print("\nDROPPED (must be zero + zero both passes):")
        for sid, why in dropped:
            print(f"  {sid}: {why}")

    # Second verification pass, from the FULL judge file re-read independently.
    # Even if a row were never dropped, its question or evidence signature must
    # not appear in the probe. We re-load the judge file (not the cached copy)
    # and re-check everything that is about to be WRITTEN.
    jrows2, jqnorm2, jevids2 = _load_judge_fingerprints(judge_path)
    leftovers = [(t["task_id"], _check_disjoint(t["task_id"], t["prompt"], jqnorm2, jevids2))
                 for t in kept]
    bad = [t for t in leftovers if not t[1][0]]
    if bad:
        for sid, (_, why) in bad[:20]:
            print(f"  STILL COLLIDES after drop: {sid}: {why}")
        raise SystemExit(f"FATAL: {len(bad)} rows still overlap the judge probe "
                         f"on second pass -- refusing to write {args.out}.")
    if not kept:
        raise SystemExit("FATAL: empty task set after disjointness filtering.")

    digest = _write_tasks(kept, Path(args.out))
    hop_mix = dict(sorted(collections.Counter(t["task_type"] for t in kept).items()))
    n_pass = collections.Counter(len(t["documents"]) for t in kept)
    print(f"wrote {len(kept)} tasks -> {args.out}")
    print(f"  hop mix        : {hop_mix}")
    print(f"  passages/task  : {dict(sorted(n_pass.items()))}")
    print(f"  oracle plans   : {dict(sorted(collections.Counter(len(t['oracle_plan']) for t in kept).items()))}")
    print(f"  sha256         : {digest}")
    print("\nOK: double-disjoint from the 54-example judge probe (both passes, "
          "both fingerprints). Commit the hash + row gas + dropped count if you "
          "bump these numbers in a README/notes.")


if __name__ == "__main__":
    main()