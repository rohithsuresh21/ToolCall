"""Build the judge task set from the PUBLIC HF dataset, reproducibly.

    python scripts/make_judge_tasks.py                    # -> data/judge_tasks.jsonl
    python scripts/make_judge_tasks.py --src local.parquet

Default source is the Hugging Face hub:

    dataset  YashBhamare123/tools-benchmark
    config   default
    split    public        (54 examples)

This used to read a parquet from one developer's `E:\\...` drive, which made the
task set a remote-only artifact: `artifacts/judge_tasks.jsonl` existed on the GPU
box and nowhere else, so no judge number could be reproduced or even checked for
row count from a clean checkout. Pulling from the hub by name makes the input
addressable; `--src` still accepts a local parquet for an offline box.

Output is written SORTED BY id with sorted JSON keys, so two runs on the same
split produce byte-identical files and the printed sha256 is a real fingerprint
you can compare across machines. It lands in `data/` rather than `artifacts/`
on purpose -- `artifacts/` is gitignored, and a gitignored input shadowing a
committed one is exactly how the leaky SFT set got trained on.

Each row becomes one ATR task: id, question, answer, hops, task_type
(musique_{hops}hop), gold, and the example's own 20-passage context as the
searchable candidate set. Nothing here is filtered or resampled -- the judge set
is a probe, and quietly dropping rows from it would flatter the score.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

HF_DATASET = "YashBhamare123/tools-benchmark"
HF_CONFIG = "default"
HF_SPLIT = "public"
EXPECTED_ROWS = 54


def _rows_from_hub() -> list[dict]:
    from datasets import load_dataset
    print(f"loading {HF_DATASET} (config={HF_CONFIG}, split={HF_SPLIT}) from the hub")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    return [dict(r) for r in ds]


def _rows_from_parquet(src: Path) -> list[dict]:
    import pandas as pd
    print(f"loading local parquet {src}")
    return list(pd.read_parquet(src).to_dict(orient="records"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="local parquet to use instead of the HF hub (offline boxes)")
    ap.add_argument("--out", default="data/judge_tasks.jsonl",
                    help="where to write the ATR tasks jsonl")
    ap.add_argument("--expect", type=int, default=EXPECTED_ROWS,
                    help="expected row count; 0 disables the check")
    args = ap.parse_args()

    rows = _rows_from_parquet(Path(args.src)) if args.src else _rows_from_hub()
    print(f"got {len(rows)} rows; columns: {sorted(rows[0]) if rows else '(none)'}")

    tasks = []
    for row in rows:
        hops = int(row["metadata"]["hops"])
        context = [{"title": p["title"], "paragraph_text": p["paragraph_text"]}
                   for p in row["context"]]
        tasks.append({
            "id": str(row["id"]),
            "question": row["question"],
            "answer": row["answer"],
            "hops": hops,
            "task_type": f"musique_{hops}hop",
            "gold": {"kind": "text", "value": row["answer"]},
            "context": context,
        })
    tasks.sort(key=lambda t: t["id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    hop_mix = dict(sorted(collections.Counter(t["task_type"] for t in tasks).items()))
    n_ctx = collections.Counter(len(t["context"]) for t in tasks)
    print(f"\nwrote {len(tasks)} tasks -> {out}")
    print(f"  hop mix        : {hop_mix}")
    print(f"  passages/task  : {dict(sorted(n_ctx.items()))}")
    print(f"  sha256         : {digest}")

    if not tasks:
        raise SystemExit("FATAL: no rows written.")
    if args.expect and len(tasks) != args.expect:
        raise SystemExit(
            f"FATAL: expected {args.expect} rows, got {len(tasks)}. The public split "
            f"changed upstream -- confirm before scoring against it, or pass "
            f"--expect {len(tasks)} once you have.")
    if len({t["id"] for t in tasks}) != len(tasks):
        raise SystemExit("FATAL: duplicate ids in the judge set.")
    blank = [t["id"] for t in tasks if not str(t["answer"]).strip()]
    if blank:
        raise SystemExit(f"FATAL: {len(blank)} rows have an empty gold answer: {blank[:5]}")
    print("\nOK: row count, ids and gold answers all check out.")


if __name__ == "__main__":
    main()
