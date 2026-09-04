"""Convert the judge's public benchmark split (parquet) into ATR tasks jsonl.

Each public example becomes one line with: id, question, answer, hops,
task_type (musique_{hops}hop), gold, and the example's own 20-passage context
as the searchable candidate set. Run on a machine with pandas/pyarrow.

The judge benchmark parquet lives on the DEV/LOCAL machine (E:\\... public-*.parquet),
so run THIS script locally to emit judge_tasks.jsonl, then push that file to the
remote node (where the trained adapter + GPU sit) and run scripts/judge_eval.py there.

Usage:
  python scripts/make_judge_tasks.py [--src <parquet>] [--out <jsonl>]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="path to the judge public-*.parquet (default: the known local path)")
    ap.add_argument("--out", default="artifacts/judge_tasks.jsonl",
                    help="where to write the ATR tasks jsonl")
    args = ap.parse_args()

    src = (Path(args.src) if args.src else
           Path(r"E:\Multi modal reasoning tool\Benchmark\tools-benchmark\data\public-00000-of-00001.parquet"))
    out = Path(args.out)
    df = pd.read_parquet(src)
    print("columns:", list(df.columns), "rows:", len(df))
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            hops = int(row["metadata"]["hops"])
            context = [{"title": p["title"], "paragraph_text": p["paragraph_text"]}
                       for p in row["context"]]
            rec = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "hops": hops,
                "task_type": f"musique_{hops}hop",
                "gold": {"kind": "text", "value": row["answer"]},
                "context": context,
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} tasks -> {out}")
    print(dict(collections.Counter(d["task_type"] for d in
                                   [json.loads(l) for l in out.read_text().splitlines()])))


if __name__ == "__main__":
    main()