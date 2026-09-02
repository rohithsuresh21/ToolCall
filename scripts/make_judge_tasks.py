"""Convert the judge's public benchmark split (parquet) into ATR tasks jsonl.

Each public example becomes one line with: id, question, answer, hops,
task_type (musique_{hops}hop), gold, and the example's own 20-passage context
as the searchable candidate set. Run on a machine with pandas/pyarrow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    src = Path(r"E:\Multi modal reasoning tool\Benchmark\tools-benchmark\data\public-00000-of-00001.parquet")
    out = Path(r"E:\Multi modal reasoning tool\atr\artifacts\judge_tasks.jsonl")
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
    import collections
    print(dict(collections.Counter(d["task_type"] for d in
                                   [json.loads(l) for l in out.read_text().splitlines()])))


if __name__ == "__main__":
    main()