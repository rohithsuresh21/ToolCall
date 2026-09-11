"""Resolve the real MuSiQue train pool into executable, leak-free oracle chains.

    python scripts/make_musique_chain_tasks.py     # -> data/musique_chain_tasks.jsonl

Input is data/musique_train_tasks.jsonl, which is already double-disjoint from the
54-row judge probe (scripts/make_musique_train_tasks.py enforces that, twice, by two
fingerprints). This script does not resample and does not touch the hub, so that
disjointness carries over unchanged -- it only REWRITES each row's oracle_plan,
turning MuSiQue's `#N` placeholders into strings a BM25 search can actually take,
and drops the rows where no honest resolution exists.

See atr/data/real_chains.py for what "honest" means here and what each gate costs.
Output is sorted by task_id with sorted JSON keys, so two runs are byte-identical
and the printed sha256 is a real fingerprint.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.data.real_chains import (ChainConfig, build_chain_tasks,  # noqa: E402
                                  load_rows, verify_index, write_tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/musique_train_tasks.jsonl")
    ap.add_argument("--out", default="data/musique_chain_tasks.jsonl")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--subject-rule", choices=["subset", "overlap"], default="subset")
    ap.add_argument("--allow-leaks", action="store_true",
                    help="keep chains whose gold answer appears before the final hop")
    ap.add_argument("--allow-psychic-first-query", action="store_true")
    ap.add_argument("--allow-psychic-later-queries", action="store_true",
                    help="keep chains whose hop-2+ query names an entity neither the "
                         "question nor any earlier hit contains")
    ap.add_argument("--skip-index-check", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.src)
    print(f"source: {len(rows)} real tasks from {args.src}")

    if not args.skip_index_check:
        good, checked = verify_index(rows)
        print(f"index fidelity: {good}/{checked} queries rank identically to builtin._search")
        if good != checked:
            raise SystemExit("FATAL: the resolution index does not match the shipped BM25. "
                             "Every yield number below would be a statement about a "
                             "re-implementation, not about the data.")

    cfg = ChainConfig(top_k=args.top_k, subject_rule=args.subject_rule,
                      require_leak_free=not args.allow_leaks,
                      require_writable_first_query=not args.allow_psychic_first_query,
                      require_writable_later_queries=not args.allow_psychic_later_queries)
    kept, stats = build_chain_tasks(rows, cfg)
    if not kept:
        raise SystemExit("FATAL: every row was rejected -- refusing to write an empty pool.")

    digest = hashlib.sha256(write_tasks(kept, args.out).read_bytes()).hexdigest()
    hop_mix = dict(sorted(collections.Counter(t["task_type"] for t in kept).items()))
    plan_len = dict(sorted(collections.Counter(len(t["oracle_plan"]) for t in kept).items()))
    subs = dict(sorted(collections.Counter(
        sum(1 for c in t["chain"] if c) for t in kept).items()))

    print(f"\nwrote {len(kept)}/{len(rows)} tasks ({len(kept)/len(rows):.1%}) -> {args.out}")
    print(f"  hop mix          : {hop_mix}")
    print(f"  searches per plan: {plan_len}")
    print(f"  resolved names   : {subs}   (chain entries a <think> block can name)")
    print(f"  sha256           : {digest}")
    print("\nrejections (one key per axis -- a single total cannot tell an "
          "unresolvable placeholder from a leaky chain):")
    for k, v in stats.items():
        if k.startswith("reject:"):
            print(f"  {k[7:]:<34} {v:>5}  {v/len(rows):>6.1%}")


if __name__ == "__main__":
    main()
