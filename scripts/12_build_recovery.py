"""Build the RECOVERY arm: data/sft_r0_max.jsonl with ~10% of its trajectories
rewritten as STeCa-style recover-from-a-bad-retrieval episodes.

    python scripts/12_build_recovery.py --out data/sft_r0_max_rec.jsonl

WHY THIS TRANSFORMS AN EXISTING SET RATHER THAN BUILDING A NEW ONE. The point of
the arm is to isolate ONE variable, and a fresh `generate()` run cannot do that:
the record population would move underneath the comparison even with the same
seeds and mix, because the filters re-evaluate. So the base set is read back and
the ~90% of records that are not converted stay BYTE-IDENTICAL, while the
converted ones keep every message they had and gain exactly two: the executed bad
call's tool_response, and the repaired assistant turn. Record count, hop mix,
task ids and questions are unchanged.

The Task is reconstructed from the record's own `task_id`, which carries the seed,
and the world is a pure function of the seed -- so the bad query is executed
against exactly the corpus that episode ran on. The reconstruction is VERIFIED per
record (prompt must match the exported user turn) rather than assumed; a mismatch
is counted and the record is left alone, which is the only safe failure.

Rates are per hop family and deliberately uneven (see RecoveryConfig): the looping
this is meant to fix was measured at 4 hops, and 2-hop is already at 80.0% judge
F1. A record that admits no genuine miss at any hop is skipped and the next one in
the deterministic order takes its place, so the requested count is met exactly
whenever supply allows.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.data.recovery import RecoveryConfig, build_recovery  # noqa: E402
from atr.tasks.generator import _hop_or_shorter  # noqa: E402
from atr.tools.world import build_world  # noqa: E402


def _rebuild(task_id: str, difficulty: int, prompt: str):
    """Reconstruct (Task, World) from the record. `_hop_or_shorter` can degrade a
    longer request to a shorter chain, and it consumes from the generator's rng on
    the way, so the requested hop count is not always the recorded difficulty --
    try each one that could have produced this record and keep the one whose prompt
    matches exactly."""
    seed = int(task_id.split("-")[1])
    w = build_world(seed)
    for h in range(difficulty, 5):
        t = _hop_or_shorter(random.Random(seed * 7919 + 13), w, seed, h)
        if t.task_id == task_id and t.difficulty == difficulty and t.prompt == prompt:
            return t, w
    return None, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_r0_max.jsonl")
    ap.add_argument("--out", default="data/sft_r0_max_rec.jsonl")
    ap.add_argument("--rates", default="2:0.05,3:0.12,4:0.15",
                    help="share of each hop family converted to a recovery trajectory")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    cfg = RecoveryConfig(rate_by_hop={int(k): float(v) for k, v in
                                      (p.split(":") for p in args.rates.split(","))})

    rows = [json.loads(l) for l in
            Path(args.base).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_hop = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_hop[r["meta"]["difficulty"]].append(i)
    print(f"[base] {args.base}: {len(rows)} records, hop mix "
          f"{dict(sorted((h, len(v)) for h, v in by_hop.items()))}")

    flavours, at_hop, empties = collections.Counter(), collections.Counter(), 0
    n_recov = 0
    failed_rebuild, no_miss = 0, 0
    for hop in sorted(by_hop):
        idxs = by_hop[hop]
        want = int(round(cfg.rate_by_hop.get(hop, 0.0) * len(idxs)))
        # Deterministic candidate order: a hash of the task id, so the slice is
        # reproducible and is not biased toward low seeds.
        order = sorted(idxs, key=lambda i: hashlib.sha1(
            rows[i]["meta"]["task_id"].encode()).hexdigest())
        made = 0
        for i in order:
            if made >= want:
                break
            rec = rows[i]
            prompt = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            task, world = _rebuild(rec["meta"]["task_id"], hop, prompt)
            if task is None:
                failed_rebuild += 1
                continue
            out = build_recovery(task, rec["messages"], cfg, world=world)
            if out is None:
                no_miss += 1
                continue
            rec["messages"] = out["messages"]
            rec["meta"]["num_calls"] = rec["meta"].get("num_calls", 0) + 1
            rec["meta"]["recovery"] = {"hop": out["hop"], "flavour": out["flavour"],
                                       "num_results": out["num_results"],
                                       "query": out["query"]}
            flavours[out["flavour"]] += 1
            at_hop[(hop, out["hop"])] += 1
            empties += out["num_results"] == 0
            made += 1
            n_recov += 1
        print(f"[{hop}-hop] {made}/{want} converted from {len(idxs)} records "
              f"({made / len(idxs):.1%})")

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    print(f"\nrecovery trajectories: {n_recov}/{len(rows)} = {n_recov / len(rows):.1%}")
    print(f"  rebuild failures {failed_rebuild}   no genuine miss at any hop {no_miss}")
    print(f"  bad result was EMPTY  {empties}/{n_recov} = {empties / max(n_recov, 1):.1%}"
          f"   (the rest returned irrelevant passages)")
    print("  flavour (proposed by recovery._FLAVOURS, kept only if it really missed):")
    for k, v in flavours.most_common():
        print(f"    {k:<16} {v:>5}  {v / max(n_recov, 1):>6.1%}")
    for k in sorted(set(_ for _ in ("wrong_entity", "absent_entity", "conversational",
                                    "wrong_keyword")) - set(flavours)):
        print(f"    {k:<16} {0:>5}   0.0%  (never survived the genuine-miss check)")
    print("  injected at plan position (hop family, 0-based call index):")
    for k in sorted(at_hop):
        print(f"    {k[0]}-hop call {k[1]}: {at_hop[k]}")
    print(f"\nwrote {args.out}")

    if not args.no_audit:
        print(f"\n=== auditing {args.out} ===")
        rc = subprocess.call([sys.executable, "tests/audit_sft.py", args.out], cwd=REPO)
        if rc != 0:
            raise SystemExit(f"DEFECTS PRESENT in {args.out} -- do not train on it.")


if __name__ == "__main__":
    main()
