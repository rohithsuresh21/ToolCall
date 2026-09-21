"""Build the TEACHER-REASONING arm: data/sft_r0.jsonl with the ~33% of <think>
blocks that face a same-kind distractor rewritten by a larger model.

    python scripts/14_build_teacher.py --legacy-vocab              # mint + build
    python scripts/14_build_teacher.py --legacy-vocab --no-mint    # build from cache only

WHY A TRANSFORM, not a rebuild -- the same argument as 12_build_recovery.py and
13_rebalance_4hop.py. The family is drawn per index from one `random.Random`
stream and the filters re-evaluate on top of that, so a fresh `generate()` moves
the population underneath the comparison. Every record this script does not
convert is written back BYTE-IDENTICALLY, from its original line rather than a
re-serialisation, so `sft_r0` vs `sft_r0_teacher` isolates the reasoning text and
nothing else. Record count, hop mix, task ids, questions, queries and tool
responses are all unchanged; only the prose inside selected <think> tags moves.

WHAT --legacy-vocab DOES HERE, AND WHAT IT DOES NOT. In the other two arms it is
functional: they execute new queries against the world behind a record, and under
the widened pools `build_world(seed)` returns a different world, so the
reconstruction fails and the arm silently degrades to a copy of its base. This
script executes nothing -- selection, prompting and gating all read the stored
messages -- so it does not NEED the world. The flag is kept as a VERIFICATION
gate: it rebuilds each candidate's Task and checks the prompt matches, which
proves the base is the population the cache was minted against. Same flag, same
requirement on a pre-widening base, weaker consequence if omitted -- so it is
checked and reported rather than assumed, and the run refuses when most
candidates fail.

THE CACHE IS THE DETERMINISM. An LLM cannot hold reasoning.py's
sha1(task_id|step) contract, so the minted text is committed and the build reads
it. The cache records the base's fingerprint and this script refuses a mismatch:
the keys are task ids and they resolve against any set built from the same seeds,
so a stale cache would load in silence and paste another population's
deliberation into this one. That is the same hazard CLAUDE.md records for a
naturalization cache minted before the unanchored templates.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.data.reasoning_teacher import (Cache, TeacherConfig, apply_block, fingerprint,  # noqa: E402
                              rebuild_ok, select, walk)
from atr.data.reasoning_mint import mint, report  # noqa: E402
from atr.tools.legacy_vocab import legacy_vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_r0.jsonl")
    ap.add_argument("--out", default="data/sft_r0_teacher.jsonl")
    ap.add_argument("--cache", default="data/teacher_cache_r0.json")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--min-distractors", type=int, default=1)
    ap.add_argument("--legacy-vocab", action="store_true",
                    help="verify the Task reconstruction under the pre-widening "
                         "name pools; required for any base built before 566f4d2")
    ap.add_argument("--no-mint", action="store_true",
                    help="build from the committed cache only; never call the API")
    ap.add_argument("--dry-run", action="store_true",
                    help="select and price, write nothing, call nothing")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="mint N blocks synchronously, print them, write nothing. "
                         "Validates the request shape before a batch is submitted, "
                         "and puts real prose on screen for the one check the gates "
                         "cannot make (an invented LOWERCASE claim).")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    cfg = TeacherConfig(model=args.model, effort=args.effort,
                        min_distractors=args.min_distractors)

    lines = [l for l in Path(args.base).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    fp = fingerprint(args.base)
    by_hop = collections.Counter(r["meta"]["difficulty"] for r in rows)
    print(f"[base] {args.base}: {len(rows)} records, hop mix {dict(sorted(by_hop.items()))}, "
          f"fingerprint {fp}")

    # --- select ------------------------------------------------------------
    jobs, sel_by_hop, blocks = [], collections.Counter(), 0
    for r, line in zip(rows, lines):
        msgs = r["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        blocks += sum(1 for _ in walk(msgs))
        for b in select(msgs, cfg):
            sel_by_hop[r["meta"]["difficulty"]] += 1
            jobs.append({"task_id": r["meta"]["task_id"], "step": b["step"],
                         "question": question, "seen_norm": b["seen_norm"],
                         "messages": msgs, "block": b, "row": r})
    print(f"[select] {len(jobs)} of {blocks} blocks = {len(jobs) / blocks:.1%}  "
          f"(>= {cfg.min_distractors} same-kind distractor alongside the rank-1 hit)")
    for h in sorted(sel_by_hop):
        print(f"           {h}-hop: {sel_by_hop[h]}")

    # --- verify the base is the population we think it is ------------------
    ctx = legacy_vocab() if args.legacy_vocab else contextlib.nullcontext()
    if args.legacy_vocab:
        print("[vocab] pre-widening name pools (24x20 people, 14x8 orgs)")
    seen_ids, ok, bad = set(), 0, 0
    with ctx:
        for j in jobs:
            if j["task_id"] in seen_ids:
                continue
            seen_ids.add(j["task_id"])
            r = j["row"]
            if rebuild_ok(j["task_id"], r["meta"]["difficulty"], j["question"]):
                ok += 1
            else:
                bad += 1
    print(f"[verify] Task reconstruction: {ok} ok, {bad} failed "
          f"of {ok + bad} distinct task ids")
    if bad > ok:
        raise SystemExit(
            "more than half the records do not reconstruct from their task_id. "
            "For a base built before commit 566f4d2 pass --legacy-vocab; otherwise "
            "the base and the installed world generator disagree.")

    if args.dry_run:
        print("\n[dry-run] nothing minted, nothing written")
        return

    if args.probe:
        # Spread the sample across hop families and positions rather than taking
        # the first N, which would all be 2-hop step 1 from the lowest seeds.
        step = max(1, len(jobs) // args.probe)
        sample = jobs[::step][:args.probe]
        from atr.data.reasoning_mint import probe as _probe
        print(f"\n[probe] minting {len(sample)} blocks synchronously with "
              f"{cfg.model} (effort={cfg.effort})")
        got = _probe(sample, cfg, n=len(sample))
        for k, g in enumerate(got, 1):
            print(f"\n{'=' * 78}\n[{k}] {g['task_id']}  step {g['step']}"
                  f"{'  *** ' + g['violations'][0] if g['violations'] else ''}")
            print(f"  Q       {g['question']}")
            print(f"  query   {g['query']}")
            print(f"  BEFORE  {g['template']}")
            print(f"  AFTER   {g['teacher']}")
        bad = sum(1 for g in got if g["violations"])
        ti = sum(g["in_tok"] for g in got)
        to = sum(g["out_tok"] for g in got)
        print(f"\n[probe] {len(got)} blocks, {bad} would be rejected by the gate")
        print(f"[probe] {ti} input + {to} output tokens "
              f"(${(ti * 5 + to * 25) / 1e6:.4f} at Opus 5 sync rates)")
        print(f"[probe] extrapolated to {len(jobs)} blocks: "
              f"${(ti / len(got) * 5 + to / len(got) * 25) * len(jobs) / 1e6 / 2:.2f} "
              f"batched, before repair passes")
        print("\n[probe] nothing minted to cache, nothing written")
        return

    # --- mint --------------------------------------------------------------
    cache = Cache.load(args.cache)
    if cache.blocks and cache.base_fingerprint and cache.base_fingerprint != fp:
        raise SystemExit(
            f"cache {args.cache} was minted against base {cache.base_fingerprint}, "
            f"but {args.base} fingerprints {fp}. The task ids would still resolve, so "
            f"this would silently paste another population's reasoning into this set. "
            f"Mint a new cache or point --base at the set it came from.")
    if cache.blocks and cache.model and cache.model != cfg.model:
        print(f"[cache] NOTE existing blocks were minted with {cache.model}, "
              f"now running {cfg.model}; cached blocks are kept as-is",
              file=sys.stderr)
    cache.base_fingerprint, cache.model = fp, cfg.model

    stats = mint(jobs, cache, cfg, poll=args.poll, dry_run=args.no_mint)
    report(stats, cache)
    if not args.no_mint:
        cache.save()
        print(f"[cache] wrote {args.cache} ({len(cache.blocks)} blocks)")

    # --- apply -------------------------------------------------------------
    converted_rows, converted_blocks = set(), 0
    for j in jobs:
        hit = cache.get(j["task_id"], j["step"])
        if hit is None:
            continue
        r = j["row"]
        i = j["block"]["msg_index"]
        r["messages"][i]["content"] = apply_block(r["messages"][i]["content"], hit["think"])
        converted_rows.add(id(r))
        converted_blocks += 1

    out_lines = []
    for r, line in zip(rows, lines):
        out_lines.append(json.dumps(r, ensure_ascii=False) if id(r) in converted_rows
                         else line)
    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    verbatim = len(rows) - len(converted_rows)
    print(f"\nteacher blocks applied: {converted_blocks}/{len(jobs)} "
          f"({converted_blocks / len(jobs):.1%}); "
          f"{len(jobs) - converted_blocks} kept their template")
    print(f"records touched {len(converted_rows)}/{len(rows)}, "
          f"{verbatim} lines verbatim")
    print(f"wrote {args.out}")

    if not args.no_audit:
        print(f"\n=== auditing {args.out} ===")
        rc = subprocess.call([sys.executable, "tests/audit_sft.py", args.out], cwd=REPO)
        if rc != 0:
            raise SystemExit(f"DEFECTS PRESENT in {args.out} -- do not train on it.")


if __name__ == "__main__":
    main()
