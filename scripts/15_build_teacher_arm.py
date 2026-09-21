"""Build data/sft_r0_teacher.jsonl from a minted cache. Calls nothing, mints nothing.

    python scripts/15_build_teacher_arm.py --legacy-vocab
    python scripts/15_build_teacher_arm.py --cache data/teacher_cache_r0.json --legacy-vocab

The cache-only half of the teacher arm, split out of
scripts/14_build_teacher_reasoning.py so the machine that BUILDS never needs the
machine that MINTS: no `anthropic`, no key, no torch, no GPU, stdlib plus this
repo. 14_ still mints-and-builds through the Batch API and is the right entry
point when that is what you are doing; this one is what runs after
scripts/14_local_teacher.py has filled a cache overnight on a laptop.

It reads BOTH cache formats and does not care which produced it -- the JSONL the
local minter appends, and the single-object JSON the batch minter writes -- because
the keys, the gate and the selector are the same on both paths. The format is a
crash-safety decision, not a semantic one.

WHY A TRANSFORM, not a rebuild -- the same argument as 12_build_recovery.py and
13_rebalance_4hop.py. The family is drawn per index from one `random.Random` stream
and the filters re-evaluate on top of that, so a fresh `generate()` moves the
population underneath the comparison. Every record this script does not convert is
written back BYTE-IDENTICALLY, from its original line rather than a
re-serialisation, so `sft_r0` vs `sft_r0_teacher` isolates the reasoning text and
nothing else. Record count, hop mix, task ids, questions, queries and tool
responses are all unchanged; only the prose inside selected <think> tags moves.

THE CACHE IS THE DETERMINISM, and a stale one is the hazard. An LLM cannot hold
reasoning.py's sha1(task_id|step) contract, so the minted text is committed and the
build reads it. Cache keys are task ids, and they resolve against ANY set built
from the same seeds -- so a cache minted against another population would load in
total silence and paste another set's deliberation into this one. Both formats
carry the base's fingerprint and this script refuses a mismatch. Same hazard
CLAUDE.md records for a naturalization cache minted before the unanchored
templates.

A NULL BLOCK IS NOT A MISSING BLOCK. The local minter records a block that failed
the gate on every attempt as `think: null`, so it is not re-minted every night.
Here that reads as "keep the template" -- identical to what the batch path does by
simply having no entry -- and both are COUNTED and printed, because an arm that
silently degraded into a copy of its base is the failure mode 12_build_recovery.py
was bitten by.

WHAT --legacy-vocab DOES HERE, AND WHAT IT DOES NOT. In the recovery and 4-hop arms
it is functional: they execute new queries against the world behind a record, and
under the widened pools `build_world(seed)` returns a different world, so the
reconstruction fails and the arm silently degrades. This script executes nothing --
selection, splicing and auditing all read the stored messages -- so it does not
NEED the world. The flag is kept as a VERIFICATION gate: it rebuilds each
candidate's Task and checks the prompt matches, which proves the base is the
population the cache was minted against, independently of the fingerprint. Same
flag, same requirement on a pre-widening base, weaker consequence if omitted.
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

from atr.data.reasoning_teacher import (TeacherConfig, apply_block, cache_key,  # noqa: E402
                              fingerprint, rebuild_ok, select, walk)
from atr.tools.legacy_vocab import legacy_vocab  # noqa: E402

CACHE_KIND = "teacher-local-v1"


def load_any_cache(path: Path) -> tuple[dict, dict]:
    """(header, {key: record}) from either minting path.

    JSONL is the local minter's append-only log; a single JSON object is
    `reasoning_teacher.Cache`. Both key on `task_id|step` and both carry the base
    fingerprint, so everything downstream of here is format-blind."""
    if not path.exists():
        raise SystemExit(f"no cache at {path} -- mint one first with "
                         f"scripts/14_local_teacher.py (local) or "
                         f"scripts/14_build_teacher_reasoning.py (batch).")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        d = json.loads(text)
        header = {"base_fingerprint": d.get("base_fingerprint", ""),
                  "model": d.get("model", ""), "format": "batch json"}
        return header, {k: dict(v, key=k) for k, v in d.get("blocks", {}).items()}

    header, blocks = {"format": "local jsonl"}, {}
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            if n == len(lines) - 1:
                print(f"[cache] ignoring a truncated final line -- an interrupted "
                      f"append; that block keeps its template", file=sys.stderr)
                continue
            raise SystemExit(f"{path}:{n + 1} is not JSON and is not the last line "
                             f"-- the cache is corrupt, not merely interrupted.")
        if rec.get("cache") == CACHE_KIND:
            header.update(rec)
        elif "key" in rec:
            blocks[rec["key"]] = rec
    return header, blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_r0.jsonl")
    ap.add_argument("--out", default="data/sft_r0_teacher.jsonl")
    ap.add_argument("--cache", default="data/teacher_cache_r0_local.jsonl")
    ap.add_argument("--min-distractors", type=int, default=1,
                    help="must match the value the cache was minted with, or the "
                         "selection here and the selection there disagree")
    ap.add_argument("--legacy-vocab", action="store_true",
                    help="verify the Task reconstruction under the pre-widening "
                         "name pools; required for any base built before 566f4d2")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build from an incomplete cache; without it a cache "
                         "covering under 90%% of the selected blocks refuses")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    cfg = TeacherConfig(min_distractors=args.min_distractors)

    base = Path(args.base)
    lines = [l for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    fp = fingerprint(base)
    by_hop = collections.Counter(r["meta"]["difficulty"] for r in rows)
    print(f"[base] {base}: {len(rows)} records, hop mix {dict(sorted(by_hop.items()))}, "
          f"fingerprint {fp}")

    # --- select, exactly as the minter did ---------------------------------
    jobs, sel_by_hop, blocks = [], collections.Counter(), 0
    for r in rows:
        msgs = r["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        blocks += sum(1 for _ in walk(msgs))
        for b in select(msgs, cfg):
            sel_by_hop[r["meta"]["difficulty"]] += 1
            jobs.append({"task_id": r["meta"]["task_id"], "step": b["step"],
                         "question": question, "difficulty": r["meta"]["difficulty"],
                         "msg_index": b["msg_index"], "row": r})
    print(f"[select] {len(jobs)} of {blocks} blocks = {len(jobs) / blocks:.1%}  "
          f"(>= {cfg.min_distractors} same-kind distractor alongside the rank-1 hit)")
    for h in sorted(sel_by_hop):
        print(f"           {h}-hop: {sel_by_hop[h]}")

    # --- verify the base is the population the cache was minted against ----
    ctx = legacy_vocab() if args.legacy_vocab else contextlib.nullcontext()
    if args.legacy_vocab:
        print("[vocab] pre-widening name pools (24x20 people, 14x8 orgs)")
    seen_ids, ok, bad_ids = set(), 0, 0
    with ctx:
        for j in jobs:
            if j["task_id"] in seen_ids:
                continue
            seen_ids.add(j["task_id"])
            if rebuild_ok(j["task_id"], j["difficulty"], j["question"]):
                ok += 1
            else:
                bad_ids += 1
    print(f"[verify] Task reconstruction: {ok} ok, {bad_ids} failed of "
          f"{ok + bad_ids} distinct task ids")
    if bad_ids > ok:
        raise SystemExit(
            "more than half the records do not reconstruct from their task_id. "
            "For a base built before commit 566f4d2 pass --legacy-vocab; otherwise "
            "the base and the installed world generator disagree.")

    # --- the cache ---------------------------------------------------------
    cache_path = Path(args.cache)
    header, cache = load_any_cache(cache_path)
    if header.get("base_fingerprint") and header["base_fingerprint"] != fp:
        raise SystemExit(
            f"cache {cache_path} was minted against base "
            f"{header['base_fingerprint']}, but {base} fingerprints {fp}. The task "
            f"ids would still resolve, so this would silently paste another "
            f"population's reasoning into this set. Point --base at the set the "
            f"cache came from, or mint a new cache.")
    minted = {k for k, v in cache.items() if v.get("think")}
    failed = len(cache) - len(minted)
    att = sum(v.get("attempts", 1) for v in cache.values()) or 0
    print(f"[cache] {cache_path} ({header.get('format', '?')}), model "
          f"{header.get('model', '?')}: {len(cache)} keys, {len(minted)} usable, "
          f"{failed} gave up"
          + (f", {att / len(cache):.2f} attempts/block" if cache else ""))

    coverage = len(minted) / len(jobs) if jobs else 0.0
    print(f"[cover] {coverage:.1%} of the {len(jobs)} selected blocks have teacher "
          f"prose")
    if coverage < 0.9 and not args.allow_partial:
        raise SystemExit(
            f"the cache covers {coverage:.1%} of the selected blocks. Finish the "
            f"mint (re-run scripts/14_local_teacher.py, it resumes), or pass "
            f"--allow-partial to build the arm from what is there. An arm that is "
            f"mostly its own base measures nothing.")

    # --- apply -------------------------------------------------------------
    converted_rows, converted_blocks, kept_template = set(), 0, 0
    applied_by_hop = collections.Counter()
    for j in jobs:
        hit = cache.get(cache_key(j["task_id"], j["step"]))
        if hit is None or not hit.get("think"):
            kept_template += 1
            continue
        r = j["row"]
        i = j["msg_index"]
        r["messages"][i]["content"] = apply_block(r["messages"][i]["content"],
                                                  hit["think"])
        converted_rows.add(id(r))
        converted_blocks += 1
        applied_by_hop[j["difficulty"]] += 1

    out_lines = [json.dumps(r, ensure_ascii=False) if id(r) in converted_rows else line
                 for r, line in zip(rows, lines)]
    out = Path(args.out)
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    verbatim = len(rows) - len(converted_rows)
    print(f"\nteacher blocks applied: {converted_blocks}/{len(jobs)} "
          f"({converted_blocks / len(jobs):.1%}); {kept_template} kept their template")
    for h in sorted(applied_by_hop):
        print(f"           {h}-hop: {applied_by_hop[h]}")
    print(f"records touched {len(converted_rows)}/{len(rows)}, "
          f"{verbatim} lines verbatim")
    print(f"wrote {out}")

    if not args.no_audit:
        print(f"\n=== auditing {out} ===")
        rc = subprocess.call([sys.executable, "tests/audit_sft.py", str(out)],
                             cwd=REPO)
        if rc != 0:
            raise SystemExit(f"DEFECTS PRESENT in {out} -- do not train on it.")


if __name__ == "__main__":
    main()
