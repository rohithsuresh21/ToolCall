"""Mint teacher blocks through the Batch API, gate them, and cache what survives.

Split from atr/data/teacher.py on purpose: everything that DECIDES anything --
which blocks to rewrite, what the prompt is, whether a block is admissible -- is
stdlib-only and testable with no network and no key, exactly like the rest of the
task-generation path. This file is the only part that needs `anthropic`, and it
makes no judgements of its own.

Batched because the work is offline by construction (the cache is the
determinism, see teacher.py), so nothing waits on latency and the 50% discount is
free. Two passes at most: mint, gate, re-mint the failures with the offending
token named, gate again, keep the template for whatever still fails.

RETRIES ARE COUNTED AND THE COUNT IS WIRED. atr/data/naturalize.py initialised
`stats["retries"] = 0` with nothing to increment it, so every run it ever made
reported exactly 0 retries -- a number that read as a clean result and was an
unwired counter, while deciding whether minting was affordable at all. The attempt
count here is carried per block into the cache and summed from the cache, so it
cannot drift away from what actually happened.
"""
from __future__ import annotations

import sys
import time

from .reasoning_teacher import (Cache, TeacherConfig, SYSTEM, custom_id, repair_hint,
                      user_message, violations)


def _client():
    try:
        import anthropic
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit("the `anthropic` package is required to mint: "
                         "pip install anthropic") from e
    return anthropic.Anthropic()


def _params(cfg: TeacherConfig, messages: list[dict]) -> dict:
    return {"model": cfg.model, "max_tokens": cfg.max_tokens, "system": SYSTEM,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": cfg.effort}}


def _text_of(message) -> str:
    """The visible answer. Thinking blocks come first in `content` and carry no
    text under the default display, so match on type rather than taking [0]."""
    for b in message.content:
        if getattr(b, "type", None) == "text":
            return b.text.strip().strip('"').strip()
    return ""


def _run_batch(client, requests: list[dict], poll: int, label: str) -> dict:
    """Submit, wait, return {custom_id: message}. Errors are left out rather than
    raised: a single failed request must not lose the other few thousand."""
    batch = client.messages.batches.create(requests=requests)
    print(f"[{label}] submitted {len(requests)} requests as {batch.id}", flush=True)
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"[{label}] {b.processing_status} "
              f"succeeded={c.succeeded} errored={c.errored} processing={c.processing} "
              f"({time.time() - t0:.0f}s)", flush=True)
        time.sleep(poll)
    out, errors = {}, 0
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            out[r.custom_id] = r.result.message
        else:
            errors += 1
    print(f"[{label}] ended: {len(out)} succeeded, {errors} failed "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def probe(jobs: list[dict], cfg: TeacherConfig, n: int = 20) -> list[dict]:
    """Mint `n` blocks SYNCHRONOUSLY and return them ungated-but-labelled.

    Two jobs, both worth the ~$0.15. It validates the request shape against the
    live API before a batch of several thousand is submitted -- `thinking`,
    `output_config` and `effort` are all rejected at submit time by models that do
    not take them, and finding that out on request 1 is cheaper than on request
    3720. And it puts real teacher prose on screen to be READ, which is the only
    check that covers what the gates cannot see: an invented LOWERCASE claim.
    `teacher_caps` scans capitalised runs and `unseen_numbers` scans digits, so
    "Vanguard Maritime is a shipping company" passes both with `shipping` invented.

    Nothing is cached here. A probe is for looking at, not for building from."""
    client = _client()
    out = []
    for j in jobs[:n]:
        msg = client.messages.create(
            **_params(cfg, [{"role": "user",
                             "content": user_message(j["messages"], j["block"])}]))
        text = _text_of(msg)
        out.append({"task_id": j["task_id"], "step": j["step"],
                    "question": j["question"], "query": j["block"]["query"],
                    "template": j["block"]["template"], "teacher": text,
                    "violations": violations(text, j["question"], j["seen_norm"]),
                    "in_tok": msg.usage.input_tokens,
                    "out_tok": msg.usage.output_tokens})
        print(f"  probed {len(out)}/{min(n, len(jobs))}", end="\r", flush=True)
    print()
    return out


def mint(jobs: list[dict], cache: Cache, cfg: TeacherConfig,
         poll: int = 20, dry_run: bool = False) -> dict:
    """`jobs` are {task_id, step, question, seen_norm, messages, block}.

    Only cache misses are sent. Returns a stats dict; the cache is mutated in
    place and is the caller's to save."""
    todo = [j for j in jobs if cache.get(j["task_id"], j["step"]) is None]
    stats = {"jobs": len(jobs), "cached": len(jobs) - len(todo), "sent": len(todo),
             "accepted": 0, "repaired": 0, "gave_up": 0,
             "rejected_first_pass": 0, "reasons": {}}
    if dry_run or not todo:
        return stats

    client = _client()
    by_id = {custom_id(j["task_id"], j["step"]): j for j in todo}
    reqs = [{"custom_id": cid,
             "params": _params(cfg, [{"role": "user",
                                      "content": user_message(j["messages"], j["block"])}])}
            for cid, j in by_id.items()]

    got = _run_batch(client, reqs, poll, "pass 1")
    retry = {}
    for cid, msg in got.items():
        j = by_id[cid]
        text = _text_of(msg)
        bad = violations(text, j["question"], j["seen_norm"])
        if not bad:
            cache.put(j["task_id"], j["step"], text, 1)
            stats["accepted"] += 1
        else:
            stats["rejected_first_pass"] += 1
            stats["reasons"][bad[0].split(":")[0]] = \
                stats["reasons"].get(bad[0].split(":")[0], 0) + 1
            retry[cid] = (j, text, bad)

    if retry and cfg.max_attempts > 1:
        reqs2 = [{"custom_id": cid,
                  "params": _params(cfg, [
                      {"role": "user", "content": user_message(j["messages"], j["block"])},
                      {"role": "assistant", "content": prev},
                      {"role": "user", "content": repair_hint(bad)}])}
                 for cid, (j, prev, bad) in retry.items()]
        got2 = _run_batch(client, reqs2, poll, "pass 2")
        for cid, msg in got2.items():
            j = retry[cid][0]
            text = _text_of(msg)
            if not violations(text, j["question"], j["seen_norm"]):
                cache.put(j["task_id"], j["step"], text, 2)
                stats["repaired"] += 1

    stats["gave_up"] = stats["sent"] - stats["accepted"] - stats["repaired"]
    return stats


def report(stats: dict, cache: Cache) -> None:
    print("\n--- mint ---")
    for k in ("jobs", "cached", "sent", "accepted", "repaired", "gave_up"):
        print(f"  {k:<20} {stats.get(k, 0)}")
    if stats.get("reasons"):
        print("  first-pass rejections by reason:")
        for k, v in sorted(stats["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<18} {v}")
    n = len(cache.blocks)
    if n:
        att = sum(b.get("attempts", 1) for b in cache.blocks.values())
        print(f"  cache holds {n} blocks, {att} teacher calls, "
              f"{att / n:.2f} attempts/block")
    if stats.get("gave_up"):
        print(f"  NOTE {stats['gave_up']} blocks kept their TEMPLATE "
              f"(failed the gate twice)", file=sys.stderr)
