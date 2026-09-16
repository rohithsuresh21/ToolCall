"""Invariants for the STeCa-style recovery trajectories (atr/data/recovery.py).

Run it:  PYTHONPATH=. python tests/test_recovery.py

The one that matters most is `test_miss_is_genuine`. A recovery record is only
worth training on if the bad call ACTUALLY lost ground: in this world `search`
boosts a title match 1.6x, so a query carrying the source entity's name retrieves
the source's own passage -- the very passage naming the next hop -- and splicing a
"that didn't work, let me retry" onto it would teach re-querying while holding the
answer, which is the loop the file exists to remove. The genuine-miss check is
what stands between the two, so it is checked here on real episodes rather than
trusted.

The others pin the properties audit_sft.py cannot see on a set it was not given,
plus the two determinism rules a rebuild depends on.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.agent.loop import LoopConfig  # noqa: E402
from atr.data.build_sft import ExportConfig, trajectory_to_record  # noqa: E402
from atr.data.recovery import (_EMPTY, _IRRELEVANT, RecoveryConfig,  # noqa: E402
                               build_recovery)
from atr.data.teacher import collect_oracle  # noqa: E402
from atr.tasks.generator import _gold_as_answer, generate  # noqa: E402
from atr.tasks.schema import norm_text, psychic_caps  # noqa: E402
from atr.tools.world import build_world  # noqa: E402

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)

FAILED = 0


def check(cond, label):
    global FAILED
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED += 1


def _episodes(n=18, seed_start=880_000):
    """Fresh tasks + their exported records, the same pair the build script has."""
    tasks = generate(n, seed_start=seed_start,
                     mix={"musique_2hop": 0.34, "musique_3hop": 0.33, "musique_4hop": 0.33})
    recs = collect_oracle(tasks, cfg=LoopConfig(max_steps=10), progress=False)
    cfg = ExportConfig(oracle_rationale=True)
    out = []
    for t, j, _ in recs:
        r = trajectory_to_record(t, j, cfg)
        if r is None:
            continue
        got = build_recovery(t, r["messages"], world=build_world(t.seed))
        if got is not None:
            out.append((t, r, got))
    return out


def _hits(content):
    try:
        return json.loads(content).get("results", []) or []
    except ValueError:
        return []


def _blob(content):
    return [norm_text(f"{h.get('title', '')} {h.get('text', '')}") for h in _hits(content)]


EPS = _episodes()
print(f"\n{len(EPS)} recovery episodes built\n")
check(len(EPS) >= 12, "build_recovery finds a genuine miss on nearly every task")


# --- 1. the miss is genuine -------------------------------------------------
bad_ok = next_ok = True
for t, r, got in EPS:
    msgs = got["messages"]
    calls = [i for i, m in enumerate(msgs)
             if m["role"] == "assistant" and CALL.search(m["content"])]
    n = got["hop"]
    bad_result = msgs[calls[n] + 1]["content"]
    gold = norm_text(_gold_as_answer(t.gold))
    if any(gold in b for b in _blob(bad_result)):
        bad_ok = False
    chain = list(t.chain)
    if n + 1 < len(chain):
        nxt = norm_text(chain[n + 1])
        if any(nxt in b for b in _blob(bad_result)):
            next_ok = False
check(bad_ok, "the bad call never returns the gold answer (audit axis 4 stays clean)")
check(next_ok, "the bad call never returns the entity the next hop is written from")


# --- 2. the repair is the oracle's own call ---------------------------------
repaired_ok = shape_ok = True
for t, r, got in EPS:
    msgs = got["messages"]
    calls = [i for i, m in enumerate(msgs)
             if m["role"] == "assistant" and CALL.search(m["content"])]
    n = got["hop"]
    want = t.oracle_plan[n]["arguments"]["query"]
    got_q = json.loads(CALL.search(msgs[calls[n + 1]]["content"]).group(1))["arguments"]["query"]
    if got_q != want:
        repaired_ok = False
    if len(msgs) != len(r["messages"]) + 2 or len(calls) != len(t.oracle_plan) + 1:
        shape_ok = False
    if msgs[-1] != r["messages"][-1]:
        shape_ok = False
check(repaired_ok, "the turn after the miss is the oracle plan's own query, verbatim")
check(shape_ok, "exactly two messages are added; the final answer turn is untouched")


# --- 3. hop 0 is never the target -------------------------------------------
check(all(got["hop"] >= 1 for _, _, got in EPS),
      "hop 0 is never corrupted (a bad FIRST query is audit axis 1)")


# --- 4. the recovery <think> names only what the episode has seen ------------
# Same reader audit_sft.py uses, and the same accumulation rule: the prompt plus
# every tool_response STRICTLY earlier than the turn. Assistant text is not "seen",
# which is why no template may quote the bad query back.
psychic = 0
for t, r, got in EPS:
    msgs = got["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    tools = [m["content"] for m in msgs if m["role"] == "tool"]
    seen = norm_text(user)
    i = 0
    for m in msgs:
        if m["role"] == "assistant":
            tm = THINK.search(m["content"])
            if tm:
                for c in psychic_caps(tm.group(1), user, prose=True):
                    if norm_text(c) and norm_text(c) not in seen:
                        psychic += 1
                        break
            if i < len(tools):
                seen += " " + " ".join(norm_text(x) for x in _blob(tools[i]))
                i += 1
check(psychic == 0, "no <think> block names an entity the episode has not been shown")


# --- 5. no template opens a sentence with a bare entity ---------------------
# psychic_caps(prose=True) ignores sentence-initial capitals, so a one-word
# invented entity parked there is invisible to the auditor. Same rule and same
# reason as tests/test_real_chains.py pins for atr/data/reasoning.py.
opens = []
for tmpl in _EMPTY + _IRRELEVANT:
    for m in re.finditer(r"\{(prev|goal)\}", tmpl):
        before = tmpl[:m.start()].rstrip()
        if not before or before[-1] in ".?!:":
            opens.append(tmpl[:60])
check(not opens, f"no recovery template opens a sentence with a bare entity {opens}")


# --- 6. determinism ---------------------------------------------------------
same = True
for t, r, got in EPS[:8]:
    again = build_recovery(t, r["messages"], world=build_world(t.seed))
    if again is None or again["messages"] != got["messages"]:
        same = False
check(same, "build_recovery is deterministic: no live RNG anywhere in the splice")

# ... and specifically not sensitive to the global random stream, which would make
# a rebuild depend on whatever ran before it.
random.seed(999)
[random.random() for _ in range(50)]
t, r, got = EPS[0]
check(build_recovery(t, r["messages"], world=build_world(t.seed))["messages"]
      == got["messages"], "the splice does not draw from the global random stream")


# --- 7. the config's rates are weighted toward the long chains --------------
rates = RecoveryConfig().rate_by_hop
check(rates[4] > rates[3] > rates[2],
      "recovery rate rises with hop count (the looping is at 3 and 4 hops)")

print(f"\n{'ALL CHECKS PASSED' if FAILED == 0 else f'{FAILED} CHECK(S) FAILED'}")
sys.exit(1 if FAILED else 0)
