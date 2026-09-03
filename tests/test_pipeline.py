"""End-to-end pipeline test. Runs on CPU in seconds; no model required."""
import json, pathlib, random, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.agent.chatml import assistant_spans, render
from atr.agent.loop import LoopConfig
from atr.data.build_sft import ExportConfig, export
from atr.data.rejection import FilterConfig, filter_and_balance
from atr.data.teacher import collect, dump, load
from atr.eval.harness import aggregate, evaluate, oracle_backend
from atr.tasks.generator import dev_set, generate
from atr.tasks.retrievability import assert_answer_retrievable
from atr.tools.adapter import get_registry

FAILS = []
def check(cond, msg):
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    if not cond: FAILS.append(msg)

# 1. determinism of the world / tasks
t1 = generate(30, seed_start=0); t2 = generate(30, seed_start=0)
check([t.prompt for t in t1] == [t.prompt for t in t2], "task generation is deterministic")
train = {t.seed for t in generate(500, seed_start=0)}
dev = {t.seed for t in dev_set(6)}
check(not (train & dev), "train and dev seed ranges are disjoint")

# 2. oracle must be perfect (the harness's own unit test)
tasks = dev_set(6)
cards, trajs = evaluate(tasks, oracle_backend(tasks), cfg=LoopConfig(max_steps=10), progress=False)
o = aggregate(cards)["overall"]
check(o["success"] == 1.0, f"oracle task success == 1.0 (got {o['success']})")
check(o["selection_ok"] == 1.0, f"oracle tool selection == 1.0 (got {o['selection_ok']})")
check(o["args_strict"] == 1.0, f"oracle strict args == 1.0 (got {o['args_strict']})")
check(o["necessity_ok"] == 1.0, f"oracle necessity == 1.0 (got {o['necessity_ok']})")

# 3. family-conditional checks. `all()` over an empty list is True, so any check
#    scoped to one family passes VACUOUSLY once that family stops being minted.
#    Gate on active_families() and assert non-emptiness, so dropping a family
#    turns these off explicitly instead of leaving them green and meaningless.
from atr.tasks.generator import active_families

ACTIVE = active_families()
check(set(ACTIVE) == {"musique_2hop", "musique_3hop", "musique_4hop"},
      f"active families are the judge's 2/3/4-hop only (got {ACTIVE})")
check(not [t for t in tasks if t.task_type not in ACTIVE],
      f"dev set mints nothing outside the active families "
      f"(stray: {sorted({t.task_type for t in tasks if t.task_type not in ACTIVE})})")

from atr.agent.backends import MockBackend

if "no_tool" in ACTIVE:
    nt = [t for t in tasks if t.task_type == "no_tool"]
    ntc = [c for c in cards if c.task_type == "no_tool"]
    check(len(nt) > 0 and len(ntc) > 0,
          f"no_tool is an active family so the dev set must contain some (got {len(nt)})")
    check(all(c.num_calls == 0 for c in ntc), "oracle makes zero calls on no_tool tasks")
    greedy = MockBackend(
        plans={t.task_id: [{"name": "search", "arguments": {"query": t.prompt[:40]}}] for t in nt},
        answers={t.task_id: t.oracle_answer for t in nt}, degrade_on_error=False)
    gc, _ = evaluate(nt, greedy, cfg=LoopConfig(max_steps=4), progress=False)
    check(all(not c.success for c in gc),
          "right answer + unnecessary tool call == failure on no_tool")
else:
    # Not skipped silently: assert the family really is gone end to end.
    check(not [t for t in tasks if t.task_type == "no_tool"],
          "no_tool is inactive, so the dev set must contain none")
    check(all(t.needs_tool for t in tasks),
          "every active task needs the tool (no_tool is the only family that does not)")
    print("SKIP  no_tool necessity checks -- family weight is 0 (not in the official eval)")

# Every hop task must actually require dependent searches.
hop = [t for t in tasks if t.task_type.startswith("musique")]
check(len(hop) == len(tasks) > 0, f"the whole dev set is hop tasks ({len(hop)}/{len(tasks)})")
check(all(len(t.oracle_plan) >= 1 for t in hop), "every hop task carries a search plan")

# 4. toolset is the single BM25 `search` tool (the judge-collapsed contract:
#    calculator, web_search, fetch_page, db_query, db_aggregate and the retired
#    action family are all gone), and oracle remains perfect
reg5 = get_registry("builtin")
tool_names = sorted(reg5.names())
check(tool_names == ["search"], f"active toolset is exactly the 1 BM25 search tool (got {tool_names})")
check("send_message" not in tool_names and "calculator" not in tool_names
      and "db_query" not in tool_names and "fetch_page" not in tool_names,
      "retired tools are not registered")
check(all(t.task_type in ACTIVE for t in tasks),
      f"only active families are generated (got {sorted({t.task_type for t in tasks})})")
o5, _ = evaluate(tasks, oracle_backend(tasks), cfg=LoopConfig(max_steps=10), progress=False)
agg5 = aggregate(o5)["overall"]
check(agg5["success"] == 1.0, f"oracle success stays 1.0 on the single-tool toolset (got {agg5['success']})")

# 4b. the oracle score above CANNOT see whether the plan retrieves its own answer:
#     oracle_backend replays the plan, then emits oracle_answer from a lookup table
#     without reading a tool result, so it prints 1.0 either way. Replay the plans
#     through the real registry and check the passages that actually come back.
ret = assert_answer_retrievable(tasks)
check(ret["unretrievable"] == 0,
      f"every oracle plan retrieves its own gold answer ({ret['n_scored']} scored, "
      f"{ret['n_skipped']} skipped)")
check(all(f["unretrievable"] == 0 for f in ret["by_family"].values()),
      f"no family has an unretrievable answer ({ret['by_family']})")
# ... and the answer must not be retrievable EARLY either. A hit before the last
# call means the chain is truncatable: the model can stop short of its labelled
# hop count and still be scored right, which is the same lazy policy the
# disconnection filter exists to suppress. The generator's prefix-leak filter
# rejects those candidates, so a non-zero count here means it regressed.
check(ret["answer_before_last_call"] == 0,
      f"no dev task leaks its answer before the terminal read "
      f"({ret['answer_before_last_call']}/{ret['n_scored']} do)")

# 5. multi-hop oracle trajectories really do perform dependent searches
mh = [c for c in cards if c.task_type.startswith("musique")]
check(all(c.num_calls >= 2 for c in mh), "multi-hop oracle trajectories make 2+ searches")
# L relation steps == L+1 searches: L to walk the chain, 1 to read the leaf passage.
check(all(len(t.oracle_plan) == t.difficulty + 1 for t in hop),
      "every hop task plans difficulty+1 searches (chain walk + terminal read)")
check(all(c.recovery_ok is None for c in cards), "no fabricated recovery events without recovery family")

# 6. tokenisation spans line up with assistant turns
reg = get_registry("builtin")
multi = next(j for j in trajs if j.num_tool_calls >= 2)
msgs = multi.messages
ids, spans = assistant_spans(msgs, reg.schemas(), lambda s: list(s))
full = "".join(ids)
n_assist = sum(1 for m in msgs if m["role"] == "assistant")
check(len(spans) == n_assist, f"one loss span per assistant turn ({len(spans)} vs {n_assist})")
check(all(msgs[i]["content"] in full[s:e]
          for (s, e), i in zip(spans, [k for k, m in enumerate(msgs) if m["role"] == "assistant"])),
      "each span contains exactly its assistant turn's text")
check("<tool_response>" in render(msgs, reg.schemas()), "tool results render as tool_response blocks")

# 7. data pipeline
rng = random.Random(0)
corrupt = lambda tid, step, call: (None if rng.random() < 0.15 else call)
recs = collect(generate(80, seed_start=10_000), oracle_backend(generate(80, seed_start=10_000), corrupt),
               cfg=LoopConfig(max_steps=10), samples_per_task=2, progress=False)
kept, summary = filter_and_balance(recs, FilterConfig(max_per_task=1))
check(0 < len(kept) < len(recs), f"filter rejects some and keeps some ({len(kept)}/{len(recs)})")
check(all(c.success for _, _, c in kept), "every kept trajectory is successful")
p = pathlib.Path("artifacts/_test_sft.jsonl")
stats = export(kept, p, ExportConfig())
rows = [json.loads(l) for l in p.read_text().splitlines()]
check(len(rows) == stats["written"] > 0, "SFT export writes records")
check(all(r["messages"][-1]["role"] == "assistant" for r in rows), "every record ends on an assistant turn")
check(all("<final_answer>" in r["messages"][-1]["content"] for r in rows), "every record ends with a final answer")
import re
bad = [m["content"] for r in rows for m in r["messages"]
       if m["role"] == "assistant" and "<tool_call>" in m["content"]
       and not re.search(r'<tool_call>\{"name": ".+?", "arguments": \{.*\}\}</tool_call>', m["content"], re.S)]
check(not bad, f"all exported tool calls are canonical ({len(bad)} bad)")

# 8. reward ordering
from atr.train.reward import compute_reward, group_advantages
pairs = [(t, c) for (t, _, c) in recs]
good = [compute_reward(t, c)[0] for t, c in pairs if c.success]
bad_r = [compute_reward(t, c)[0] for t, c in pairs if not c.success]
check(min(good) > max(bad_r), f"every success outranks every failure ({min(good):.2f} > {max(bad_r):.2f})")
check(group_advantages([0.5]*6) == [0.0]*6, "a dead group produces zero advantage")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
