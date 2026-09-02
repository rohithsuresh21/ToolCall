# Comprehensive end-to-end dataset verification for the ATR report.
# Exercises all 5 families with the naturalized loader WIRED IN (the exact path
# that produces sft.jsonl), plus the CLI oracle path. Prints real numbers used by
# the report. Exits non-zero on any failure.
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atr"))

from atr.tasks.generator import active_families, generate, dev_set, _SHORTCUT_STATS
from atr.tools.world import build_world
from atr.data.naturalize import load_naturalized_loader
from atr.eval.harness import oracle_backend, evaluate, aggregate
from atr.agent.loop import LoopConfig

CACHE = "artifacts/naturalized_passages_scaled.json"
failures = []
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)

# ---- 0. load the naturalized loader (the wired opt-in hook) ----
loader = load_naturalized_loader(CACHE)
check("naturalized cache loads", loader is not None, f"cache={CACHE}")

# ---- 1. train generation with the loader + shortcut filter ----
n = 200
before = dict(_SHORTCUT_STATS)
tasks = generate(n, seed_start=0, filter_shortcuts=True, text_loader=loader)
types = Counter(t.task_type for t in tasks)
check("train generates 200 tasks", len(tasks) == n)
# Family coverage is asserted against the ACTIVE mix, not a hard-coded list of 5.
# Two directions, both needed: every active family must actually appear (a family
# that silently stops being minted is a broken generator), and no INACTIVE family
# may appear (a zero weight that still leaks is the bug this replaced).
ACTIVE = set(active_families())
check("active families are the judge's 2/3/4-hop only",
      ACTIVE == {"musique_2hop", "musique_3hop", "musique_4hop"}, f"{sorted(ACTIVE)}")
check("every active family is represented", set(types) >= ACTIVE,
      f"missing={sorted(ACTIVE - set(types))} got={dict(types)}")
check("no inactive family is minted", not (set(types) - ACTIVE),
      f"leaked={sorted(set(types) - ACTIVE)} got={dict(types)}")
after = dict(_SHORTCUT_STATS)
checked = after.get("checked", 0) - before.get("checked", 0)
rejected = after.get("rejected", 0) - before.get("rejected", 0)
rate = (100.0 * rejected / checked) if checked else 0.0
check("shortcut filter active", checked > 0 and rejected >= 0,
      f"checked={checked} rejected={rejected} rate={rate:.1f}%")

# ---- 2. gold invariant: answerable tasks have a value; unanswerable must be kind=none ----
answerable = [t for t in tasks if t.task_type != "unanswerable"]
unan = [t for t in tasks if t.task_type == "unanswerable"]
good = all(t.oracle_answer and t.gold.get("value") not in (None, "", []) for t in answerable) and \
       all(t.gold.get("kind") == "none" and not t.gold.get("value") for t in unan)
check("gold invariant holds (ans. have value; unans. kind=none)", good,
      f"answerable={len(answerable)} unanswerable={len(unan)}")
# The unanswerable half of that check is vacuous while the family is inactive --
# say so rather than reporting a green tick for an assertion over an empty list.
check("gold invariant is non-vacuous on the answerable side", len(answerable) > 0,
      f"answerable={len(answerable)}")
if "unanswerable" not in ACTIVE:
    check("unanswerable is inactive, so none are minted", len(unan) == 0, f"unanswerable={len(unan)}")
    print("NOTE  the unans. kind=none clause above is vacuous (family weight is 0)")

# ---- 3. naturalization actually flows into task worlds (prose differs) ----
nat_seen = 0
nat_docs = 0
for seed in range(0, 4):
    w = build_world(seed, text_loader=loader)
    nat_docs += sum(1 for d in w.documents if d.get("naturalized"))
    nat_seen += 1
check("naturalized prose wired into build_world", nat_docs > 0, f"docs naturalized(seeds0-3)={nat_docs}")

# ---- 4. dev set uses held-out dev shapes with the loader ----
dev = dev_set(n_per_type=3, text_loader=loader)
dev_types = Counter(t.task_type for t in dev)
check("dev set covers exactly the active families", set(dev_types) == ACTIVE, f"{dict(dev_types)}")
# dev_set feeds the GRPO canary, so an off-family task here re-enters checkpoint
# selection through the back door even with the train mix correct.
check("dev set is balanced across active families", len(set(dev_types.values())) == 1,
      f"{dict(dev_types)}")

# ---- 5. end-to-end CLI-equivalent oracle with naturalization active ----
cfg = LoopConfig(max_steps=10, text_loader=loader)
cards, _ = evaluate(dev, oracle_backend(dev), cfg=cfg, progress=False)
rep = aggregate(cards)["overall"]
succ = rep["success"]
check("oracle dev success == 100% w/ naturalized prose", succ == 1.0, f"success={100*succ:.1f}%")

print()
print(f"SUMMARY: {len(dev)} dev tasks; failures={len(failures)}")
print(f"dataset families: {dict(types)}")
print(f"shortcut-filter: checked={checked} rejected={rejected} rate={rate:.1f}%")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL DATASET CHECKS PASS")
