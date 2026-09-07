# Per-source GRPO group accounting: does ONE step tell us the live-group rate for
# synthetic vs real tasks?
#
# This exists to replace a guess. `real_fraction` is 0.2, picked before MuSiQue was
# permitted and never measured. The number that should set it is the share of each
# source's groups that come back LIVE, because a dead group costs a full rollout
# and contributes exactly zero gradient. Synthetic 2-hop is solved ~100% of the
# time, so its groups die on zero variance no matter how many are drawn; real tasks
# sit near 46% and disagree with themselves, which is the only condition under which
# a GRPO group teaches anything.
#
# CPU only, no model, no GPU -- built with __new__ like test_fix2/test_grpo_resume.
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.tasks.schema import ScoreCard, Task, task_source
from atr.train.grpo import GRPOConfig, GRPOTrainer

FAILED = 0


def check(cond, label):
    global FAILED
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED += 1


def _task(kind: str, i: int) -> Task:
    """A synthetic task has no documents; a real one owns its candidate set."""
    docs = [{"doc_id": "0", "title": "T", "text": "x"}] if kind == "real" else []
    return Task(task_id=f"{kind}-{i}", seed=i, prompt="q", task_type="musique_2hop",
                difficulty=2, gold={"kind": "text", "value": "a"}, documents=docs)


def _rec(kind: str, i: int, reward: float):
    card = ScoreCard(task_id=f"{kind}-{i}", task_type="musique_2hop", difficulty=2,
                     num_calls=3)
    card.final_f1 = reward

    class _Step:
        pass

    class _Traj:
        stop_reason = "answered"
        steps = [_Step()]

    return {"group": f"{kind}-{i}", "task": _task(kind, i), "reward": reward,
            "card": card, "void": False, "traj": _Traj()}


def _trainer() -> GRPOTrainer:
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.cfg = GRPOConfig(gigpo=False, dqw=False, efficiency_lambda=0.0)
    t.rng = random.Random(0)
    t.history = []
    t._last_group_stats = []
    return t


# --- task_source is the single place the distinction is named ---------------
check(task_source(_task("synthetic", 0)) == "synthetic",
      "task_source: a task with no documents is synthetic")
check(task_source(_task("real", 0)) == "real",
      "task_source: a task carrying its own candidate set is real")

# --- the split itself -------------------------------------------------------
# Two synthetic groups whose rollouts all score identically (the "solved 100% of
# the time" case -> zero variance -> dead) and two real groups that disagree.
records = []
for i in range(2):
    records += [_rec("synthetic", i, 1.0) for _ in range(4)]      # no disagreement
for i in range(2):
    records += [_rec("real", i, 1.0), _rec("real", i, 0.0),
                _rec("real", i, 1.0), _rec("real", i, 0.0)]        # disagreement
for r in records:                       # group id must be per TASK, not per record
    r["group"] = r["task"].task_id

t = _trainer()
info = t.assign_advantages(records)

check("groups_by_source" in info, "assign_advantages reports groups_by_source")
bs = info.get("groups_by_source", {})
check(set(bs) == {"real", "synthetic"},
      f"both sources appear in one step's breakdown (got {sorted(bs)})")
check(bs.get("synthetic", {}).get("live_rate") == 0.0,
      f"synthetic groups with no disagreement are dead "
      f"(live_rate {bs.get('synthetic', {}).get('live_rate')})")
check(bs.get("real", {}).get("live_rate") == 1.0,
      f"real groups that disagree are live "
      f"(live_rate {bs.get('real', {}).get('live_rate')})")

# The whole point: the two rates are DIFFERENT, so one step distinguishes them.
check(bs["synthetic"]["live_rate"] != bs["real"]["live_rate"],
      "one step separates the two sources' live-group rates -- which is what "
      "should set real_fraction instead of a guess")

# --- every group carries its source, and a group has exactly one -------------
gs = t._last_group_stats
check(all("source" in g for g in gs), "every group_stats row carries a source")
check({g["source"] for g in gs} == {"real", "synthetic"},
      "group sources cover both kinds")
check(all(g["source"] == ("real" if g["group"].startswith("real") else "synthetic")
          for g in gs), "each group's source matches its task")

# --- dead synthetic groups are attributed to zero_variance, not to failure ---
dead_syn = [g for g in gs if g["source"] == "synthetic" and not g["live"]]
check(dead_syn and all(g["dead_reason"] == "zero_variance" for g in dead_syn),
      f"a synthetic group dies from lack of DISAGREEMENT, not low reward "
      f"(reasons {sorted({g['dead_reason'] for g in dead_syn})})")

# --- per-source F1 in the sampled stats -------------------------------------
smp = GRPOTrainer._sampled_stats(records)
check("smp_f1_by_source" in smp, "sampled stats split final_f1 by source")
check(smp["smp_f1_by_source"]["synthetic"] == 1.0,
      f"synthetic F1 is saturated (got {smp['smp_f1_by_source'].get('synthetic')}) "
      f"-- which is exactly why its groups carry no gradient")
check(smp["smp_f1_by_source"]["real"] == 0.5,
      f"real F1 is not saturated (got {smp['smp_f1_by_source'].get('real')})")

print()
print("ALL PASS" if not FAILED else f"FAILURES: {FAILED}")
sys.exit(1 if FAILED else 0)
