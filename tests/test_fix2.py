"""FIX-2 verification suite: one test block per F-item. CPU-only, no model downloads."""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
FAILS = []


def check(cond, label):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


# ---------------------------------------------------------------- F11 bank
from atr.tasks.generator import NO_TOOL_BANK, generate

check(len(NO_TOOL_BANK) >= 300, f"no-tool bank expanded ({len(NO_TOOL_BANK)} items)")
kinds = {g["kind"] for _, g in NO_TOOL_BANK}
check({"text", "numeric", "all_of"} <= kinds, f"bank covers gold kinds: {sorted(kinds)}")

# ---------------------------------------------------------------- F12 surface variation
prompts = set()
for s in range(40):
    ts = [t for t in generate(24, seed_start=100_000 + s * 97) if t.task_type == "musique_3hop"]
    prompts.update(t.prompt for t in ts)
check(len(prompts) >= 3, f"musique_3hop questions vary across seeds ({len(prompts)} distinct)")
import atr.tools.world as world_mod
check(len(world_mod.NATIONS) >= 6 and len(world_mod.ORG_WORDS) >= 8,
      "world entity pools are substantial (geography + organisations)")

# ---------------------------------------------------------------- F13 strict match
from atr.tasks.schema import match_answer

gold = {"kind": "numeric", "value": 42.0, "tol": 0}
loose = match_answer(gold, "The answer could be 41, 42, or 43.")
strict = match_answer(gold, "The answer could be 41, 42, or 43.", strict=True)
check(loose[0] is True and strict[0] is False, f"shotgun accepted loose ({loose}) rejected strict ({strict})")
ok_s = match_answer(gold, "It is 42.", strict=True)
check(ok_s[0] is True, "clean numeric answer still passes strict")

# ---------------------------------------------------------------- F8b argument normalisation
from atr.tasks.verifiers import _key_args_match as km

check(km({"q": " Berlin "}, {"q": "berlin"}), "arg match: case/space insensitive")
check(km({"limit": 20}, {"limit": "20"}), "arg match: numeric string coerces")
check(km({"where": {"a": 1}}, {"where": {"a": 1.0}}), "arg match: nested dict int/float")
check(not km({"q": "berlin"}, {"query": "berlin"}), "arg match: wrong key still fails")
check(not km({"n": 5}, {"n": "six"}), "arg match: non-numeric string mismatch fails")
check(km({"ids": ["A", "B"]}, {"ids": ["b", "a"]}), "arg match: lists are order-free")

# ---------------------------------------------------------------- F5 truncation flag + penalty
from atr.agent.loop import LoopConfig, Step, Trajectory
from atr.tasks.schema import ScoreCard, Task
from atr.tasks.verifiers import score
from atr.train.reward import RewardConfig, compute_reward, scale_by_efficiency

tr = Trajectory(task_id="t", prompt="p", stop_reason="answered")
tr.steps = [Step(index=0, assistant_text="<think>cut off",
                 thinking="", tool_calls=[], tool_results=[],
                 parse_errors=["unterminated_think"], strict_format=False)]
card = score(Task(task_id="t", seed=0, prompt="p", task_type="no_tool", difficulty=0,
                  gold={"kind": "none", "value": None}), tr)
r_plain, parts_plain = compute_reward(
    Task(task_id="t", seed=0, prompt="p", task_type="no_tool", difficulty=0,
         gold={"kind": "none", "value": None}),
    ScoreCard(task_id="t", task_type="no_tool", difficulty=0))
r_tr, parts_tr = compute_reward(
    Task(task_id="t", seed=0, prompt="p", task_type="no_tool", difficulty=0,
         gold={"kind": "none", "value": None}),
    ScoreCard(task_id="t", task_type="no_tool", difficulty=0,
              detail={"truncated": True}))
check(card.detail["truncated"] is True, "verifier flags unterminated turns as truncated")
check(parts_tr.get("overlong") == -RewardConfig().p_truncated and r_tr < r_plain,
      "overlong soft penalty applied (F5)")
check("overlong" not in parts_plain, "clean episodes carry no overlong term")

# ---------------------------------------------------------------- F7 multiplicative efficiency
base = [1.0, 1.0, 0.9]
calls = [3, 6, 2]
succ = [True, True, False]
out = scale_by_efficiency(base, calls, succ, lam=0.15)
check(out[0] == 1.0, "efficiency: best-caller keeps full reward")
check(abs(out[1] - 1.0 * max(0.2, 1 - 0.15 * 3 / 3)) < 1e-9,
      f"efficiency: slower success scaled down ({out[1]:.4f})")
check(out[2] == 0.9, "efficiency: failures untouched (cannot farm by zero calls)")
check(scale_by_efficiency(base, calls, succ, 0.0) == base, "efficiency: lam=0 disables")
all_fail = scale_by_efficiency([0.1, 0.2], [4, 7], [False, False], 0.15)
check(all_fail == [0.1, 0.2], "efficiency: no successes -> group untouched")

# ---------------------------------------------------------------- F3 GiGPO grouping math
import hashlib


def fake_record(gid, msgs_prefixes, rewards_path):
    """Minimal record the trainer helpers operate on. Assistant content IS the
    action: shared first prefix -> anchored turn 0; diverging action -> turn 1
    states differ, exactly like real episodes."""
    steps = []
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for k, act in enumerate(msgs_prefixes):
        messages.append({"role": "assistant", "content": act})
        steps.append(Step(index=k, assistant_text=act, thinking="",
                          tool_calls=[], tool_results=[], parse_errors=[],
                          strict_format=True))
    traj = Trajectory(task_id=gid, prompt="p", steps=steps, messages=messages,
                      call_log=[], sent_messages=[])
    return {"group": gid.split("#")[0], "void": False,
            "reward": rewards_path, "card": ScoreCard(task_id=gid, task_type="multi_hop", difficulty=3),
            "traj": traj}


from atr.train.grpo import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(steps=100, gigpo=True, gigpo_omega=0.5, efficiency_lambda=0.0)
trainer = GRPOTrainer.__new__(GRPOTrainer)
trainer.cfg = cfg

# two episodes share the turn-0 STATE (same prefix), take different actions there,
# so from turn 1 onward their histories differ; third never repeats anything
ra = fake_record("T#g0", ["shared_obs_act_A", "then_a"], 1.0)
rb = fake_record("T#g1", ["shared_obs_act_B", "then_b"], 0.0)
rc = fake_record("T#g2", ["zzz_act", "yyy_act"], 0.5)
records = [ra, rb, rc]
info = trainer.assign_advantages(records)

h_a0 = hashlib.md5(json.dumps(ra["traj"].messages[:2], sort_keys=True).encode()).hexdigest()
h_b0 = hashlib.md5(json.dumps(rb["traj"].messages[:2], sort_keys=True).encode()).hexdigest()
check(h_a0 == h_b0, "turn-0 prefixes identical -> same anchor state")

ep_a, ep_b = ra["advantage"], rb["advantage"]
ta_a0, ta_b0 = ra["turn_advantages"][0], rb["turn_advantages"][0]
micro_pair = abs(ta_a0 - ep_a) + abs(ta_b0 - ep_b)
check(micro_pair > 0.05, f"shared-state turn got a micro advantage ({micro_pair:.3f})")
check(abs((ta_a0 - ep_a) + (ta_b0 - ep_b)) < 1e-6,
      "micro advantages are group-centred (sum to zero)")
check(abs(ra["turn_advantages"][1] - ep_a) < 1e-9,
      "diverged state keeps pure episode advantage (graceful fallback)")
check(rc["turn_advantages"][0] == rc["advantage"],
      "episode with no repeated states degrades to plain GRPO")
check(info["frac_void_episodes"] == 0.0, "void accounting intact")

# ---------------------------------------------------------------- F4 dynamic sampling
cfg2 = GRPOConfig(steps=100, tasks_per_step=2, batch_multiplier=3, max_gen_batches=3,
                  dynamic_sampling=True, group_size=2)
t2 = GRPOTrainer.__new__(GRPOTrainer)
t2.cfg = cfg2
t2.rng = __import__("random").Random(0)
t2.history = []


class FakeBackend:
    name = "fake"


def fake_rollout(tasks):
    recs = []
    for i, t in enumerate(tasks):
        good = (i % 2 == 0)          # half the groups live, half dead
        r = fake_record(f"{t.task_id}#g0", ["only"], 1.0 if good else 0.0)
        r["group"] = t.task_id
        r["live"] = good
        recs.append(r)
    return recs


t2.rollout = fake_rollout
# assign_advantages is replaced so the test exercises ONLY the collection logic
t2.assign_advantages = lambda recs: {"n_groups": 0, "dead_groups": 0,
                                     "frac_dead_groups": 0.0, "frac_void_episodes": 0.0}
from atr.tasks.generator import generate as _gen
t2.sample_tasks = lambda n=None: _gen(n or 6, seed_start=500_000)
pool, dinfo = t2.collect_batch()
live_gids = {r["group"] for r in pool}
check(len(live_gids) >= min(2, len(live_gids)) and all(r["live"] for r in pool),
      f"dynamic sampling kept only live groups ({len(live_gids)}, discarded={dinfo['discarded_groups']})")
check(dinfo["gen_batches"] <= 3, "gen-batch cap respected")

# ---------------------------------------------------------------- F10 frozen eval config
from atr.eval.harness import EVAL_DEFAULTS, load_eval_config

ec = load_eval_config()
check(ec["repeat_guard"] == 0 and ec["temperature"] == 0.0,
      "frozen eval config defaults mirror official conditions")
tmp_cfg = pathlib.Path(tempfile_dir := __import__("tempfile").mkdtemp()) / "eval.json"
tmp_cfg.write_text(json.dumps({"temperature": 0.35}))
check(load_eval_config(str(tmp_cfg))["temperature"] == 0.35 and
      load_eval_config(str(tmp_cfg))["max_steps"] == EVAL_DEFAULTS["max_steps"],
      "config override merges over defaults")

# ---------------------------------------------------------------- F14 repeat_guard plumbing
check(hasattr(GRPOConfig(), "repeat_guard"), "GRPOConfig exposes repeat_guard knob")

# ---------------------------------------------------------------- F9 sandbox
from atr.tools.registry import ToolError
from atr.tools.sandbox import SessionShim, timed


def _fast():
    return {"ok": True}


def _slow():
    time.sleep(2.0)
    return {}


check(timed(_fast, timeout_s=0.5) == {"ok": True}, "timed(): fast call passes through")
t0 = time.time()
try:
    timed(_slow, timeout_s=0.3, tool_name="web_search")
    raised = False
except ToolError as e:
    raised = (e.kind == "timeout")
check(raised and time.time() - t0 < 1.0,
      f"timed(): hung tool becomes readable timeout payload ({time.time()-t0:.2f}s)")


def _boom():
    raise ValueError("env crashed")


try:
    timed(_boom, timeout_s=1.0)
    crashed = False
except ToolError as e:
    crashed = (e.kind == "internal_error")
check(crashed, "timed(): environment crash -> internal_error payload")

shim = SessionShim()
shim.record("db_query", {"table": "orders"}, {"rows": []}, ok=True)
check(len(shim.call_log) == 1 and shim.call_log[0]["name"] == "db_query",
      "SessionShim records registry-style entries")
check(hasattr(shim, "sent_messages") and shim.sent_messages == [],
      "SessionShim exposes sent_messages for verifiers")

# ---------------------------------------------------------------- dead code removed
import atr.agent.chatml as chatml
import atr.agent.loop as loop_mod
from atr.agent.backends import SamplingParams

check(not hasattr(chatml, "render_upto"), "dead code removed: render_upto")
check(not hasattr(loop_mod, "run_episode"), "dead code removed: run_episode")
check("n" not in SamplingParams().__dict__, "dead knob removed: SamplingParams.n")
src = pathlib.Path(__file__).resolve().parents[1] / "atr" / "train" / "grpo.py"
check("entropy_bonus" not in src.read_text(), "dead knob removed: GRPOConfig.entropy_bonus")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
