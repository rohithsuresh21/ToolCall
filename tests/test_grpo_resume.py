"""GRPO durability: checkpoint, resume, wall-clock stop, history integrity.

A fixed GPU reservation must never cost more than `save_every` steps. Before this,
`train()` wrote `save_pretrained` only -- WEIGHTS -- so a run killed at 3h58m lost
the optimiser moments, the step counter, the `best` record and the RNG position,
and there was no way to continue it at all. None of that raises and none of it
appears in history.jsonl: a cold-restarted run just learns worse.

Everything here runs on CPU with a 2-parameter stand-in model. No GPU, no
transformers, no model download -- the point is the trainer's control flow, which
is the part that had never executed.

    python tests/test_grpo_resume.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from atr.tasks.schema import ScoreCard
from atr.train.grpo import GRPOConfig, GRPOTrainer

FAILS = []


def check(cond, label):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


# ---------------------------------------------------------------------------
class FakeSaveable(torch.nn.Module):
    """Stands in for the PEFT model / tokenizer: real parameters, and a
    `save_pretrained` that writes something so the checkpoint dir is realistic."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.lin(x)

    def save_pretrained(self, path):
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.safetensors.stub").write_text("weights")
        (p / "adapter_config.json").write_text(json.dumps({"r": 32}))


class FakeTok:
    def save_pretrained(self, path):
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json.stub").write_text("tok")


def make_trainer(cfg: GRPOConfig, step_cost: float = 0.0) -> GRPOTrainer:
    """A trainer with everything the durability path touches and nothing else.

    Built with __new__ (the pattern test_fix2.py already uses) so no model is
    loaded; the training internals are stubbed so `train()` exercises loop control
    flow -- resume range, history rewrite, wall-clock break, final save -- rather
    than the optimiser.
    """
    import random as _random
    import time as _time

    t = GRPOTrainer.__new__(GRPOTrainer)
    t.cfg = cfg
    t.model = FakeSaveable()
    t.tok = FakeTok()
    t.opt = torch.optim.AdamW(t.model.parameters(), lr=1e-3)
    t.rng = _random.Random(cfg.seed)
    t.history = []
    t.best = None
    t.dead_frac_window = []
    t._last_group_stats = []
    t.start_step = 0

    card = ScoreCard(task_id="x", task_type="musique_2hop", difficulty=2, num_calls=3)
    rec = {"reward": 1.0, "card": card}

    def fake_collect_batch():
        # a real optimiser step, so the moments in trainer_state.pt are non-trivial
        loss = t.model(torch.ones(1, 4)).sum()
        t.opt.zero_grad()
        loss.backward()
        t.opt.step()
        if step_cost:
            _time.sleep(step_cost)
        return [rec], {"sampled_groups": 4, "discarded_groups": 1}

    t.collect_batch = fake_collect_batch
    t.assign_advantages = lambda recs: {"n_groups": 1, "dead_groups": 0,
                                        "frac_dead_groups": 0.0,
                                        "frac_void_episodes": 0.0}
    t.encode = lambda recs: [{"n_action": 1}]
    t.optimise = lambda items: {"loss": 0.1, "n_items": len(items),
                                "clip_frac": 0.0, "grad_norm": 1.0, "kl": 0.0}
    t.run_dev_canary = lambda: {"dev_f1": 0.5, "dev_success": 0.5, "dev_necessity": 1.0}
    return t


def hist_steps(out: pathlib.Path) -> list[int]:
    rows = [json.loads(l) for l in
            (out / "history.jsonl").read_text().splitlines() if l.strip()]
    return [r["step"] for r in rows]


TMP = pathlib.Path(tempfile.mkdtemp(prefix="atr_resume_"))

# ---------------------------------------------------------------- 1. save
out1 = TMP / "run1"
cfg1 = GRPOConfig(out_dir=str(out1), steps=4, save_every=2, eval_every=0,
                  kl_beta=0.0, curriculum=False)
t1 = make_trainer(cfg1)
t1.train()

check((out1 / "final" / "trainer_state.pt").is_file(),
      "final/ carries trainer_state.pt, not just weights")
check((out1 / "step-2" / "trainer_state.pt").is_file(),
      "periodic step-N/ checkpoints are full resume points too")
st = torch.load(out1 / "final" / "trainer_state.pt", map_location="cpu",
                weights_only=False)
check(st["step"] == 4, f"saved step counter is the last COMPLETED step (got {st['step']})")
check("optimizer" in st and st["optimizer"]["state"],
      "optimiser state is non-empty (AdamW moments actually captured)")
check(len(st["history"]) == 4, f"history rows travel with the checkpoint ({len(st['history'])})")
check("rng_python" in st and "rng_torch" in st, "RNG states are captured")
check(hist_steps(out1) == [1, 2, 3, 4], f"fresh run logs steps 1..4 ({hist_steps(out1)})")

# ---------------------------------------------------------------- 2. restore
out2 = TMP / "run2"
cfg2 = GRPOConfig(out_dir=str(out2), steps=4, save_every=2, eval_every=0,
                  kl_beta=0.0, curriculum=False,
                  resume_from=str(out1 / "step-2"))
t2 = make_trainer(cfg2)
t2._restore_trainer_state(str(out1 / "step-2"))
check(t2.start_step == 2, f"resume restores the step counter (got {t2.start_step})")
check(len(t2.history) == 2, f"resume restores history rows (got {len(t2.history)})")

ref = torch.load(out1 / "step-2" / "trainer_state.pt", map_location="cpu",
                 weights_only=False)["optimizer"]
got = t2.opt.state_dict()
same = all(torch.allclose(got["state"][k]["exp_avg"], ref["state"][k]["exp_avg"])
           for k in ref["state"])
check(bool(ref["state"]) and same,
      "resume restores AdamW exp_avg moments exactly (not a cold restart)")

# RNG position must continue, or the resumed run re-draws seeds it already trained on
r_saved = torch.load(out1 / "step-2" / "trainer_state.pt", map_location="cpu",
                     weights_only=False)["rng_python"]
import random as _r
probe = _r.Random(); probe.setstate(r_saved)
check(probe.randrange(1 << 30) == t2.rng.randrange(1 << 30),
      "resume restores the task-sampling RNG position")

# ---------------------------------------------------------------- 3. refuse weights-only
weights_only_dir = TMP / "weights_only"
weights_only_dir.mkdir()
FakeSaveable().save_pretrained(weights_only_dir)
t3 = make_trainer(GRPOConfig(out_dir=str(TMP / "run3"), kl_beta=0.0))
try:
    t3._restore_trainer_state(str(weights_only_dir))
    refused, msg = False, ""
except FileNotFoundError as e:
    refused, msg = True, str(e)
check(refused, "a weights-only directory is REFUSED, never cold-started silently")
check("--adapter" in msg, "the refusal names the flag that does start fresh from weights")

# ---------------------------------------------------------------- 4. no duplicate steps
# Simulate the real crash shape: the previous session checkpointed at step 2 but
# kept logging to step 4 before it died. Those tail rows must not survive a resume,
# or history.jsonl carries two rows numbered 3 and two numbered 4.
out4 = TMP / "run4"
out4.mkdir(parents=True, exist_ok=True)
with (out4 / "history.jsonl").open("w") as f:
    for i in range(1, 5):
        f.write(json.dumps({"step": i, "reward_mean": 0.1 * i}) + "\n")
cfg4 = GRPOConfig(out_dir=str(out4), steps=5, save_every=10, eval_every=0,
                  kl_beta=0.0, curriculum=False, resume_from=str(out1 / "step-2"))
t4 = make_trainer(cfg4)
t4._restore_trainer_state(cfg4.resume_from)
t4.train()
steps4 = hist_steps(out4)
check(steps4 == [1, 2, 3, 4, 5], f"resumed history is contiguous 1..5 (got {steps4})")
check(len(steps4) == len(set(steps4)), f"no duplicate step numbers after resume ({steps4})")

# ---------------------------------------------------------------- 5. wall-clock stop
out5 = TMP / "run5"
cfg5 = GRPOConfig(out_dir=str(out5), steps=50, save_every=100, eval_every=0,
                  kl_beta=0.0, curriculum=False, max_seconds=2)
t5 = make_trainer(cfg5, step_cost=0.4)
t5.train()
steps5 = hist_steps(out5)
check(0 < len(steps5) < 50,
      f"wall-clock budget stops the run short of --steps ({len(steps5)}/50 steps)")
check((out5 / "final" / "trainer_state.pt").is_file(),
      "a wall-clock stop still writes a full resume point")
st5 = torch.load(out5 / "final" / "trainer_state.pt", map_location="cpu",
                 weights_only=False)
check(st5["step"] == steps5[-1],
      f"the saved step matches the last logged step ({st5['step']} vs {steps5[-1]})")
check(len(st5["history"]) == len(steps5), "the stopped run's history is complete")

# and the stopped run continues cleanly
cfg6 = GRPOConfig(out_dir=str(out5), steps=50, save_every=100, eval_every=0,
                  kl_beta=0.0, curriculum=False, max_seconds=2,
                  resume_from=str(out5 / "final"))
t6 = make_trainer(cfg6, step_cost=0.4)
t6._restore_trainer_state(cfg6.resume_from)
t6.train()
steps6 = hist_steps(out5)
check(steps6 == sorted(set(steps6)) and steps6[0] == 1,
      f"the continued run's history stays strictly increasing from 1 ({steps6})")
check(len(steps6) > len(steps5),
      f"the continued run made further progress ({len(steps5)} -> {len(steps6)} steps)")

# ---------------------------------------------------------------- 6. off by default
check(GRPOConfig().max_seconds == 0 and GRPOConfig().resume_from is None,
      "both knobs default to off, so an unflagged run behaves exactly as before")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
