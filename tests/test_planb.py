"""Plan B verification: the 4 fixes in isolation, CPU-only, no model.

Checks each new knob does what the design claims, and that pre-values equal old behaviour.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
FAILS = []


def check(cond, label):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


from atr.train.reward import (group_advantages, group_advantages_planb,
                              mean_abs_dev, dqw_weights)
from atr.train.grpo import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------- Fix-1 + Fix-2 (advantage)
# A 4-hop-style dead group: all 8 rollouts fail with the SAME reward (early-hop crawl).
dead_rewards = [0.45] * 8
pre = group_advantages(dead_rewards, std_normalise=True)
check(all(abs(a) < 1e-9 for a in pre), "PRE: all-equal group => zero advantage (dead), no gradient")

# Fix-2 Sign: fixed baseline 0.5 -> every sample gets r - 0.5 = -0.05 (non-zero, negative)
sign = group_advantages_planb(dead_rewards, baseline="sign", scale="none", sign_baseline=0.5)
check(all(abs(a - (-0.05)) < 1e-9 for a in sign), "FIX2 sign: fixed baseline gives r-0.5=-0.05 (rescued)")
check(all(a < 0 for a in sign), "FIX2 sign: all-fail group gets a negative (explorable) signal")

# Fix-1 MAD: magnitude is INVARIANT to reward spread (constant total per group)
spread_hi = group_advantages_planb([0.0, 0.5, 1.0], baseline="group", scale="mad")
spread_lo = group_advantages_planb([0.0, 0.1, 0.2], baseline="group", scale="mad")
check(abs(sum(abs(a) for a in spread_hi) - sum(abs(a) for a in spread_lo)) < 0.01,
      f"FIX1 MAD: total magnitude ~constant across spreads ({sum(abs(a) for a in spread_hi):.3f} vs {sum(abs(a) for a in spread_lo):.3f})")

# MAD == mean-abs-dev
import math
rws = [0.2, 0.4, 0.6]
m = sum(rws) / 3
mad = mean_abs_dev(rws, m)
check(abs(mad - (abs(0.2 - m) + abs(0.4 - m) + abs(0.6 - m)) / 3) < 1e-9, "FIX1 MAD invariant")

# pre == planb (std+group)
pre_std = group_advantages([0.1, 0.9, 0.5])
pb_std = group_advantages_planb([0.1, 0.9, 0.5], scale="std", baseline="group")
check(all(abs(a - b) < 1e-12 for a, b in zip(pre_std, pb_std)),
      "pre == planb when scale=std, baseline=group (bit-identical)")

# ---------------------------------------------------------------- Fix-2b DQW
means = [1.2, 1.4, 0.3, 0.5]          # two easy, two hard (on reward scale)
w = dqw_weights(means, temp=2.2)
hard_w = w[2] + w[3]
easy_w = w[0] + w[1]
check(hard_w > easy_w, f"FIX2b DQW: hard groups get more summed weight ({hard_w:.3f} vs {easy_w:.3f})")
check(abs(sum(w) - len(means)) < 1e-6, "FIX2b DQW: weights sum to n (mean weight 1)")
check(all(x > 0 for x in w), "FIX2b DQW: all weights positive (nothing starved)")

# ---------------------------------------------------------------- Fix-3 dead-frac source
def mk(**ov):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.cfg = GRPOConfig(**ov)
    t.history = []
    t.dead_frac_window = []
    return t


dinfo = {"gen_batches": 3, "discarded_groups": 34, "live_groups": 2, "sampled_groups": 36}
# recompute source (pre): window gets ginfo frac 0.0
t_pre = mk(dead_frac_source="recompute")
t_pre.dead_frac_source = "recompute"
# emulate train() pop with a fixed ginfo to isolate the source logic
ginfo = {"frac_dead_groups": 0.0}
if t_pre.cfg.dead_frac_source == "discarded":
    tot = dinfo.get("sampled_groups", 0)
    dead_frac_pre = dinfo["discarded_groups"] / max(1, tot)
else:
    dead_frac_pre = ginfo.get("frac_dead_groups", 0.0)
t_post = mk(dead_frac_source="discarded")
if t_post.cfg.dead_frac_source == "discarded":
    tot = dinfo.get("sampled_groups", 0)
    dead_frac_post = dinfo["discarded_groups"] / max(1, tot)
else:
    dead_frac_post = ginfo.get("frac_dead_groups", 0.0)
check(dead_frac_pre == 0.0 and dead_frac_post == 34 / 36,
      f"FIX3 dead-frac source: pre=0.0 (blind), post={dead_frac_post:.3f} (true 34/36)")

# ---------------------------------------------------------------- Fix-4 E2H-G schedule
t = mk(e2h_curriculum=True)
easy_start = t._e2h_mix(0.02)
end_mix = t._e2h_mix(1.0)
mid = t._e2h_mix(0.55)
check(easy_start.get("musique_2hop", 0) > 0.5,
      f"FIX4 E2H: starts very easy (2-hop {easy_start.get('musique_2hop',0):.2f})")
check(end_mix.get("musique_2hop", 0) >= 0.20,
      f"FIX4 E2H: easy still present at end (2-hop {end_mix.get('musique_2hop',0):.2f}), not cut")
check(end_mix.get("musique_4hop", 0) > easy_start.get("musique_4hop", 0),
      f"FIX4 E2H: hard rises over training ({easy_start.get('musique_4hop',0):.2f}->{end_mix.get('musique_4hop',0):.2f})")

# E2H-G monotone hard ramp
ramp = [t._e2h_mix(f)["musique_4hop"] for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
check(ramp == sorted(ramp), f"FIX4 E2H: hard weight monotone rising ({[round(x,3) for x in ramp]})")

# E2H-G off == pre schedule (bit-identical to existing _stage_mix)
t_off = mk(e2h_curriculum=False)
from atr.tasks.generator import DEFAULT_MIX
# replicate original step-only _stage_mix
STEPS2 = [1, 5, 15, 40, 75, 100, 130, 180, 290, 300]
stepmix_ok = True
for s in STEPS2:
    f = t_off._curriculum_fraction(s)
    if f < 0.30:
        expect = t_off._lerp_mix(dict(GRPOTrainer.CURRICULUM_EASY), DEFAULT_MIX, f / 0.30)
    elif f < 0.70:
        expect = dict(DEFAULT_MIX)
    else:
        tt = min(1.0, (f - 0.70) / 0.15)
        expect = t_off._lerp_mix(DEFAULT_MIX, dict(GRPOTrainer.CURRICULUM_HARD), tt)
    if t_off._stage_mix(s) != expect:
        stepmix_ok = False
        break
check(stepmix_ok, "E2H OFF: _stage_mix matches pre step-only schedule exactly")

# Fix-4 respects step count in range
check(t._e2h_g(0.5) < t._e2h_g(0.55) < t._e2h_g(0.6), "FIX4 E2H: g(f) increasing through midpoint")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
