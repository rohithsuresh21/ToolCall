"""Curriculum-feedback verification: frac_dead_groups -> _stage_mix close of the
loop. CPU-only, no GPU/model. Mirrors the plain PASS/FAIL style of test_fix2."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
FAILS = []


def check(cond, label):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


from atr.train.grpo import GRPOConfig, GRPOTrainer


def make_trainer(**overrides):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.cfg = GRPOConfig(**overrides)
    t.history = []
    t.dead_frac_window = []
    return t


STEPS = [1, 5, 15, 40, 75, 100, 130, 180, 220, 290, 300]

# ---------------------------------------------------------------- defaults wiring
cfg0 = GRPOConfig()
check(hasattr(cfg0, "curriculum_feedback") and cfg0.curriculum_feedback is True,
      "GRPOConfig.curriculum_feedback exists and defaults ON")
check(hasattr(cfg0, "dead_group_window") and cfg0.dead_group_window == 5,
      "GRPOConfig.dead_group_window exists and defaults to 5")
check(hasattr(cfg0, "dead_group_threshold") and cfg0.dead_group_threshold == 0.5,
      "GRPOConfig.dead_group_threshold exists and matches README advisory (0.5)")

# ---------------------------------------------------------------- OFF == step-only
# With curriculum_feedback=False the mix must equal the pure step-only schedule
# across a range of step values (existing behaviour provably unchanged).
t_off = make_trainer(curriculum_feedback=False)
t_off_hi = make_trainer(curriculum_feedback=False)
t_off_hi.dead_frac_window = [0.9, 0.9, 0.9]   # even high dead fraction must NOT matter
off_eq_baseline = True
for s in STEPS:
    if t_off._stage_mix(s) != t_off_hi._stage_mix(s):
        off_eq_baseline = False
        break
check(off_eq_baseline, "curriculum_feedback OFF: dead fraction never affects the mix")

# ---------------------------------------------------------------- ON + low dead == step-only
# Below the threshold the feedback is inert: output matches the step-only baseline.
t_low = make_trainer(curriculum_feedback=True)
t_low.dead_frac_window = [0.0, 0.1, 0.0, 0.05]   # all well under 0.5
low_eq_baseline = True
for s in STEPS:
    if t_low._stage_mix(s) != t_off._stage_mix(s):
        low_eq_baseline = False
        break
check(low_eq_baseline, "curriculum_feedback ON + low dead fraction: matches step-only baseline")

# ---------------------------------------------------------------- ON + high dead pulls to EASY
# Extreme overshoot => pullback=1.0 => effective f=0 => mix snapped to the
# CURRICULUM_EASY values on the curriculum key set (strictly easier: lower
# hard-task weight than the step-only baseline).
t_hi = make_trainer(curriculum_feedback=True)
t_hi.dead_frac_window = [1.0, 1.0]              # mean 1.0 -> full pull-back
EASY = dict(GRPOTrainer.CURRICULUM_EASY)
CUR_KEYS = set(EASY)


def easy_aligned(m):
    # On the curriculum key set, the pulled mix must match CURRICULUM_EASY values
    # (tolerating zero-valued extra keys that _lerp_mix with task_mix injects).
    for k in CUR_KEYS:
        if abs(m.get(k, 0.0) - EASY[k]) > 1e-9:
            return False
    return True


# Difficulty under a hop-only curriculum is CHAIN LENGTH, so the "how hard is this
# mix" scalar is its weight-averaged hop count. The previous version of this summed
# multi_hop + multi_hop_discount -- families deleted in the 8->5 tool reduction, so
# it read 0.0 for every mix and the comparison below was 0.0 <= 0.0: green whatever
# the curriculum did. Guard the key set so it fails loudly if it empties again.
HOPS = {"musique_2hop": 2, "musique_3hop": 3, "musique_4hop": 4}
check(CUR_KEYS and CUR_KEYS <= set(HOPS),
      f"curriculum families are the hop-only set (got {sorted(CUR_KEYS)})")


def hard_weight(m):
    """Weight-averaged hop count: higher = harder. Non-vacuous by construction --
    CURRICULUM_EASY scores 2.5, task_mix 2.9, CURRICULUM_HARD 3.25."""
    return sum(m.get(k, 0.0) * h for k, h in HOPS.items())


pulled_to_easy = True
pulled_hard_zone_easier = True
hard_zone_tested = False
for s in STEPS:
    pulled = t_hi._stage_mix(s)
    f_base = t_off._curriculum_fraction(s)
    # Only the step-only schedule's HARD zone (f>0.70 ramp toward CURRICULUM_HARD)
    # is meaningful to "pull back from" -- in the easy/default zones the baseline is
    # already at CURRICULUM_EASY-or-default, so hard_weight is only a valid signal
    # once the curriculum has started ramping toward CURRICULUM_HARD. (task_mix and
    # the curriculum tiers now share the musique_* key namespace.)
    if f_base > 0.70:
        hard_zone_tested = True
        if not (hard_weight(pulled) <= hard_weight(t_off._stage_mix(s))):
            pulled_hard_zone_easier = False
    if not easy_aligned(pulled):
        pulled_to_easy = False
check(pulled_to_easy, "curriculum_feedback ON + high dead fraction: mix matches CURRICULUM_EASY values")
check(hard_zone_tested and pulled_hard_zone_easier,
      "curriculum_feedback ON + high dead fraction: in the hard zone, hard-task weight <= step-only baseline")

# ---------------------------------------------------------------- partial pullback is graded
# A moderate overshoot (mean 0.75) yields a partial pull-back: easier than baseline
# but not as easy as the full pull-back (mean 1.0). Confirms proportionality.
t_mid = make_trainer(curriculum_feedback=True)
t_mid.dead_frac_window = [0.75]                  # mean 0.75 -> overshoot 0.5 -> pullback 0.5
s_hard = 250                                     # step-only fraction ~0.83 (deep in hard zone)
f_base = t_off._curriculum_fraction(s_hard)
f_mid = t_mid._curriculum_fraction(s_hard)
f_hi = t_hi._curriculum_fraction(s_hard)
check(f_mid > 0 and f_mid < f_base,
      f"partial pullback: mid overshoot fraction {f_mid:.3f} in (0, baseline {f_base:.3f})")
check(f_hi == 0.0, f"full pullback drives effective fraction to 0 (got {f_hi})")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
