"""
Reward shaping for multi-turn RL.

The reward is dominated by the outcome (`success`). Everything else is small and
exists only to break ties *inside a group* -- which is all that matters, because
GRPO normalises advantages within the group of G rollouts for one task. A shaping
term that fires identically on all G rollouts contributes exactly nothing, so
shaping should only encode differences you actually want the model to notice.

Deliberate choices worth arguing about:

* The outcome term is token-level F1 (`card.final_f1`), not exact match, because
  that is what the official judge scores. Optimising normalised-exact-match while
  being graded on F1 trains for the wrong target, and the gap is not cosmetic:
  a correct-but-padded answer is full marks under exact match and mediocre under
  F1. F1 is also continuous, so it gives partial credit even when the task fails.
  On a 4-hop task a group where every rollout scores 0 gives zero gradient -- the
  classic GRPO dead group -- and partial credit keeps hard tasks learnable
  instead of silently dropping out of training.
* There is no separate verbosity penalty. F1 precision already charges for every
  token the answer carries beyond the gold span, continuously and in the units
  the judge uses; a second length term on top double-counts the same signal and
  lets a hand-tuned character threshold overrule the real metric.
* `p_no_answer` stays. Pure-F1 objectives are known to collapse into
  answer-avoidance -- the safest way to protect precision is to say nothing --
  and that penalty is the mitigation, so it must not be dropped as redundant.
* Redundant calls are penalised per-call, not as a threshold. A threshold makes
  the penalty invisible until it trips; per-call gives a smooth slope toward
  shorter trajectories. This additive term applies to FAILED episodes; on
  successful ones efficiency is handled multiplicatively by `scale_by_efficiency`
  (F7) -- OTC-PO showed the additive form is gameable while the multiplicative
  form cut redundant calls sharply at equal accuracy.
* Truncated episodes (generation hit the token cap mid-stream, F5) take a soft
  penalty AND are masked out of the gradient in grpo.encode -- DAPO's overlong
  shaping: a rambling answer must never be rewarded just because one number in
  it matched by luck.
* Unrequested side effects are penalised hard and asymmetrically. Getting the
  answer right after emailing a stranger is not a partial success.
* There is no reward for *using* a tool. Reward tool use and you will get tool
  use on the no-tool tasks, which is the exact failure we are trying to remove.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..tasks.schema import ScoreCard, Task


@dataclass
class RewardConfig:
    w_success: float = 1.0
    w_final_correct: float = 0.30      # scales card.final_f1 (the judge metric);
                                       # continuous, so hard groups stay alive
    w_selection: float = 0.10
    w_args: float = 0.10
    w_args_strict: float = 0.05
    w_format_strict: float = 0.05
    w_recovery: float = 0.10
    p_side_effect: float = 0.50        # unrequested write
    p_necessity: float = 0.20          # used tools on a no-tool task, or vice versa
    p_per_extra_call: float = 0.04     # failed episodes only (see scale_by_efficiency)
    p_extra_call_cap: float = 0.24
    p_no_answer: float = 0.20          # anti-abstention guard; see module docstring
    p_truncated: float = 0.15          # F5: generation was cut off mid-stream
    # Per-hop chain shaping for the multi-hop collapse (GRPO only). These dense
    # terms exist so a trajectory that does hops 1-2 correctly but trips on hop 3
    # still earns credit, instead of the all-or-nothing final-answer signal.
    # Anti-hack (review): `w_reformulate` only pays when the query ADVANCED to the
    # next oracle-chain entity (progress > 0), never for spraying different strings.
    w_anchor: float = 0.15
    w_progress: float = 0.20
    w_reformulate: float = 0.10
    clip_low: float = -1.0
    # Max positive sum is 1.00+0.30+0.10+0.10+0.05+0.05+0.10 = 1.70 from the
    # outcome/selection/args/format/recovery terms, plus 0.45 from the per-hop
    # chain shaping (w_anchor+w_progress+w_reformulate) = 2.15. The ceiling must
    # clear this so a perfect episode never clips, which would silently erase the
    # small shaping terms inside a group (w_recovery AND the hop terms).
    clip_high: float = 2.3


def compute_reward(task: Task, card: ScoreCard, cfg: RewardConfig | None = None) -> tuple[float, dict]:
    cfg = cfg or RewardConfig()
    parts: dict[str, float] = {}

    parts["success"] = cfg.w_success * float(card.success)
    parts["final_correct"] = cfg.w_final_correct * float(card.final_f1)
    parts["selection"] = cfg.w_selection * float(card.selection_ok)
    parts["args"] = cfg.w_args * float(card.args_ok)
    asf = card.detail.get("args_strict_frac")
    parts["args_strict"] = cfg.w_args_strict * float(asf) if asf is not None else 0.0
    parts["format"] = cfg.w_format_strict * float(card.format_strict)
    parts["recovery"] = cfg.w_recovery * (1.0 if card.recovery_ok is True else 0.0)

    # Per-hop chain shaping. `reformulate` is gated on genuine forward progress so
    # a policy that fires three random entity names cannot collect it for "moving
    # to a different string" -- it must actually advance along the oracle chain.
    anchor_ok = card.detail.get("anchor_ok")
    if anchor_ok is not None:
        parts["anchor"] = cfg.w_anchor * float(anchor_ok)
        progress = float(card.detail.get("progress_frac") or 0.0)
        parts["progress"] = cfg.w_progress * progress
        if progress > 0 and card.detail.get("reformulate_ok"):
            parts["reformulate"] = cfg.w_reformulate * float(card.detail.get("reformulate_frac") or 0.0)

    if card.side_effect_ok is False and not task.expect_side_effect:
        parts["side_effect"] = -cfg.p_side_effect
    if not card.necessity_ok:
        parts["necessity"] = -cfg.p_necessity
    budget = max(1, len(task.oracle_plan))
    extra = max(0, card.num_calls - budget)
    # Additive per-call penalty applies to FAILED episodes only. On successful
    # episodes tool-efficiency is already handled multiplicatively by
    # scale_by_efficiency (F7) in grpo.assign_advantages -- charging the flat
    # additive term on top of that would double-count the same signal, and it
    # would do so asymmetrically (additive vs group-relative), biasing the
    # policy against calling tools even on tasks that legitimately need more.
    if extra and not card.success:
        parts["extra_calls"] = -min(cfg.p_extra_call_cap, extra * cfg.p_per_extra_call)
    if card.detail.get("answer_reason") == "no_answer":
        parts["no_answer"] = -cfg.p_no_answer
    if card.detail.get("truncated"):
        parts["overlong"] = -cfg.p_truncated

    total = sum(parts.values())
    total = max(cfg.clip_low, min(cfg.clip_high, total))
    return total, parts


def scale_by_efficiency(rewards: list[float], num_calls: list[int],
                        successes: list[bool], lam: float,
                        floor: float = 0.2) -> list[float]:
    """
    F7 multiplicative tool-efficiency (OTC-PO style), applied per GROUP.

    Among the group's SUCCESSFUL episodes, find the fewest calls `best`; every
    successful episode is scaled by max(floor, 1 - lam*(calls-best)/max(1,best)).
    Failed episodes pass through unchanged (their additive extra_calls penalty
    already pushes toward brevity). Multiplying only when correct means the
    coefficient can never be farmed by refusing to call tools.

    lam=0 disables entirely (returns inputs untouched).
    """
    if lam <= 0 or not rewards:
        return list(rewards)
    best = min((c for c, s in zip(num_calls, successes) if s), default=None)
    if best is None:
        return list(rewards)
    denom = max(1, best)
    out = []
    for r, c, s in zip(rewards, num_calls, successes):
        if s:
            out.append(r * max(floor, 1.0 - lam * (c - best) / denom))
        else:
            out.append(r)
    return out


def group_advantages(rewards: list[float], eps: float = 1e-4,
                     std_normalise: bool = True) -> list[float]:
    """
    Group-relative advantage. With `std_normalise=False` you get Dr.GRPO-style
    mean-only centring, which avoids the known bias where low-variance groups get
    their gradients blown up. On small models that bias shows up as the policy
    over-fitting whichever task family happens to be nearly-solved.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    centred = [r - mean for r in rewards]
    if not std_normalise:
        return centred
    var = sum(c * c for c in centred) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n          # dead group: no signal, contribute nothing
    return [c / (std + eps) for c in centred]
