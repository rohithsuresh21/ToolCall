"""
Rejection sampling and data hygiene.

Filtering on `success` alone is the mistake everyone makes first. A trajectory
can be successful and still be poison:

  * it flailed through six redundant calls before stumbling onto the answer --
    train on it and you teach flailing;
  * it repeated the same call verbatim -- you teach loops, which is the single
    most common way a small model burns its step budget;
  * it fired a side effect nobody asked for and got lucky on the text answer;
  * it is the 400th near-identical `db_lookup` -- you teach the easy family and
    starve the hard one.

Equally, do NOT filter out trajectories that hit tool errors. A trajectory that
errored and then recovered is the single most valuable kind of example you can
train on, and it is exactly what naive success-filtering under-samples, because
error paths are longer and rarer. `keep_recoveries` protects them.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from ..agent.loop import Trajectory
from ..tasks.schema import ScoreCard, Task


@dataclass
class FilterConfig:
    require_success: bool = True
    require_strict_format: bool = False     # keep loose early; tighten once format is learned
    max_call_bloat: float = 2.0             # calls <= oracle_calls * this (+1 slack)
    forbid_repeat_calls: bool = True
    forbid_unrequested_side_effects: bool = True
    max_answer_chars: int = 600
    max_per_task: int = 1                   # best-of-N per task -> keeps diversity honest
    dedupe_by_shape: bool = True            # cap identical (type, tool-sequence) shapes
    max_per_shape: int = 400
    keep_recoveries: bool = True            # never drop a recovered-from-error trajectory for bloat
    target_mix: dict[str, float] | None = None
    seed: int = 0


def _shape(task: Task, traj: Trajectory) -> str:
    sig = f"{task.task_type}|{'>'.join(traj.tool_names)}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def _has_repeat(traj: Trajectory) -> bool:
    seen = set()
    for c in traj.call_log:
        k = f"{c['name']}:{json.dumps(c.get('args'), sort_keys=True, default=str)}"
        if k in seen:
            return True
        seen.add(k)
    return False


def _quality(task: Task, traj: Trajectory, card: ScoreCard, cfg: FilterConfig) -> str | None:
    """Return a rejection reason, or None to keep."""
    if cfg.require_success and not card.success:
        return "not_success"
    if cfg.require_strict_format and not card.format_strict:
        return "loose_format"
    if cfg.forbid_repeat_calls and _has_repeat(traj):
        return "repeated_call"
    if cfg.forbid_unrequested_side_effects and card.detail.get("unrequested_side_effect"):
        return "unrequested_side_effect"
    if traj.final_answer and len(traj.final_answer) > cfg.max_answer_chars:
        return "verbose_answer"
    recovered = card.recovery_ok is True
    if not (recovered and cfg.keep_recoveries):
        budget = max(1, task.oracle_plan and len(task.oracle_plan) or 1)
        if card.num_calls > budget * cfg.max_call_bloat + 1:
            return "call_bloat"
    return None


def filter_and_balance(records: Sequence[tuple[Task, Trajectory, ScoreCard]],
                       cfg: FilterConfig | None = None
                       ) -> tuple[list[tuple[Task, Trajectory, ScoreCard]], dict]:
    cfg = cfg or FilterConfig()
    rng = random.Random(cfg.seed)
    stats: Counter = Counter()

    kept_raw: list[tuple[Task, Trajectory, ScoreCard]] = []
    for t, j, c in records:
        reason = _quality(t, j, c, cfg)
        if reason:
            stats[f"drop:{reason}"] += 1
            continue
        kept_raw.append((t, j, c))
        stats["pass:quality"] += 1

    # --- best-of-N per task: prefer fewest calls, then recovery, then shortest text
    by_task: dict[str, list[tuple[Task, Trajectory, ScoreCard]]] = defaultdict(list)
    for t, j, c in kept_raw:
        by_task[t.task_id.split("#")[0]].append((t, j, c))
    picked: list[tuple[Task, Trajectory, ScoreCard]] = []
    for tid, group in by_task.items():
        group.sort(key=lambda r: (r[2].num_calls, -(r[2].recovery_ok is True),
                                  len(r[1].final_answer or "")))
        take = group[: cfg.max_per_task]
        stats["drop:extra_sample"] += len(group) - len(take)
        picked.extend(take)

    # --- shape cap: stops one trivial family dominating the gradient
    if cfg.dedupe_by_shape:
        counts: Counter = Counter()
        out = []
        rng.shuffle(picked)
        for t, j, c in picked:
            s = _shape(t, j)
            if counts[s] >= cfg.max_per_shape:
                stats["drop:shape_cap"] += 1
                continue
            counts[s] += 1
            out.append((t, j, c))
        picked = out

    # --- optional re-balance toward a target task-type mix
    if cfg.target_mix:
        by_type: dict[str, list] = defaultdict(list)
        for r in picked:
            by_type[r[0].task_type].append(r)
        # size the whole set by the scarcest type relative to its target share
        feasible = min((len(v) / cfg.target_mix[k]) for k, v in by_type.items()
                       if cfg.target_mix.get(k))
        out = []
        for k, v in by_type.items():
            want = int(feasible * cfg.target_mix.get(k, 0))
            rng.shuffle(v)
            out.extend(v[:want])
            stats[f"drop:mix_{k}"] += max(0, len(v) - want)
        picked = out
        rng.shuffle(picked)

    stats["kept"] = len(picked)
    summary = {
        "in": len(records), "out": len(picked),
        "keep_rate": round(len(picked) / max(1, len(records)), 4),
        "by_type": dict(Counter(t.task_type for t, _, _ in picked)),
        "by_difficulty": dict(Counter(t.difficulty for t, _, _ in picked)),
        "with_recovery": sum(1 for _, _, c in picked if c.recovery_ok is True),
        "avg_calls": round(sum(c.num_calls for _, _, c in picked) / max(1, len(picked)), 2),
        "reasons": dict(stats),
    }
    return picked, summary
