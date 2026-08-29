"""
Evaluation harness.

Two jobs:
  1. Produce the headline task-success number.
  2. Produce the per-objective breakdown that tells you what to fix next.

`oracle_backend` is the harness's own unit test. Run it first on any new task
family: if the oracle does not score 1.00, the *verifier* is wrong, and you were
about to spend GPU hours chasing a bug in your reward function.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from ..agent.backends import Backend, MockBackend, SamplingParams
from ..agent.loop import LoopConfig, Trajectory, run_episodes
from ..tasks.schema import ScoreCard, Task
from ..tasks.verifiers import score as score_traj
from ..tools.adapter import get_registry

# ---------------------------------------------------------------------------
# F10 (FIX-2): one frozen evaluation configuration. The GRPO canary, the CLI,
# and the Modal endpoint all read the same file so internal numbers always
# match official scoring conditions. CLI flags override when given.
# ---------------------------------------------------------------------------
EVAL_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "max_steps": 10,
    "repeat_guard": 0,        # official conditions: no environment-side nudges
    "n_per_type": 20,
}

_EVAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.json"


def load_eval_config(path: str | None = None) -> dict:
    p = Path(path) if path else _EVAL_CONFIG_PATH
    cfg = dict(EVAL_DEFAULTS)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def oracle_backend(tasks: Sequence[Task],
                   corrupt: Callable[[str, int, dict], dict | None] | None = None) -> MockBackend:
    """Replays each task's oracle_plan. `corrupt` lets you inject controlled
    failures to calibrate how much each skill is worth end to end, e.g.

        # what does 80% argument accuracy cost us?
        def c(tid, step, call):
            if random.random() < 0.2: call["arguments"] = {}
            return call
    """
    return MockBackend(plans={t.task_id: t.oracle_plan for t in tasks},
                       answers={t.task_id: t.oracle_answer for t in tasks},
                       corrupt=corrupt)


def evaluate(tasks: Sequence[Task], backend: Backend, env: str = "builtin",
             cfg: LoopConfig | None = None, sp: SamplingParams | None = None,
             batch_size: int = 64, progress: bool = True,
             strict_necessity: bool = True, strict_match: bool = False) -> tuple[list[ScoreCard], list[Trajectory]]:
    registry = get_registry(env)
    cards: list[ScoreCard] = []
    trajs: list[Trajectory] = []
    for i in range(0, len(tasks), batch_size):
        chunk = list(tasks[i:i + batch_size])
        if progress:
            print(f"[eval] batch {i // batch_size + 1}/{-(-len(tasks) // batch_size)} "
                  f"({len(chunk)} tasks)", flush=True)
        tj = run_episodes(chunk, registry, backend, cfg, sp)
        trajs.extend(tj)
        cards.extend(score_traj(t, j, strict_necessity=strict_necessity,
                                strict_match=strict_match)
                     for t, j in zip(chunk, tj))
    return cards, trajs


# ---------------------------------------------------------------------------
def _rate(vals: list[bool | None]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(cards: Sequence[ScoreCard]) -> dict:
    by_type: dict[str, list[ScoreCard]] = defaultdict(list)
    by_diff: dict[int, list[ScoreCard]] = defaultdict(list)
    for c in cards:
        by_type[c.task_type].append(c)
        by_diff[c.difficulty].append(c)

    def block(cs: Sequence[ScoreCard]) -> dict:
        return {
            "n": len(cs),
            "success": _rate([c.success for c in cs]),
            "final_correct": _rate([c.final_correct for c in cs]),
            "format_strict": _rate([c.format_strict for c in cs]),
            "format_loose": _rate([c.format_loose for c in cs]),
            "necessity_ok": _rate([c.necessity_ok for c in cs]),
            "selection_ok": _rate([c.selection_ok for c in cs]),
            "args_ok": _rate([c.args_ok for c in cs]),
            "args_strict": round(statistics.fmean(
                [c.detail["args_strict_frac"] for c in cs
                 if c.detail.get("args_strict_frac") is not None]), 4)
            if any(c.detail.get("args_strict_frac") is not None for c in cs) else None,
            "recovery_ok": _rate([c.recovery_ok for c in cs]),
            "side_effect_ok": _rate([c.side_effect_ok for c in cs]),
            "avg_steps": round(statistics.fmean([c.num_steps for c in cs]), 2) if cs else 0,
            "avg_calls": round(statistics.fmean([c.num_calls for c in cs]), 2) if cs else 0,
            "avg_tool_errors": round(statistics.fmean([c.num_tool_errors for c in cs]), 2) if cs else 0,
        }

    fails: dict[str, int] = defaultdict(int)
    for c in cards:
        if not c.success:
            fails[c.failure_mode.split(":")[0]] += 1

    return {
        "overall": block(cards),
        "by_task_type": {k: block(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {str(k): block(v) for k, v in sorted(by_diff.items())},
        "failure_modes": dict(sorted(fails.items(), key=lambda kv: -kv[1])),
        "stop_reasons": dict(sorted(
            ((k, sum(1 for c in cards if c.stop_reason == k))
             for k in {c.stop_reason for c in cards}), key=lambda kv: -kv[1])),
    }


_ROWS = [("success", "TASK SUCCESS"), ("final_correct", "final answer correct"),
         ("format_strict", "format strict"), ("format_loose", "format parseable"),
         ("necessity_ok", "tool-necessity decision"), ("selection_ok", "tool selection"),
         ("args_ok", "arguments usable"), ("args_strict", "arguments match oracle"),
         ("recovery_ok", "recovered after error"), ("side_effect_ok", "side effects")]


def format_report(rep: dict, title: str = "") -> str:
    def pct(v):
        return "  -- " if v is None else f"{100 * v:5.1f}%"

    L = []
    if title:
        L += [title, "=" * len(title)]
    o = rep["overall"]
    L.append(f"n = {o['n']} tasks   avg steps {o['avg_steps']}   avg calls {o['avg_calls']}"
             f"   avg tool errors {o['avg_tool_errors']}")
    L.append("")
    L.append("OVERALL")
    for key, label in _ROWS:
        L.append(f"  {label:<26} {pct(o.get(key))}")

    L.append("")
    L.append(f"{'BY TASK TYPE':<22}{'n':>4}{'success':>9}{'final':>8}{'select':>8}{'args':>7}{'steps':>7}")
    for k, b in rep["by_task_type"].items():
        L.append(f"  {k:<20}{b['n']:>4}{pct(b['success']):>9}{pct(b['final_correct']):>8}"
                 f"{pct(b['selection_ok']):>8}{pct(b['args_ok']):>7}{b['avg_steps']:>7}")

    L.append("")
    L.append(f"{'BY DIFFICULTY':<22}{'n':>4}{'success':>9}")
    for k, b in rep["by_difficulty"].items():
        L.append(f"  d{k:<19}{b['n']:>4}{pct(b['success']):>9}")

    if rep["failure_modes"]:
        L.append("")
        L.append("FAILURE MODES (count)")
        for k, v in rep["failure_modes"].items():
            L.append(f"  {k:<28}{v:>4}")
    L.append("")
    L.append("STOP REASONS: " + ", ".join(f"{k}={v}" for k, v in rep["stop_reasons"].items()))
    return "\n".join(L)


def save_run(out_dir: str | Path, cards: Sequence[ScoreCard], trajs: Sequence[Trajectory],
             rep: dict, meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"meta": meta or {}, **rep}, indent=2))
    (out / "report.txt").write_text(format_report(rep, meta.get("title", "") if meta else ""))
    with (out / "scores.jsonl").open("w") as f:
        for c in cards:
            f.write(json.dumps(c.to_dict()) + "\n")
    with (out / "trajectories.jsonl").open("w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")
    return out
