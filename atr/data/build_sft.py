"""
Trajectory -> SFT records.

The one decision in here that moves the number more than anything else is
CANONICALISATION. Whatever the teacher emitted -- markdown fences, `parameters`
instead of `arguments`, prose wrapped around the call -- the exported assistant
turn is rewritten into exactly one form:

    <think>...</think>            (optional, and short)
    <tool_call>{"name": ..., "arguments": {...}}</tool_call>

or

    <final_answer>...</final_answer>

Two reasons. First, a 1.7B model spends real capacity modelling format variance
it will never be rewarded for; removing that variance is free accuracy. Second,
it makes `format_strict` at eval time an honest measurement rather than a
measurement of your parser's leniency.

The reasoning text is *kept* (that is the part the student must learn) but
truncated: long teacher rationales teach a small model to ramble, and rambling
eats the step budget that multi-hop tasks need.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..agent.loop import Trajectory
from ..agent.prompt import system_message
from ..tasks.schema import ScoreCard, Task
from ..tools.adapter import get_registry


@dataclass
class ExportConfig:
    keep_thinking: bool = True
    max_thinking_chars: int = 400
    oracle_rationale: bool = False   # template a one-line rationale for oracle replays
    prompt_mode: str = "native"      # must match what you serve with
    include_tools_field: bool = True


_SENT = re.compile(r"(?<=[.!?])\s+")


def _shorten(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    parts = _SENT.split(text)
    out = ""
    for p in parts:
        if len(out) + len(p) + 1 > limit:
            break
        out += (" " if out else "") + p
    return out or text[:limit].rsplit(" ", 1)[0]


def _rationale(step_idx: int, call: dict, task: Task) -> str:
    """Minimal templated reasoning for oracle replays. OFF by default.

    Templated rationales are a bootstrap, not a diet: they teach the shape of
    'think then act', but a model trained only on them learns a script and stops
    adapting when a tool result surprises it. Mix at most ~30% templated.
    """
    return f"I need {call['name']} to get the next piece of information."


def canonical_assistant_turn(thinking: str, call: dict | None, answer: str | None,
                             cfg: ExportConfig) -> str:
    parts = []
    if cfg.keep_thinking and thinking:
        parts.append(f"<think>{_shorten(thinking, cfg.max_thinking_chars)}</think>")
    if call is not None:
        payload = {"name": call["name"], "arguments": call.get("arguments", {})}
        parts.append("<tool_call>" + json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
                     + "</tool_call>")
    if answer is not None:
        parts.append(f"<final_answer>{answer.strip()}</final_answer>")
    return "\n".join(parts)


def trajectory_to_record(task: Task, traj: Trajectory, cfg: ExportConfig,
                         env: str = "builtin") -> dict | None:
    registry = get_registry(env)
    messages = [{"role": "system", "content": system_message(registry, cfg.prompt_mode)},
                {"role": "user", "content": task.prompt}]

    for step in traj.steps:
        thinking = step.thinking
        if not thinking and cfg.oracle_rationale and step.tool_calls:
            thinking = _rationale(step.index, step.tool_calls[0], task)
        if step.tool_calls:
            call = step.tool_calls[0]
            messages.append({"role": "assistant",
                             "content": canonical_assistant_turn(thinking, call, None, cfg)})
            messages.append({"role": "tool", "name": call["name"],
                             "content": json.dumps(step.tool_results[0], ensure_ascii=False,
                                                   default=str)})
        else:
            if traj.final_answer is None:
                return None
            messages.append({"role": "assistant",
                             "content": canonical_assistant_turn(thinking, None,
                                                                 traj.final_answer, cfg)})

    if messages[-1]["role"] != "assistant":
        # ended on a tool result (hit max_steps) -> not a teachable trajectory
        return None

    rec = {"messages": messages,
           "meta": {"task_id": task.task_id, "task_type": task.task_type,
                    "difficulty": task.difficulty, "num_calls": len(traj.call_log)}}
    if cfg.include_tools_field:
        rec["tools"] = registry.schemas()
    return rec


def export(records: Sequence[tuple[Task, Trajectory, ScoreCard]], path: str | Path,
           cfg: ExportConfig | None = None, env: str = "builtin") -> dict:
    cfg = cfg or ExportConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_written, n_skipped = 0, 0
    turn_hist: dict[int, int] = {}
    with path.open("w", encoding="utf-8") as f:
        for t, j, _ in records:
            rec = trajectory_to_record(t, j, cfg, env)
            if rec is None:
                n_skipped += 1
                continue
            n_assist = sum(1 for m in rec["messages"] if m["role"] == "assistant")
            turn_hist[n_assist] = turn_hist.get(n_assist, 0) + 1
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n_written += 1
    return {"path": str(path), "written": n_written, "skipped": n_skipped,
            "assistant_turns_histogram": dict(sorted(turn_hist.items()))}
