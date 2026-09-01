"""
The agentic rollout loop.

Batched multi-turn by construction: N episodes advance one turn at a time
against one batched generate() call. This is not a micro-optimisation -- GRPO
needs G rollouts per task per step, and a per-episode python loop makes RL
wall-clock infeasible.

What the loop does NOT do (constraint 4/5): it never chooses a tool, never
rewrites arguments, never plans, and never injects task-specific hints. It
parses what the model emitted, executes it, and hands the result back. The only
environment-side nudges are (a) tool error payloads, which are normal tool
behaviour, and (b) an optional repeat-guard that tells the model it just made
the same call three times. Both are configurable; turn the guard off with
`repeat_guard=0` if the official benchmark forbids it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from ..tools.registry import ToolRegistry
from ..tools.world import World, build_world
from .backends import Backend, SamplingParams
from .parser import ParsedTurn, ToolCall, parse_turn
from .prompt import build_messages


@dataclass
class Step:
    index: int
    assistant_text: str
    thinking: str
    tool_calls: list[dict]          # {name, arguments, repairs}
    tool_results: list[dict]
    parse_errors: list[str]
    strict_format: bool


@dataclass
class Trajectory:
    task_id: str
    prompt: str
    steps: list[Step] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    stop_reason: str = "running"
    seed: int = 0
    meta: dict = field(default_factory=dict)
    # populated from the episode's World at the end
    call_log: list[dict] = field(default_factory=list)
    sent_messages: list[dict] = field(default_factory=list)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def tool_names(self) -> list[str]:
        return [c["name"] for s in self.steps for c in s.tool_calls]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class LoopConfig:
    max_steps: int = 10
    max_calls_per_turn: int = 1     # >1 allows parallel calls in one turn
    prompt_mode: str = "native"     # native | explicit
    repeat_guard: int = 3           # 0 disables
    truncate_result_chars: int = 2000
    stop_on_empty_turn: bool = True
    text_loader: "callable | None" = None  # Part A: naturalized-passage loader (opt-in)


class _Episode:
    def __init__(self, task, registry: ToolRegistry, cfg: LoopConfig):
        self.task = task
        self.cfg = cfg
        self.registry = registry
        self.world: World = build_world(task.seed, text_loader=cfg.text_loader)
        self.messages = build_messages(task.prompt, registry, cfg.prompt_mode)
        self.traj = Trajectory(task_id=task.task_id, prompt=task.prompt, seed=task.seed,
                               meta={"task_type": task.task_type, "difficulty": task.difficulty})
        self.done = False
        self._call_counts: dict[str, int] = {}

    # -- one turn ---------------------------------------------------------
    def ingest(self, text: str) -> None:
        idx = len(self.traj.steps)
        p: ParsedTurn = parse_turn(text)
        calls = p.tool_calls[: max(1, self.cfg.max_calls_per_turn)]

        self.messages.append({"role": "assistant", "content": text})
        results: list[dict] = []

        for c in calls:
            sig = f"{c.name}:{json.dumps(c.arguments, sort_keys=True, default=str)}"
            self._call_counts[sig] = self._call_counts.get(sig, 0) + 1
            if self.cfg.repeat_guard and self._call_counts[sig] > self.cfg.repeat_guard:
                res = {"error": "repeated_call",
                       "message": f"you have already made this exact call {self._call_counts[sig] - 1} times "
                                  f"and it did not move the task forward",
                       "hint": "change the arguments, try a different tool, or answer with what you have"}
                self.world.call_log.append({"name": c.name, "args": c.arguments, "ok": False,
                                            "coerced": False, "result": res, "latency_ms": 0.0,
                                            "guarded": True})
            else:
                res = self.registry.call(self.world, c.name, c.arguments)
            results.append(res)
            self.messages.append({
                "role": "tool", "name": c.name,
                "content": _clip(json.dumps(res, ensure_ascii=False, default=str),
                                 self.cfg.truncate_result_chars)})

        self.traj.steps.append(Step(
            index=idx, assistant_text=text, thinking=p.thinking,
            tool_calls=[{"name": c.name, "arguments": c.arguments, "repairs": c.repairs} for c in calls],
            tool_results=results, parse_errors=p.errors, strict_format=p.strict_format))

        if calls:
            if idx + 1 >= self.cfg.max_steps:
                self.finish("max_steps")
            return
        if p.final_answer is not None:
            self.traj.final_answer = p.final_answer
            self.finish("answered")
        elif self.cfg.stop_on_empty_turn:
            self.finish("empty_turn")
        elif idx + 1 >= self.cfg.max_steps:
            self.finish("max_steps")

    def finish(self, reason: str) -> None:
        self.done = True
        self.traj.stop_reason = reason
        self.traj.messages = self.messages
        self.traj.call_log = self.world.call_log
        self.traj.sent_messages = self.world.sent_messages


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 20] + f"... [+{len(s) - n + 20} chars]"


# ---------------------------------------------------------------------------
def run_episodes(tasks: Sequence[Any], registry: ToolRegistry, backend: Backend,
                 cfg: LoopConfig | None = None, sp: SamplingParams | None = None,
                 progress: bool = False) -> list[Trajectory]:
    """Advance every episode in lockstep, one batched generate() per turn."""
    cfg = cfg or LoopConfig()
    sp = sp or SamplingParams()
    eps = [_Episode(t, registry, cfg) for t in tasks]
    tools = registry.schemas() if cfg.prompt_mode == "native" else None

    for turn in range(cfg.max_steps):
        live = [e for e in eps if not e.done]
        if not live:
            break
        if progress:
            print(f"  turn {turn + 1}/{cfg.max_steps}: {len(live)} live", flush=True)
        texts = backend.generate([e.messages for e in live], tools=tools, sp=sp,
                                 ids=[e.task.task_id for e in live])
        for e, t in zip(live, texts):
            e.ingest(t)

    for e in eps:
        if not e.done:
            e.finish("max_steps")
    return [e.traj for e in eps]
