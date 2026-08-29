"""
Trajectory collection.

Three sources, all producing the same Trajectory objects so the downstream
filter/export code does not care where they came from:

  teacher   -- a large model driven through the SAME agent loop. This is the
               highest-quality source and rule 9 explicitly allows it, since the
               teacher touches nothing at inference time.
  oracle    -- replay of the generator's reference plan. Free, perfectly correct,
               and perfectly styleless: it teaches format and tool choice but not
               *deliberation*. Use it to bootstrap, not as the whole diet.
  self      -- the student's own sampled rollouts (rejection sampling / STaR).
               Cheap, on-policy, and the only source whose reasoning is already
               in the student's own idiom. Best marginal value after round 1.

The recommended recipe is all three: oracle to fix format, teacher for hard
multi-hop deliberation, self-sampling to keep the distribution on-policy.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from ..agent.backends import Backend, SamplingParams
from ..agent.loop import LoopConfig, Step, Trajectory, run_episodes
from ..eval.harness import oracle_backend
from ..tasks.schema import ScoreCard, Task
from ..tasks.verifiers import score as score_traj
from ..tools.adapter import get_registry


def collect(tasks: Sequence[Task], backend: Backend, env: str = "builtin",
            cfg: LoopConfig | None = None, sp: SamplingParams | None = None,
            samples_per_task: int = 1, batch_size: int = 64,
            progress: bool = True) -> list[tuple[Task, Trajectory, ScoreCard]]:
    """Roll out `samples_per_task` trajectories per task and score every one."""
    registry = get_registry(env)
    cfg = cfg or LoopConfig()
    sp = sp or SamplingParams()
    out: list[tuple[Task, Trajectory, ScoreCard]] = []

    expanded: list[Task] = []
    for t in tasks:
        for k in range(samples_per_task):
            if k == 0:
                expanded.append(t)
            else:
                # distinct task_id per sample so MockBackend / logs stay unambiguous,
                # same seed so the world (and therefore the gold answer) is identical
                d = t.to_dict()
                d["task_id"] = f"{t.task_id}#s{k}"
                expanded.append(Task.from_dict(d))

    for i in range(0, len(expanded), batch_size):
        chunk = expanded[i:i + batch_size]
        if progress:
            print(f"[collect] {i}/{len(expanded)}", flush=True)
        trajs = run_episodes(chunk, registry, backend, cfg, sp)
        for t, j in zip(chunk, trajs):
            out.append((t, j, score_traj(t, j)))
    return out


def collect_oracle(tasks: Sequence[Task], **kw) -> list[tuple[Task, Trajectory, ScoreCard]]:
    return collect(tasks, oracle_backend(tasks), sp=SamplingParams(temperature=0.0), **kw)


def dump(records: Iterable[tuple[Task, Trajectory, ScoreCard]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t, j, c in records:
            f.write(json.dumps({"task": t.to_dict(), "trajectory": j.to_dict(),
                                "score": c.to_dict()}, default=str) + "\n")
    return path


def load(path: str | Path) -> list[tuple[Task, Trajectory, ScoreCard]]:
    out = []
    with Path(path).open() as f:
        for line in f:
            d = json.loads(line)
            t = Task.from_dict(d["task"])
            j = Trajectory(**{**d["trajectory"],
                              "steps": [Step(**s) for s in d["trajectory"]["steps"]]})
            c = ScoreCard(**d["score"])
            out.append((t, j, c))
    return out
