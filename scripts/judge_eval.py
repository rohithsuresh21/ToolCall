"""Judge-parity evaluation of an ATR adapter against the public benchmark split.

Each public example becomes an episode whose `search` tool is BM25 over that
example's OWN 20-passage candidate set (no synthetic worlds). The model plays
the same chatml/tool protocol it was trained on; answers are scored with the
repo's match_answer (lenient text-containment) plus an exact-norm match that
mirrors how the judge compares short factual answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.agent.backends import HFBackend, SamplingParams
from atr.agent.loop import Step, Trajectory
from atr.agent.parser import parse_turn
from atr.agent.prompt import build_messages
from atr.tasks.schema import match_answer, norm_text
from atr.tools.builtin import build_registry
from atr.tools.world import World


class JudgeEpisode:
    """Single task against its own 20-passage world, mirrors loop._Episode."""

    def __init__(self, rec: dict, registry, cfg):
        self.rec = rec
        self.cfg = cfg
        self.registry = registry
        seed = int(hashlib.sha1(str(rec["id"]).encode()).hexdigest()[:8], 16)
        self.world = World(seed=seed)
        self.world.documents = [
            {"doc_id": f"{i}", "title": p["title"], "text": p["paragraph_text"],
             "facts": {}, "links": []}
            for i, p in enumerate(rec["context"])
        ]
        self.messages = build_messages(rec["question"], registry, "native")
        self.traj = Trajectory(task_id=rec["id"], prompt=rec["question"],
                               seed=seed,
                               meta={"source": "judge_public", "hops": rec["hops"]})
        self.done = False

    def ingest(self, text: str) -> None:
        idx = len(self.traj.steps)
        p = parse_turn(text)
        calls = p.tool_calls[: max(1, self.cfg["max_calls_per_turn"])]
        self.messages.append({"role": "assistant", "content": text})
        results = []
        for c in calls:
            res = self.registry.call(self.world, c.name, c.arguments)
            results.append(res)
            self.messages.append({
                "role": "tool", "name": c.name,
                "content": json.dumps(res, ensure_ascii=False, default=str)[: self.cfg["truncate_result_chars"]],
            })
        self.traj.steps.append(Step(
            index=idx, assistant_text=text, thinking=p.thinking,
            tool_calls=[{"name": c.name, "arguments": c.arguments, "repairs": c.repairs} for c in calls],
            tool_results=results, parse_errors=p.errors, strict_format=p.strict_format))
        if calls:
            if idx + 1 >= self.cfg["max_steps"]:
                self.finish("max_steps")
            return
        if p.final_answer is not None:
            self.traj.final_answer = p.final_answer
            self.finish("answered")
        elif self.cfg["stop_on_empty_turn"]:
            self.finish("empty_turn")
        elif idx + 1 >= self.cfg["max_steps"]:
            self.finish("max_steps")

    def finish(self, reason: str) -> None:
        self.done = True
        self.traj.stop_reason = reason
        self.traj.messages = self.messages
        self.traj.call_log = self.world.call_log
        self.traj.sent_messages = self.world.sent_messages


def run_tasks(recs: list[dict], backend, registry, cfg, batch_size: int) -> list[Trajectory]:
    tools = registry.schemas()
    sp = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
    trajs: list[Trajectory] = []
    for i in range(0, len(recs), batch_size):
        chunk = recs[i:i + batch_size]
        eps = [JudgeEpisode(r, registry, cfg) for r in chunk]
        print(f"[eval] batch {i // batch_size + 1}/{-(-len(recs) // batch_size)} "
              f"({len(chunk)} tasks)", flush=True)
        for turn in range(cfg["max_steps"]):
            live = [e for e in eps if not e.done]
            if not live:
                break
            texts = backend.generate([e.messages for e in live], tools=tools, sp=sp,
                                     ids=[e.rec["id"] for e in live])
            for e, t in zip(live, texts):
                e.ingest(t)
        for e in eps:
            if not e.done:
                e.finish("max_steps")
        trajs.extend(e.traj for e in eps)
    return trajs


def evaluate(recs, backend, cfg, batch_size: int, name: str, out_dir: Path):
    registry = build_registry()
    trajs = run_tasks(recs, backend, registry, cfg, batch_size)
    by_hop: dict[int, list] = {}
    rows = []
    for rec, traj in zip(recs, trajs):
        gold_kind = {"kind": "text", "value": rec["answer"]}
        ok, reason = match_answer(gold_kind, traj.final_answer)
        exact = (norm_text(traj.final_answer) == norm_text(rec["answer"])) \
            if traj.final_answer else False
        strict_format = bool(traj.steps) and all(s.strict_format for s in traj.steps)
        rows.append({
            "id": rec["id"], "hops": rec["hops"], "acc_text": ok, "acc_exact": exact,
            "final_answer": (traj.final_answer or "")[:200], "stop_reason": traj.stop_reason,
            "steps": len(traj.steps), "calls": traj.num_tool_calls,
            "format_strict": strict_format,
        })
        by_hop.setdefault(rec["hops"], []).append(rows[-1])

    lines = [f"{name} | judge public {len(recs)} tasks", "=" * 40]
    for h in sorted(by_hop):
        rs = by_hop[h]
        lines.append(
            f"{h}hop  n={len(rs):3d}  acc(text)={(sum(r['acc_text'] for r in rs) / len(rs)):.1%}  "
            f"acc(exact)={(sum(r['acc_exact'] for r in rs) / len(rs)):.1%}  "
            f"avg_steps={statistics.fmean([r['steps'] for r in rs]):.2f}  "
            f"avg_calls={statistics.fmean([r['calls'] for r in rs]):.2f}  "
            f"format={(sum(r['format_strict'] for r in rs) / len(rs)):.1%}")
    txt = "\n".join(lines)
    print("\n" + txt)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(txt)
    (out_dir / "report.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "trajectories.jsonl").open("w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")
    print(f"\nsaved -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="artifacts/judge_tasks.jsonl")
    ap.add_argument("--out", default="artifacts/judge_eval")
    ap.add_argument("--batch-size", type=int, default=27)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    print(f"loaded {len(recs)} judge tasks")
    backend = HFBackend(args.base, adapter=args.adapter)
    cfg = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
           "max_steps": args.max_steps, "max_calls_per_turn": 1,
           "truncate_result_chars": 2000, "stop_on_empty_turn": True}
    name = "SFT-adapter" if args.adapter else "BASE"
    evaluate(recs, backend, cfg, args.batch_size, name, Path(args.out))


if __name__ == "__main__":
    main()