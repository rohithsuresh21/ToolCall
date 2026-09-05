"""Judge-parity evaluation of an ATR adapter against the public benchmark split.

Each public example becomes an episode whose `search` tool is BM25 over that
example's OWN 20-passage candidate set (no synthetic worlds). The model plays
the same chatml/tool protocol it was trained on.

THE HEADLINE NUMBER IS `answer_f1`, and it is the same function the dev eval and
the GRPO reward use. This file used to report `match_answer` (lenient substring
containment) and an exact-norm match instead, and imported `answer_f1` nowhere --
so the one script that touches the judge's own data was the one script not
scoring the judge's metric. Comparing that `acc_exact` against a dev-set
`final_f1` is what produced the phantom "5.67x backend ratio": containment,
equality and token-F1 disagree by multiples on the same output (gold
"Springfield", answer "The answer is Springfield, founded in 1849." scores
containment 1.0, exact 0.0, F1 0.286), and none of the gap was the backend.

`acc_exact` is retained ONLY as a labelled secondary diagnostic -- it is strictly
harsher than the judge and must never be quoted as the score. The lenient
containment column is gone: a third matcher in the same report is what invited
the comparison in the first place.

Sampling and loop limits come from configs/eval.json via load_eval_config(), the
same frozen config the CLI, the GRPO canary and modal_app read; CLI flags default
to None so "not passed" stays distinguishable from an explicit value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.agent.backends import SamplingParams, load_backend
from atr.agent.loop import Step, Trajectory
from atr.agent.parser import parse_turn
from atr.agent.prompt import build_messages
from atr.eval.harness import load_eval_config
from atr.tasks.schema import answer_f1, norm_text
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
    sp = SamplingParams(temperature=cfg["temperature"], top_p=cfg["top_p"],
                        max_tokens=cfg["max_new_tokens"])
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
        # HEADLINE: the judge's own metric, and the same call the dev eval and the
        # GRPO reward make. Everything else on this row is a diagnostic.
        f1 = answer_f1(gold_kind, traj.final_answer)
        exact = (norm_text(traj.final_answer) == norm_text(rec["answer"])) \
            if traj.final_answer else False
        strict_format = bool(traj.steps) and all(s.strict_format for s in traj.steps)
        rows.append({
            "id": rec["id"], "hops": rec["hops"], "f1": round(f1, 4),
            "acc_exact_diagnostic": exact,
            "final_answer": (traj.final_answer or "")[:200], "stop_reason": traj.stop_reason,
            "steps": len(traj.steps), "calls": traj.num_tool_calls,
            "format_strict": strict_format,
            "answer_tokens": len((traj.final_answer or "").split()),
        })
        by_hop.setdefault(rec["hops"], []).append(rows[-1])

    lines = [f"{name} | judge public {len(recs)} tasks",
             "HEADLINE = F1 (token-level, the judge's metric). "
             "exact = secondary diagnostic, stricter than the judge -- do not quote it.",
             "=" * 78]
    for h in sorted(by_hop):
        rs = by_hop[h]
        lines.append(
            f"{h}hop  n={len(rs):3d}  F1={statistics.fmean([r['f1'] for r in rs]):.1%}  "
            f"(exact={(sum(r['acc_exact_diagnostic'] for r in rs) / len(rs)):.1%})  "
            f"avg_steps={statistics.fmean([r['steps'] for r in rs]):.2f}  "
            f"avg_calls={statistics.fmean([r['calls'] for r in rs]):.2f}  "
            f"avg_ans_tok={statistics.fmean([r['answer_tokens'] for r in rs]):.1f}  "
            f"format={(sum(r['format_strict'] for r in rs) / len(rs)):.1%}")
    lines.append("-" * 78)
    lines.append(
        f"OVERALL  n={len(rows):3d}  F1={statistics.fmean([r['f1'] for r in rows]):.1%}  "
        f"(exact={(sum(r['acc_exact_diagnostic'] for r in rows) / len(rows)):.1%})  "
        f"avg_ans_tok={statistics.fmean([r['answer_tokens'] for r in rows]):.1f}")
    lines.append(f"sampling: temperature={cfg['temperature']} top_p={cfg['top_p']} "
                 f"max_new_tokens={cfg['max_new_tokens']} max_steps={cfg['max_steps']} "
                 f"(configs/eval.json unless overridden)")
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
    ap.add_argument("--backend", default=None, help="vllm:<model> (default: vllm:<base>)")
    ap.add_argument("--tasks", default="data/judge_tasks.jsonl")
    ap.add_argument("--out", default="artifacts/judge_eval")
    ap.add_argument("--name", default=None, help="label used in the report header")
    ap.add_argument("--batch-size", type=int, default=27)
    ap.add_argument("--config", default=None,
                    help="eval config json (default: configs/eval.json)")
    # None -> resolved from the frozen eval config, exactly as atr.cli._frozen does.
    # These used to default to 0.2 / 8 / 512, so this script sampled at a different
    # temperature and a shorter step budget than every other eval in the repo while
    # CLAUDE.md claimed the frozen config was shared.
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        raise SystemExit(
            f"FATAL: {tasks_path} not found. Build it from the HF dataset first:\n"
            f"  python scripts/make_judge_tasks.py --out {tasks_path}")
    recs = [json.loads(l) for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"loaded {len(recs)} judge tasks")

    ec = load_eval_config(args.config)
    cfg = {
        "temperature": args.temperature if args.temperature is not None else ec["temperature"],
        "top_p": args.top_p if args.top_p is not None else ec["top_p"],
        "max_new_tokens": (args.max_new_tokens if args.max_new_tokens is not None
                           else ec["max_new_tokens"]),
        "max_steps": args.max_steps if args.max_steps is not None else ec["max_steps"],
        "max_calls_per_turn": 1, "truncate_result_chars": 2000, "stop_on_empty_turn": True,
    }
    print(f"[frozen] temperature={cfg['temperature']} top_p={cfg['top_p']} "
          f"max_new_tokens={cfg['max_new_tokens']} max_steps={cfg['max_steps']}")
    backend = load_backend(args.backend or f"vllm:{args.base}", adapter=args.adapter)
    name = (args.name if args.name else
            (("base-" + Path(args.adapter).name) if args.adapter else "BASE"))
    evaluate(recs, backend, cfg, args.batch_size, name, Path(args.out))


if __name__ == "__main__":
    main()