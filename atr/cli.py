"""atr -- command line entry point.  python -m atr.cli <command> --help"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .agent.backends import SamplingParams, load_backend
from .agent.loop import LoopConfig
from .data.build_sft import ExportConfig, export
from .data.rejection import FilterConfig, filter_and_balance
from .data.teacher import collect, dump, load
from .eval.harness import (aggregate, evaluate, format_report, load_eval_config,
                           oracle_backend, save_run)
from .tasks.generator import DEFAULT_MIX, dev_set, generate
from .tasks.schema import Task


def _tasks_from(args) -> list[Task]:
    if getattr(args, "tasks", None):
        return [Task.from_dict(json.loads(l)) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if getattr(args, "dev", False):
        return dev_set(args.n_per_type or load_eval_config()["n_per_type"])
    return generate(args.n, seed_start=args.seed_start)


def _backend(args):
    if args.backend == "oracle":
        return None  # built later, needs the task list
    kw = {}
    if getattr(args, "adapter", None):
        kw["adapter"] = args.adapter
    if getattr(args, "base_url", None):
        kw["base_url"] = args.base_url
    return load_backend(args.backend, **kw)


def _frozen(args):
    """F10: resolve sampling knobs -- explicit CLI flag wins over configs/eval.json."""
    ec = load_eval_config(getattr(args, "config", None))
    return {
        "temperature": args.temperature if args.temperature is not None else ec["temperature"],
        "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else ec["max_new_tokens"],
        "max_steps": args.max_steps if args.max_steps is not None else ec["max_steps"],
        "repeat_guard": args.repeat_guard if getattr(args, "repeat_guard", None) is not None else ec["repeat_guard"],
    }


# ---------------------------------------------------------------------------
def cmd_gen(args):
    tasks = _tasks_from(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")
    from collections import Counter
    print(f"wrote {len(tasks)} tasks -> {out}")
    print(dict(Counter(t.task_type for t in tasks)))


def cmd_eval(args):
    fz = _frozen(args)
    tasks = _tasks_from(args)
    be = oracle_backend(tasks) if args.backend == "oracle" else _backend(args)
    cards, trajs = evaluate(
        tasks, be, env=args.env,
        cfg=LoopConfig(max_steps=fz["max_steps"], repeat_guard=fz["repeat_guard"]),
        sp=SamplingParams(temperature=fz["temperature"], max_tokens=fz["max_new_tokens"]),
        batch_size=args.batch_size, strict_necessity=not args.loose_necessity,
        strict_match=args.strict_match)
    rep = aggregate(cards)
    title = f"{args.backend}  |  {len(tasks)} tasks"
    print("\n" + format_report(rep, title))
    if args.out:
        p = save_run(args.out, cards, trajs, rep, {"title": title, "backend": args.backend})
        print(f"\nsaved -> {p}")


def cmd_collect(args):
    fz = _frozen(args)
    tasks = _tasks_from(args)
    be = oracle_backend(tasks) if args.backend == "oracle" else _backend(args)
    recs = collect(tasks, be, env=args.env,
                   cfg=LoopConfig(max_steps=fz["max_steps"]),
                   sp=SamplingParams(temperature=fz["temperature"],
                                     max_tokens=fz["max_new_tokens"]),
                   samples_per_task=args.samples_per_task, batch_size=args.batch_size)
    n_ok = sum(c.success for _, _, c in recs)
    print(f"collected {len(recs)} trajectories, {n_ok} successful ({n_ok/len(recs):.1%})")
    print("->", dump(recs, args.out))


def cmd_build(args):
    recs = []
    for p in args.inputs:
        recs.extend(load(p))
    kept, summary = filter_and_balance(recs, FilterConfig(
        require_strict_format=args.strict_format,
        max_per_task=args.max_per_task,
        max_call_bloat=args.max_call_bloat,
        target_mix=DEFAULT_MIX if args.rebalance else None))
    print(json.dumps(summary, indent=2))
    stats = export(kept, args.out, ExportConfig(
        keep_thinking=not args.drop_thinking,
        max_thinking_chars=args.max_thinking_chars,
        oracle_rationale=args.oracle_rationale))
    print(json.dumps(stats, indent=2))


def cmd_ablate(args):
    """
    Measure how much each skill is worth end to end, by degrading a perfect
    agent one axis at a time. This is the cheapest way to decide where the next
    batch of training data should go -- and it runs on CPU in seconds.
    """
    tasks = _tasks_from(args)
    fz = _frozen(args)
    cfg = LoopConfig(max_steps=fz["max_steps"])
    rows = []

    def run(label, corrupt):
        cards, _ = evaluate(tasks, oracle_backend(tasks, corrupt), cfg=cfg, progress=False)
        rep = aggregate(cards)["overall"]
        rows.append((label, rep["success"], rep["final_correct"], rep["avg_calls"]))

    run("perfect agent", None)
    for p in (0.05, 0.10, 0.20):
        rng = random.Random(0)
        def bad_args(tid, step, call, p=p, rng=rng):
            if rng.random() < p:
                call["arguments"] = {}
            return call
        run(f"argument errors @ {int(p*100)}%", bad_args)
    for p in (0.05, 0.10, 0.20):
        rng = random.Random(1)
        def early_stop(tid, step, call, p=p, rng=rng):
            return None if rng.random() < p else call
        run(f"gives up early @ {int(p*100)}%", early_stop)
    for p in (0.05, 0.10, 0.20):
        rng = random.Random(2)
        def wrong_tool(tid, step, call, p=p, rng=rng):
            if rng.random() < p:
                call["name"] = "web_search"
                call["arguments"] = {"query": "..."}
            return call
        run(f"wrong tool @ {int(p*100)}%", wrong_tool)

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'PER-STEP DEGRADATION':<{w}}{'success':>9}{'answer':>9}{'calls':>8}")
    print("-" * (w + 26))
    for label, s, f, c in rows:
        print(f"{label:<{w}}{100*s:>8.1f}%{100*f:>8.1f}%{c:>8.2f}")
    print("\nRead this as the multiplicative cost of per-step error on end-to-end score.")


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="atr")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, backend_default="oracle"):
        sp.add_argument("--tasks", help="jsonl of tasks (overrides --n/--dev)")
        sp.add_argument("--dev", action="store_true", help="use the balanced dev set")
        sp.add_argument("--n-per-type", type=int, default=None,
                        help="default: configs/eval.json n_per_type")
        sp.add_argument("--n", type=int, default=200)
        sp.add_argument("--seed-start", type=int, default=0)
        sp.add_argument("--env", default="builtin")
        sp.add_argument("--backend", default=backend_default,
                        help="oracle | mock | hf:MODEL | vllm:MODEL | openai:MODEL")
        sp.add_argument("--adapter", default=None)
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--config", default=None,
                        help="eval config json (default: configs/eval.json)")
        # None -> resolved from the frozen eval config by _frozen()
        sp.add_argument("--max-steps", type=int, default=None)
        sp.add_argument("--max-new-tokens", type=int, default=None)
        sp.add_argument("--temperature", type=float, default=None)
        sp.add_argument("--batch-size", type=int, default=64)

    g = sub.add_parser("gen", help="mint tasks to jsonl")
    common(g); g.add_argument("--out", default="artifacts/tasks.jsonl"); g.set_defaults(fn=cmd_gen)

    e = sub.add_parser("eval", help="score a backend on tasks")
    common(e)
    e.add_argument("--out", default=None)
    e.add_argument("--repeat-guard", type=int, default=None,
                   help="default: configs/eval.json repeat_guard")
    e.add_argument("--loose-necessity", action="store_true")
    e.add_argument("--strict-match", action="store_true",
                   help="F13: refuse shotgun numeric answers and untagged replies")
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("collect", help="roll out trajectories for training data")
    common(c)
    c.add_argument("--out", default="artifacts/raw_trajectories.jsonl")
    c.add_argument("--samples-per-task", type=int, default=4)
    c.set_defaults(fn=cmd_collect)

    b = sub.add_parser("build", help="filter trajectories -> SFT jsonl")
    b.add_argument("inputs", nargs="+")
    b.add_argument("--out", default="artifacts/sft.jsonl")
    b.add_argument("--strict-format", action="store_true")
    b.add_argument("--max-per-task", type=int, default=1)
    b.add_argument("--max-call-bloat", type=float, default=2.0)
    b.add_argument("--rebalance", action="store_true")
    b.add_argument("--drop-thinking", action="store_true")
    b.add_argument("--max-thinking-chars", type=int, default=400)
    b.add_argument("--oracle-rationale", action="store_true")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("ablate", help="what is each skill worth end to end?")
    common(a); a.set_defaults(fn=cmd_ablate)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
