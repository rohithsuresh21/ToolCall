"""
F8a (FIX-2): reward-discriminativity audit.

Before trusting ANY shaping term, measure whether it actually tracks success on
real rollouts. The IRC study (arXiv 2604.02869) showed naively-designed dense
rewards can DEGRADE performance by up to 14pp because a term's sign and the
advantage direction disagree. This script computes the point-biserial correlation
between every reward part (the `parts` dict from compute_reward) and binary task
success, then flags:

  FLAT     -- |r| < flat_threshold : the term fires the same way regardless of
              outcome; it contributes noise, not signal.
  INVERTED -- r < -sign_threshold  : the term is NEGATIVELY correlated with
              success; rewarding it pushes the policy away from winning.
  OK       -- anything else (positive, even weakly).

Usage:
  python -m atr.train.reward_audit --records artifacts/raw_trajectories.jsonl
  python -m atr.train.reward_audit --backend oracle --n 300      # fresh rollouts

Exit code 0 if no INVERTED terms, 1 otherwise (CI-friendly).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _point_biserial(xs: list[float], ys: list[bool]) -> float:
    """Pearson r between a continuous variable and a binary one."""
    n = len(xs)
    if n < 3:
        return float("nan")
    y1 = [x for x, y in zip(xs, ys) if y]
    y0 = [x for x, y in zip(xs, ys) if not y]
    if not y1 or not y0:
        return float("nan")
    m_all = sum(xs) / n
    sd = math.sqrt(sum((x - m_all) ** 2 for x in xs) / n) or 1e-12
    return (sum(y1) / len(y1) - sum(y0) / len(y0)) / sd * \
        math.sqrt(len(y1) * len(y0) / (n * n))


def audit(parts_by_ep: list[dict], successes: list[bool],
          flat_threshold: float = 0.05, sign_threshold: float = -0.10) -> list[dict]:
    keys: list[str] = []
    for parts in parts_by_ep:
        for k in parts:
            if k not in keys:
                keys.append(k)
    rows = []
    for k in keys:
        xs = [float(p.get(k, 0.0)) for p in parts_by_ep]
        if len(set(xs)) < 2:
            rows.append({"term": k, "r": float("nan"), "verdict": "CONSTANT"})
            continue
        r = _point_biserial(xs, successes)
        if math.isnan(r):
            verdict = "FLAT"
        elif abs(r) < flat_threshold:
            verdict = "FLAT"
        elif r < sign_threshold:
            verdict = "INVERTED"
        else:
            verdict = "OK"
        rows.append({"term": k, "r": round(r, 4), "verdict": verdict})
    rows.sort(key=lambda x: -(x["r"] if not math.isnan(x["r"]) else -9))
    return rows


def _load_records(path: Path) -> tuple[list[dict], list[bool]]:
    """teacher.dump format: {task, trajectory, score} per line."""
    from ..tasks.schema import ScoreCard, Task
    from ..train.reward import compute_reward
    parts_by_ep, successes = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        t = Task.from_dict(d["task"])
        card = ScoreCard(**d["score"])
        _, parts = compute_reward(t, card)
        parts_by_ep.append(parts)
        successes.append(bool(card.success))
    return parts_by_ep, successes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", help="trajectory jsonl from atr.cli collect")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--backend", default=None,
                     help="roll out fresh: oracle | hf:M | vllm:M | openai:M")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=700_000)
    args = ap.parse_args()

    if args.records:
        parts_by_ep, successes = _load_records(Path(args.records))
    elif args.backend:
        from ..agent.backends import load_backend
        from ..data.teacher import collect
        from ..eval.harness import oracle_backend
        from ..tasks.generator import generate
        tasks = generate(args.n, seed_start=args.seed_start)
        be = oracle_backend(tasks) if args.backend == "oracle" else load_backend(args.backend)
        recs = collect(tasks, be, samples_per_task=1, progress=True)
        from ..train.reward import compute_reward
        parts_by_ep = [compute_reward(t, c)[1] for t, _, c in recs]
        successes = [bool(c.success) for _, _, c in recs]
    else:
        raise SystemExit("give --records or --backend")

    rows = audit(parts_by_ep, successes)
    print(f"\n{'reward term':<16}{'point-biserial r':>18}   verdict")
    print("-" * 52)
    inverted = False
    for row in rows:
        flag = "" if row["verdict"] in ("OK",) else ("   <-- FIX" if row["verdict"] == "INVERTED" else "")
        if row["verdict"] == "INVERTED":
            inverted = True
        r = row["r"]
        rs = f"{r:+.4f}" if not math.isnan(r) else "   nan"
        print(f"{row['term']:<16}{rs:>18}   {row['verdict']}{flag}")
    print(f"\nn = {len(successes)} episodes, {sum(successes)} successful")
    print("FLAT terms contribute noise, not signal -> consider removing/retuning.")
    print("INVERTED terms actively push against success -> fix before training.")
    raise SystemExit(1 if inverted else 0)


if __name__ == "__main__":
    main()
