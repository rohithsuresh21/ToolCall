"""Build the COMBINED SFT set: synthetic oracle plans + real MuSiQue chains,
both carrying synthesised <think> reasoning.

    python scripts/11_build_combined.py --n 6000 --real-frac 0.35

Why one script rather than two `atr.cli build` runs concatenated. `target_mix` in
rejection.py balances by TASK TYPE, and both sources use the same three type names
(`musique_2hop` / `3hop` / `4hop`), so it cannot see the synthetic/real split at
all -- pooling them and asking for 40/30/30 would silently let whichever source
happens to be more plentiful fill each hop family. The two sources are therefore
filtered and balanced SEPARATELY, each to its own hop mix, and only then
concatenated at an explicit ratio that this script prints.

The real slice is capped by supply, not by preference: 2009 chains exist in total
and only 335 of them are 4-hop. `--real-frac` is a request; what you get is
reported per hop family, and the script says so out loud when a family could not
be filled rather than renormalising around it (same reasoning as `target_mix`
raising on a starved family).

Output goes to a CANDIDATE path and is promoted to the committed data/sft.jsonl
only after tests/audit_sft.py passes, exactly as scripts/10_build_data.sh does --
writing straight to the training path is how a leaky build became a training set.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.agent.loop import LoopConfig  # noqa: E402
from atr.data.build_sft import ExportConfig, export  # noqa: E402
from atr.data.real_chains import load_chain_tasks  # noqa: E402
from atr.data.rejection import FilterConfig, filter_and_balance  # noqa: E402
from atr.data.teacher import collect_oracle  # noqa: E402
from atr.tasks.generator import DEFAULT_MIX, generate  # noqa: E402


def _mix_of(recs) -> dict:
    return dict(sorted(collections.Counter(t.task_type for t, _, _ in recs).items()))


def _collect(tasks, label, max_steps):
    print(f"\n[{label}] replaying {len(tasks)} oracle plans through the real registry")
    recs = collect_oracle(tasks, cfg=LoopConfig(max_steps=max_steps), progress=False)
    n_ok = sum(c.success for _, _, c in recs)
    print(f"[{label}] {n_ok}/{len(recs)} scored success ({n_ok / max(len(recs), 1):.1%})")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="synthetic seeds to draw")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--real-tasks", default="data/musique_chain_tasks.jsonl")
    ap.add_argument("--real-frac", type=float, default=0.35,
                    help="target share of the FINAL set that is real MuSiQue")
    ap.add_argument("--real-hop-mix", default="",
                    help='e.g. "2:0.40,3:0.30,4:0.30"; default = take everything')
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out", default="artifacts/sft_combined_candidate.jsonl")
    ap.add_argument("--promote-to", default="data/sft.jsonl")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="export bare tool calls, i.e. the pre-reasoning baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = ExportConfig(oracle_rationale=not args.no_reasoning)

    # --- synthetic ---------------------------------------------------------
    syn_tasks = generate(args.n, seed_start=args.seed_start, mix=DEFAULT_MIX)
    syn = _collect(syn_tasks, "synthetic", args.max_steps)
    syn_kept, syn_sum = filter_and_balance(syn, FilterConfig(
        max_per_task=1, max_per_shape=2000, target_mix=DEFAULT_MIX, seed=args.seed))
    print(f"[synthetic] kept {len(syn_kept)}  mix {_mix_of(syn_kept)}")

    # --- real --------------------------------------------------------------
    real_tasks = load_chain_tasks(args.real_tasks)
    real = _collect(real_tasks, "real", args.max_steps)
    real_mix = None
    if args.real_hop_mix:
        real_mix = {f"musique_{k}hop": float(v) for k, v in
                    (p.split(":") for p in args.real_hop_mix.split(","))}
    real_kept, real_sum = filter_and_balance(real, FilterConfig(
        max_per_task=1,
        # A real task's `route` is its evidence-id list, unique per row, so every
        # real record is its own shape and the shape cap can never bind. Say so by
        # switching it off rather than leaving a number that does nothing.
        dedupe_by_shape=False,
        target_mix=real_mix, seed=args.seed))
    print(f"[real] kept {len(real_kept)}  mix {_mix_of(real_kept)}")

    # --- combine at an explicit ratio ---------------------------------------
    frac = max(0.0, min(1.0, args.real_frac))
    if frac >= 1.0:
        n_syn, n_real = 0, len(real_kept)
    elif frac <= 0.0:
        n_syn, n_real = len(syn_kept), 0
    else:
        # take everything from whichever side is the binding constraint
        by_real = len(real_kept), int(round(len(real_kept) * (1 - frac) / frac))
        by_syn = int(round(len(syn_kept) * frac / (1 - frac))), len(syn_kept)
        n_real, n_syn = by_real if by_real[1] <= len(syn_kept) else by_syn
    rng.shuffle(syn_kept)
    rng.shuffle(real_kept)
    combined = syn_kept[:n_syn] + real_kept[:n_real]
    rng.shuffle(combined)

    got = n_real / max(len(combined), 1)
    print(f"\ncombined: {len(combined)} records = {n_syn} synthetic + {n_real} real "
          f"({got:.1%} real; asked for {frac:.1%})")
    if abs(got - frac) > 0.02:
        print(f"  NOTE: the real slice is supply-capped at {len(real_kept)} records, "
              f"so the requested share could not be met exactly.")

    stats = export(combined, args.out, cfg)
    print(json.dumps(stats, indent=2))

    src = collections.Counter(
        ("real" if t.documents else "synthetic", t.task_type) for t, _, _ in combined)
    print("\nrecords by (source, hop):")
    for k in sorted(src):
        print(f"  {k[0]:<9} {k[1]:<14} {src[k]:>5}")

    print(f"\n=== auditing {args.out} ===")
    rc = subprocess.call([sys.executable, "tests/audit_sft.py", args.out], cwd=REPO)
    if rc != 0:
        raise SystemExit(f"REFUSING TO PROMOTE: {args.out} reports DEFECTS PRESENT. "
                         f"It is left in place for inspection; {args.promote_to} is untouched.")
    if args.no_promote:
        print(f"\n--no-promote: leaving the clean build at {args.out}")
        return
    Path(args.promote_to).write_bytes(Path(args.out).read_bytes())
    Path(args.out).unlink()
    print(f"\npromoted -> {args.promote_to} ({len(combined)} records)")
    print(f"COMMIT IT: git add {args.promote_to}")


if __name__ == "__main__":
    main()
