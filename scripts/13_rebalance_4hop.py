"""Build the 4-HOP-WEIGHTED arm: data/sft_r0_max.jsonl re-mixed from 40/30/30 to
25/30/45, at the same record count.

    python scripts/13_rebalance_4hop.py --out data/sft_r0_max_4h.jsonl

4-hop sits at 31.7% judge F1 against 80.0% for 2-hop, so that is where the loss
is and where the records should go.

WHY THIS IS A TOP-UP AND NOT A FRESH `generate()` RUN. Re-running the generator
with a different mix changes WHICH task every index draws -- `generate()` picks
the family per index from one `random.Random(rng_seed)` stream, so re-weighting it
re-points the whole draw -- and the resulting set would share almost nothing with
the base. The mix would then be confounded with a complete change of population.
Instead: keep every 3-hop and every 4-hop record the base already has, drop the
2-hop surplus, and mint only the 4-hop RECORDS THAT ARE MISSING, from a disjoint
seed range with the same generator, templates, gates and reasoning synthesiser.
Everything carried over is byte-identical, so the mix is the only variable.

CAPACITY IS CHECKED FIRST, AND COMPUTED RATHER THAN REMEMBERED. A question names
only its HEAD entity, so a route's distinct-question capacity is its head kind's
name vocabulary times the number of usable terminal attributes on its leaf kind
(`country.population` is excluded on every route: it leaks through the capital's
passage, same number by construction, so the gate rejects it everywhere).
`capacity_report()` sums that over the train route pool, which is what says
whether a requested share is reachable at all before any seeds are spent.

CAPACITY IS A PROPERTY OF THE VOCABULARY THE BASE SET WAS BUILT AGAINST, so
`--legacy-vocab` moves it as well as the mint. The 4x name-pool widening
(commit 566f4d2) took 4-hop capacity from 2030 to 8030; a base built before it --
`data/sft.jsonl`, `sft_r0`, `sft_r35`, `sft_r0_big`, `sft_r35_big` -- has to be
topped up from the SAME pools it was drawn from, or the new 4-hop slice carries
names the rest of the set cannot contain and the mix stops being the only
variable. Reading the live 8030 for such a base would also wave through a share
that is unreachable, so the flag is checked before any seeds are spent.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.agent.loop import LoopConfig  # noqa: E402
from atr.data.build_sft import ExportConfig, export  # noqa: E402
from atr.data.rejection import FilterConfig, filter_and_balance  # noqa: E402
from atr.data.teacher import collect_oracle  # noqa: E402
from atr.tasks.generator import (_LEAF_ATTR, _REL, _ROUTES_TRAIN,  # noqa: E402
                                 generate)
from atr.tools import world as W  # noqa: E402
from atr.tools.legacy_vocab import legacy_vocab  # noqa: E402

def _vocab() -> dict[str, int]:
    """Distinct NAMES each entity kind can carry across all worlds -- the pools
    build_world draws from, not the 6/8/10 it instantiates per world.

    Read at CALL time, never snapshotted at import: `legacy_vocab()` narrows these
    lists in place, and a module-level dict would have been computed against the
    live pools before the context manager ever ran, silently reporting the widened
    8030 for a set whose real 4-hop capacity is 2030."""
    return {
        "person": len(W.PERSON_FIRST) * len(W.PERSON_LAST),
        "organisation": len(W.ORG_WORDS) * len(W.ORG_KIND),
        "work": len(W.WORK_WORDS),
        "city": len(W.NATIONS),
        "country": len(W.NATIONS),
        "feature": len(W.GEO_FEATURES),
    }
# leaf kind -> terminal attributes that actually survive the gates.
_DEAD = {("country", "population")}   # leaks through the capital's passage


def _leaf_kind(route: list[str]) -> str:
    return _REL[route[-1]][2]


def _usable_attrs(kind: str) -> int:
    return sum(1 for a in _LEAF_ATTR.get(kind, {}) if (kind, a) not in _DEAD)


def capacity_report() -> dict[int, int]:
    """Distinct questions each hop family can mint, summed over the train routes."""
    vocab = _vocab()
    out = {}
    for hops, routes in sorted(_ROUTES_TRAIN.items()):
        total = 0
        for r in routes:
            total += vocab.get(_REL[r[0]][0], 0) * _usable_attrs(_leaf_kind(r))
        out[hops] = total
    return out


def _order(rows: list[dict], idxs: list[int]) -> list[int]:
    return sorted(idxs, key=lambda i: hashlib.sha1(
        rows[i]["meta"]["task_id"].encode()).hexdigest())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_r0_max.jsonl")
    ap.add_argument("--out", default="data/sft_r0_max_4h.jsonl")
    ap.add_argument("--mix", default="2:0.25,3:0.30,4:0.45")
    ap.add_argument("--seed-start", type=int, default=100_000,
                    help="disjoint from the base build (0..72k) and from dev (900k+)")
    ap.add_argument("--draws", type=int, default=6000,
                    help="4-hop seeds to draw for the top-up")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy-vocab", action="store_true",
                    help="narrow the name pools to their pre-widening prefixes; "
                         "required for any base built before commit 566f4d2")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    target = {int(k): float(v) for k, v in (p.split(":") for p in args.mix.split(","))}
    rng = random.Random(args.seed)

    ctx = legacy_vocab() if args.legacy_vocab else contextlib.nullcontext()
    if args.legacy_vocab:
        print("[vocab] pre-widening name pools (24x20 people, 14x8 orgs) -- "
              "capacity AND the top-up mint both read them")
    with ctx:
        _build(args, target, rng)


def _build(args, target, rng) -> None:
    cap = capacity_report()
    print("distinct-question capacity per hop family (head vocabulary x usable "
          "terminal attributes, summed over the train routes):")
    for h, c in cap.items():
        print(f"  {h}-hop: {c}")
    ceiling = min(cap[h] / s for h, s in target.items() if s > 0)
    print(f"set ceiling at this mix: {ceiling:,.0f} records "
          f"(binding family: {min(target, key=lambda h: cap[h] / target[h])}-hop)")

    rows = [json.loads(l) for l in
            Path(args.base).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_hop = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_hop[r["meta"]["difficulty"]].append(i)
    n_total = len(rows)
    print(f"\n[base] {args.base}: {n_total} records, hop mix "
          f"{dict(sorted((h, len(v)) for h, v in by_hop.items()))}")

    want = {h: int(round(s * n_total)) for h, s in target.items()}
    want[max(want)] += n_total - sum(want.values())      # absorb the rounding
    print(f"[target] {want}  = {sum(want.values())} records")
    for h in sorted(want):
        if want[h] > cap[h]:
            raise SystemExit(f"{h}-hop asks for {want[h]} records against a capacity "
                             f"of {cap[h]} -- unreachable at any number of seeds.")
        print(f"  {h}-hop {want[h]:>5} = {want[h] / cap[h]:>5.1%} of its capacity "
              f"(base was {len(by_hop[h]) / cap[h]:.1%})")

    kept: list[dict] = []
    need_new = 0
    for h in sorted(by_hop):
        have = by_hop[h]
        if want[h] <= len(have):
            take = _order(rows, have)[: want[h]]
            kept.extend(rows[i] for i in take)
            print(f"[{h}-hop] keeping {len(take)} of {len(have)} base records")
        else:
            kept.extend(rows[i] for i in have)
            need_new = want[h] - len(have)
            print(f"[{h}-hop] keeping all {len(have)} base records, "
                  f"minting {need_new} more")

    seen_q = {" ".join(next(m["content"] for m in r["messages"]
                            if m["role"] == "user").split()).lower() for r in rows}

    if need_new:
        print(f"\n[top-up] drawing {args.draws} 4-hop seeds from {args.seed_start}")
        tasks = generate(args.draws, seed_start=args.seed_start,
                         mix={"musique_4hop": 1.0})
        recs = collect_oracle(tasks, cfg=LoopConfig(max_steps=args.max_steps),
                              progress=False)
        n_ok = sum(c.success for _, _, c in recs)
        print(f"[top-up] oracle success {n_ok}/{len(recs)} ({n_ok / len(recs):.1%})")
        # Same filter the base build ran, minus target_mix (one family here).
        pool, fsum = filter_and_balance(recs, FilterConfig(
            max_per_task=1, max_per_shape=2000, target_mix=None, seed=args.seed))
        # The shape cap is per (task_type, route) and never bound on the base
        # build; raising the 4-hop share is exactly the change that could make it
        # bind, so report it rather than let a silent drop absorb the difference.
        capped = fsum["reasons"].get("drop:shape_cap", 0)
        print(f"[top-up] shape cap (2000 per route) dropped {capped} "
              f"{'-- it is now binding, raise it or widen the route pool' if capped else '(not binding)'}")
        # dedupe_by_question ran WITHIN the new pool; it cannot see the base set,
        # so the cross-set collision is removed here. A question already carrying a
        # label in the base file must not come back with another one.
        fresh = [r for r in pool
                 if " ".join((r[0].prompt or "").split()).lower() not in seen_q]
        print(f"[top-up] {len(pool)} kept by the filters, {len(fresh)} of them new "
              f"against the base questions")
        if len(fresh) < need_new:
            raise SystemExit(f"only {len(fresh)} new 4-hop records from {args.draws} "
                             f"seeds, need {need_new}; raise --draws.")
        rng.shuffle(fresh)
        fresh = fresh[:need_new]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "new.jsonl"
            export(fresh, p, ExportConfig(oracle_rationale=True))
            kept.extend(json.loads(l) for l in
                        p.read_text(encoding="utf-8").splitlines() if l.strip())

    rng.shuffle(kept)
    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")

    hops = collections.Counter(r["meta"]["difficulty"] for r in kept)
    print(f"\nwrote {args.out}: {len(kept)} records, hop mix "
          f"{dict(sorted(hops.items()))} = "
          f"{'/'.join(f'{hops[h] / len(kept):.0%}' for h in sorted(hops))}")

    if not args.no_audit:
        print(f"\n=== auditing {args.out} ===")
        rc = subprocess.call([sys.executable, "tests/audit_sft.py", args.out], cwd=REPO)
        if rc != 0:
            raise SystemExit(f"DEFECTS PRESENT in {args.out} -- do not train on it.")


if __name__ == "__main__":
    main()
