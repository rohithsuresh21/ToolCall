"""Token statistics for a built SFT set, with the REAL tokenizer and the repo's
own renderer.

    python scripts/token_stats.py data/sft_r0_max.jsonl [more.jsonl ...]

This reproduces exactly what `atr.train.sft.build_dataset` prints at the top of a
training run -- same `chatml.assistant_spans`, same masking contract, same
`--max-len` drop rule -- but without importing torch or peft, so the number can be
checked on a CPU box before a GPU hour is spent. It exists because that line is
quoted in every build's commit message and was previously re-derived by hand each
time.

It also re-runs the masking contract check: for every assistant turn, the span
must start exactly where the prefix render ends, and `min(e, len(full_ids))` must
never have to clamp. A violation there means training and inference disagree about
where an assistant turn begins, which costs accuracy silently.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.agent.chatml import assistant_spans, render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--spans", action="store_true",
                    help="also compute supervised-token share and the masking check "
                         "(several tokenizations per record, so much slower)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    for path in args.paths:
        rows = [json.loads(l) for l in
                Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        lens, sup, clamps = [], [], 0
        by_hop = collections.defaultdict(list)
        for r in rows:
            if args.spans:
                ids, spans = assistant_spans(r["messages"], r.get("tools"), encode)
                sup.append(sum(e - s for s, e in spans))
                clamps += sum(1 for s, e in spans if e > len(ids))
            else:
                ids = encode(render(r["messages"], r.get("tools"),
                                    add_generation_prompt=False))
            lens.append(len(ids))
            by_hop[r["meta"].get("difficulty")].append(len(ids))

        over = sum(1 for n in lens if n > args.max_len)
        print(f"\n{path}   {len(rows)} records")
        print(f"  tokens/example: mean {sum(lens) / len(lens):.0f}  max {max(lens)}  "
              f"min {min(lens)}")
        for h in sorted(by_hop, key=lambda x: (x is None, x)):
            v = by_hop[h]
            print(f"    {h}-hop: n {len(v):>5}  mean {sum(v) / len(v):>6.0f}  "
                  f"max {max(v):>5}")
        print(f"  over --max-len {args.max_len}: {over} "
              f"({'nothing is dropped' if over == 0 else 'THESE WOULD BE DROPPED'})")
        if args.spans:
            print(f"  supervised tokens: mean {sum(sup) / len(sup):.0f} "
                  f"({100 * sum(sup) / sum(lens):.2f}% of all tokens)")
            print(f"  masking contract: {clamps} span clamps "
                  f"({'clean' if clamps == 0 else 'VIOLATION'})")


if __name__ == "__main__":
    main()
