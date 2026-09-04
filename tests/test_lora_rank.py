"""vLLM max_lora_rank is derived from the adapter, never assumed.

vLLM's `max_lora_rank` defaults to 16 and raises at engine construction when the
adapter's own `r` exceeds it. `SFTConfig.lora_r` is 32 and `50_sft_4b.sh` trains
r=64, so serving ANY adapter this repo produces under the default fails outright
-- which is why `scripts/40_eval.sh` (the vLLM dev eval) could not run against a
trained adapter at all.

The rank is read from the adapter's own `adapter_config.json` rather than
hardcoded to 32: hardcoding fixes today and breaks silently the next time someone
edits `--lora-r`. These checks pin the reading, the round-up to a supported
bucket, and the fallbacks -- none of which need vLLM or a GPU to verify.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atr.agent.backends import _max_lora_rank_for, lora_rank_of

FAILS = 0


def check(cond, label):
    global FAILS
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILS += 1


def mk(tmp, name, payload):
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / "adapter_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(d)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # the two ranks this repo actually trains: 1.7B r=32, 4B r=64
        a32 = mk(tmp, "r32", {"r": 32, "lora_alpha": 64, "peft_type": "LORA"})
        a64 = mk(tmp, "r64", {"r": 64, "lora_alpha": 128, "peft_type": "LORA"})
        a8 = mk(tmp, "r8", {"r": 8})
        a17 = mk(tmp, "r17", {"r": 17})
        a128 = mk(tmp, "r128", {"r": 128})
        nocfg = mk(tmp, "nocfg", None)
        bad = mk(tmp, "bad", None)
        (Path(bad) / "adapter_config.json").write_text("{not json", encoding="utf-8")

        check(lora_rank_of(a32) == 32, "reads r=32 from adapter_config.json")
        check(lora_rank_of(a64) == 64, "reads r=64")
        check(lora_rank_of(nocfg) is None, "missing adapter_config.json -> None")
        check(lora_rank_of(bad) is None, "malformed adapter_config.json -> None (no crash)")
        check(lora_rank_of(None) is None, "no adapter -> None")

        # THE BUG: r=32 under vLLM's default 16 fails outright.
        check(_max_lora_rank_for(a32) == 32,
              "r=32 -> max_lora_rank 32 (the 1.7B adapter; was failing at the default 16)")
        check(_max_lora_rank_for(a64) == 64, "r=64 -> 64 (the 4B adapter)")
        check(_max_lora_rank_for(a8) == 8, "r=8 -> 8, not padded up to 16")
        check(_max_lora_rank_for(a17) == 32, "r=17 -> 32 (rounds up to a supported bucket)")
        check(_max_lora_rank_for(a128) == 128, "r=128 survives a future rank bump")
        check(_max_lora_rank_for(nocfg) == 16, "unreadable config -> vLLM's default 16")
        check(_max_lora_rank_for(None) == 16, "no adapter -> default")

        try:
            _max_lora_rank_for(mk(tmp, "r999", {"r": 999}))
            check(False, "rank beyond vLLM's largest bucket raises")
        except ValueError as e:
            check("exceeds vLLM's largest" in str(e),
                  "rank beyond vLLM's largest bucket raises a clear error")

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
