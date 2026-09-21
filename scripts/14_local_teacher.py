"""Mint teacher <think> blocks on a LOCAL 4-bit Qwen3-8B, resumably, on a laptop.

    python scripts/14_local_teacher.py --legacy-vocab --limit 20     # the test run
    python scripts/14_local_teacher.py --legacy-vocab                # the long run
    python scripts/14_local_teacher.py --dry-run                     # select + count, no model

Same arm as scripts/14_build_teacher_reasoning.py and the same decisions --
`select` picks the blocks, `user_message` builds the prefix-only prompt,
`violations` gates the answer, all of it in atr/data/reasoning_teacher.py. Only
the thing that produces the text changes: a 4-bit Qwen3-8B on one consumer GPU
instead of the Batch API. Nothing here decides anything the batch path does not,
which is what keeps the two caches interchangeable.

WHY 8B IN 4-BIT. The teacher only has to beat the student (Qwen3-1.7B / 4B) at the
one thing the template cannot do: weighing a same-kind distractor that is already
on screen. It is not being asked for knowledge -- the passages are in the prompt
and the gate rejects anything not in them -- so the constraint is "largest that
fits 8GB VRAM", and nf4 Qwen3-8B is ~5.4GB of weights, leaving room for a ~2k-token
prefix. If it OOMs, `--model Qwen/Qwen3-4B` is the fallback and still clears the
student.

THINKING IS OFF. Qwen3 is a hybrid-thinking model and `enable_thinking=False` is
passed to its chat template. With it on, a 70-word block costs several hundred
reasoning tokens first, which at ~20 tok/s is the difference between one night and
four. Any `<think>` the model emits regardless is stripped before gating -- the
gate rejects `<` and `>` outright, so an unstripped tag would read as a content
violation and burn a retry on a formatting artifact.

(Note this is `tokenizer.apply_chat_template` on the TEACHER, which is fine.
CLAUDE.md's prohibition is about rendering the STUDENT's episodes, where Qwen3's
template silently strips `<think>` from previous assistant turns and desynchronises
multi-turn training from multi-turn inference. Nothing here touches
atr/agent/chatml.py.)

CRASH SAFETY IS THE FORMAT. The batch path writes one JSON object at the end of a
batch; a laptop run is interrupted by a closed lid, a driver reset or a power cut,
and rewriting a whole JSON file per block would eventually truncate it mid-write
and lose everything minted so far. So this cache is JSONL, appended and fsynced one
line per decided block: an interrupted run loses at most the block in flight, and a
half-written trailing line is dropped on load with a warning rather than raising.
Line 0 is a header carrying the base fingerprint, for the same staleness argument
as the batch cache -- the keys are task ids and they resolve against ANY set built
from the same seeds, so a cache minted against another population would load in
silence.

RESUME IS BY KEY, NOT BY POSITION. Every key already in the cache is skipped on
start, including the ones that GAVE UP -- a block that failed the gate four times
is recorded with `think: null` so the next night does not spend four more attempts
on it. `--retry-failed` is the way back in once the prompt or the gate changes. The
builder reads a null as "keep the template", which is the same fallback the batch
path takes.

ATTEMPTS ARE COUNTED AND THE COUNT IS WIRED, for the reason atr/data/naturalize.py
recorded the hard way: it initialised `stats["retries"] = 0` with nothing to
increment it and reported exactly 0 retries on every run it ever made. The count
lives per block in the cache and is summed back from the cache, so it cannot drift
away from what happened.

DETERMINISM, as far as a local model carries it. Attempt 1 is greedy, so it is
reproducible for a given model and build. Retries must not re-roll the same dice,
so they sample -- seeded from sha1(task_id|step|attempt), never a live RNG, the
same rule atr/data/reasoning.py and atr/data/recovery.py follow. Reproducibility
still lives in the CACHE rather than in the sampler; the cache is the artifact that
gets committed.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atr.data.reasoning_teacher import (SYSTEM, TeacherConfig, cache_key,  # noqa: E402
                              fingerprint, rebuild_ok, repair_hint, select,
                              user_message, violations, walk)
from atr.tools.legacy_vocab import legacy_vocab  # noqa: E402

CACHE_KIND = "teacher-local-v1"
_THINK_TAG = re.compile(r"<think>.*?</think>", re.S)
_OPEN_THINK = re.compile(r"^\s*<think>.*", re.S)


# --- the append-only cache -------------------------------------------------

def load_cache(path: Path) -> tuple[dict, dict]:
    """(header, {key: record}). Tolerates a truncated FINAL line, because the
    process that wrote it may have been killed mid-append; anything earlier is
    corruption and raises, since that cannot have come from an interruption."""
    if not path.exists():
        return {}, {}
    header, blocks = {}, {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            if n == len(lines) - 1:
                print(f"[cache] dropping a truncated final line ({len(line)} bytes) "
                      f"-- an interrupted append; its block will be re-minted",
                      file=sys.stderr)
                continue
            raise SystemExit(f"{path}:{n + 1} is not JSON and is not the last line "
                             f"-- the cache is corrupt, not merely interrupted.")
        if rec.get("cache") == CACHE_KIND:
            header = rec
        elif "key" in rec:
            blocks[rec["key"]] = rec
    return header, blocks


def append(fh, rec: dict) -> None:
    """One line, flushed AND fsynced. Without the fsync the OS buffer can hold
    several minutes of a slow run, which is exactly the window a lid-close or a
    driver reset lands in."""
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


# --- the teacher -----------------------------------------------------------

class LocalTeacher:
    """Qwen3 through transformers, 4-bit by default. Weights are loaded in
    `load()` rather than `__init__` so --dry-run and a mistyped path cost nothing
    and need neither torch nor a download."""

    def __init__(self, model_id: str, quantise: bool = True,
                 max_new_tokens: int = 192):
        self.model_id = model_id
        self.quantise = quantise
        self.max_new_tokens = max_new_tokens
        self.tok = None
        self.model = None
        self.torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        t0 = time.time()
        print(f"[model] loading {self.model_id} "
              f"({'nf4' if self.quantise else 'fp16'}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        kw = {"device_map": {"": 0}}
        if self.quantise:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        try:                          # transformers renamed torch_dtype -> dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float16, **kw)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, **kw)
        self.model.eval()
        if torch.cuda.is_available():
            print(f"[model] ready in {time.time() - t0:.0f}s on "
                  f"{torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB allocated",
                  flush=True)
        else:
            print(f"[model] ready in {time.time() - t0:.0f}s -- NO CUDA DEVICE, "
                  f"this will run at minutes per block", file=sys.stderr, flush=True)

    def _render(self, turns: list[dict]) -> str:
        msgs = [{"role": "system", "content": SYSTEM}] + turns
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:             # a template without the hybrid switch
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, turns: list[dict], seed: int | None = None) -> str:
        torch = self.torch
        inputs = self.tok([self._render(turns)], return_tensors="pt").to(
            self.model.device)
        kw = {"max_new_tokens": self.max_new_tokens,
              "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id}
        if seed is None:
            kw["do_sample"] = False   # attempt 1: greedy, hence reproducible
        else:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        new = out[0][inputs["input_ids"].shape[1]:]
        return clean(self.tok.decode(new, skip_special_tokens=True))


def clean(raw: str) -> str:
    """Strip the artifacts of a chat model so the gate judges the PROSE.

    A retry spent on an unclosed `<think>` tag or a pair of quotation marks
    teaches nothing and buys nothing; a retry spent on an invented name is the
    whole point of having retries."""
    t = _THINK_TAG.sub(" ", raw or "")
    t = _OPEN_THINK.sub(" ", t)       # thinking that ran into the token budget
    t = " ".join(t.split())
    for lead in ("Reasoning:", "REASONING:", "Answer:", "Output:"):
        if t.startswith(lead):
            t = t[len(lead):].strip()
    return t.strip().strip('"').strip("'").strip()


def _seed_for(task_id: str, step: int, attempt: int) -> int:
    """sha1, never a live RNG -- the same contract as reasoning.py's phrasing."""
    return int(hashlib.sha1(
        f"{task_id}|{step}|{attempt}".encode()).hexdigest()[:8], 16)


# --- the run ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_r0.jsonl")
    ap.add_argument("--cache", default="data/teacher_cache_r0_local.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--min-distractors", type=int, default=1)
    ap.add_argument("--attempts", type=int, default=4,
                    help="total tries per block: 1 mint + 3 retries, after which "
                         "the block keeps its template")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="stop after N blocks are DECIDED this session (accepted "
                         "or given up); 0 means all of them")
    ap.add_argument("--legacy-vocab", action="store_true",
                    help="verify the Task reconstruction under the pre-widening "
                         "name pools; required for any base built before 566f4d2")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-mint the blocks recorded as gave-up as well")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in fp16 -- needs ~16GB and will OOM an 8GB card")
    ap.add_argument("--dry-run", action="store_true",
                    help="select, verify and report; load no model, write nothing")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = TeacherConfig(model=args.model, max_attempts=args.attempts,
                        min_distractors=args.min_distractors)

    base = Path(args.base)
    lines = [l for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    fp = fingerprint(base)
    by_hop = collections.Counter(r["meta"]["difficulty"] for r in rows)
    print(f"[base] {base}: {len(rows)} records, hop mix {dict(sorted(by_hop.items()))}, "
          f"fingerprint {fp}")

    # --- select ------------------------------------------------------------
    jobs, sel_by_hop, blocks = [], collections.Counter(), 0
    for r in rows:
        msgs = r["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        blocks += sum(1 for _ in walk(msgs))
        for b in select(msgs, cfg):
            sel_by_hop[r["meta"]["difficulty"]] += 1
            jobs.append({"task_id": r["meta"]["task_id"], "step": b["step"],
                         "question": question, "seen_norm": b["seen_norm"],
                         "difficulty": r["meta"]["difficulty"],
                         "messages": msgs, "block": b})
    print(f"[select] {len(jobs)} of {blocks} blocks = {len(jobs) / blocks:.1%}  "
          f"(>= {cfg.min_distractors} same-kind distractor alongside the rank-1 hit)")
    for h in sorted(sel_by_hop):
        print(f"           {h}-hop: {sel_by_hop[h]}")

    # --- verify the base is the population we think it is ------------------
    ctx = legacy_vocab() if args.legacy_vocab else contextlib.nullcontext()
    if args.legacy_vocab:
        print("[vocab] pre-widening name pools (24x20 people, 14x8 orgs)")
    seen_ids, ok, bad_ids = set(), 0, 0
    with ctx:
        for j in jobs:
            if j["task_id"] in seen_ids:
                continue
            seen_ids.add(j["task_id"])
            if rebuild_ok(j["task_id"], j["difficulty"], j["question"]):
                ok += 1
            else:
                bad_ids += 1
    print(f"[verify] Task reconstruction: {ok} ok, {bad_ids} failed of "
          f"{ok + bad_ids} distinct task ids")
    if bad_ids > ok:
        raise SystemExit(
            "more than half the records do not reconstruct from their task_id. "
            "For a base built before commit 566f4d2 pass --legacy-vocab; otherwise "
            "the base and the installed world generator disagree.")

    # --- the cache, and what is left to do ---------------------------------
    cache_path = Path(args.cache)
    header, cached = load_cache(cache_path)
    if header:
        if header.get("base_fingerprint") and header["base_fingerprint"] != fp:
            raise SystemExit(
                f"cache {cache_path} was minted against base "
                f"{header['base_fingerprint']}, but {base} fingerprints {fp}. The "
                f"task ids would still resolve, so this would silently paste "
                f"another population's reasoning into this set. Mint a new cache, "
                f"or point --base at the set it came from.")
        if header.get("model") and header["model"] != cfg.model:
            print(f"[cache] NOTE existing blocks were minted with "
                  f"{header['model']}, now running {cfg.model}; cached blocks are "
                  f"kept as-is", file=sys.stderr)

    gave_up_before = sum(1 for v in cached.values() if v.get("think") is None)
    done = {k for k, v in cached.items()
            if v.get("think") is not None or not args.retry_failed}
    todo = [j for j in jobs if cache_key(j["task_id"], j["step"]) not in done]
    print(f"[cache] {cache_path}: {len(cached)} keys "
          f"({len(cached) - gave_up_before} minted, {gave_up_before} gave up)")
    print(f"[todo]  {len(todo)} blocks remain of {len(jobs)}"
          + (f"; this session will decide at most {args.limit}" if args.limit else ""))

    if args.dry_run:
        print("\n[dry-run] no model loaded, nothing written")
        return
    if not todo:
        print("\nnothing to do -- every selected block is already in the cache.")
        return

    # --- mint --------------------------------------------------------------
    teacher = LocalTeacher(cfg.model, quantise=not args.no_4bit,
                           max_new_tokens=args.max_new_tokens)
    teacher.load()

    target = min(args.limit, len(todo)) if args.limit else len(todo)
    stats, reasons = collections.Counter(), collections.Counter()
    t0, n = time.time(), 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        if not header:
            append(fh, {"cache": CACHE_KIND, "base": str(base),
                        "base_fingerprint": fp, "model": cfg.model,
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        try:
            for j in todo:
                if n >= target:
                    break
                turns = [{"role": "user",
                          "content": user_message(j["messages"], j["block"])}]
                text, bad, attempts = "", ["empty"], 0
                while attempts < cfg.max_attempts:
                    attempts += 1
                    text = teacher(turns, seed=(None if attempts == 1 else
                                                _seed_for(j["task_id"], j["step"],
                                                          attempts)))
                    bad = violations(text, j["question"], j["seen_norm"])
                    if not bad:
                        break
                    # The repair turn names the offending token, so the retry is
                    # informed rather than a re-roll of the same dice.
                    turns = turns[:1] + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": repair_hint(bad)}]
                n += 1

                rec = {"key": cache_key(j["task_id"], j["step"]),
                       "task_id": j["task_id"], "step": j["step"],
                       "think": None if bad else text, "attempts": attempts,
                       "model": cfg.model}
                if bad:
                    rec["reason"] = bad[0]
                    reasons[bad[0].split(":")[0]] += 1
                    stats["gave_up"] += 1
                else:
                    stats["accepted"] += 1
                    stats["repaired"] += int(attempts > 1)
                stats["attempts"] += attempts
                append(fh, rec)

                el = time.time() - t0
                rate = n / el * 3600
                left = target - n
                tag = "GAVE UP" if bad else ("ok" if attempts == 1
                                             else f"ok after {attempts}")
                msg = (f"[{n}/{target}] {j['task_id']} step {j['step']}  {tag}  "
                       f"{rate:.0f} blk/h  {left} left this session "
                       f"(~{left / max(rate, 1e-9):.1f} h)")
                if bad or attempts > 1 or n == 1 or n % args.log_every == 0:
                    print(msg, flush=True)
                    if not bad:
                        print(f"      {text[:160]}", flush=True)
                else:
                    print(msg + "        ", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\n[interrupted] the cache is complete through the last decided "
                  "block; re-run the same command to continue.", file=sys.stderr)

    # --- report ------------------------------------------------------------
    el = max(time.time() - t0, 1e-9)
    rate = n / el * 3600
    print("\n--- local mint ---")
    print(f"  model               {cfg.model} "
          f"({'fp16' if args.no_4bit else 'nf4 4-bit'})")
    print(f"  decided this run    {n}")
    print(f"  accepted            {stats['accepted']} "
          f"({stats['repaired']} needed a retry)")
    print(f"  gave up (template)  {stats['gave_up']}")
    if n:
        print(f"  attempts/block      {stats['attempts'] / n:.2f}")
    if reasons:
        print("  rejections by reason (final attempt):")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<18} {v}")
    print(f"  elapsed             {el / 60:.1f} min  "
          f"({rate:.0f} blocks/hour, {el / max(n, 1):.1f} s/block)")

    _, cached = load_cache(cache_path)
    minted = sum(1 for v in cached.values() if v.get("think") is not None)
    remain = len(jobs) - len(cached)
    print(f"  cache               {cache_path}: {len(cached)} keys, {minted} minted")
    print(f"  REMAINING           {remain} of {len(jobs)} selected blocks"
          + (f"  (~{remain / rate:.1f} h at this rate)" if n and rate else ""))
    if remain:
        print("\nre-run the same command to continue; every cached key is skipped.")
    else:
        print("\ncomplete. Build the arm with:\n"
              f"  python scripts/15_build_teacher_arm.py --cache {cache_path} "
              f"--legacy-vocab")


if __name__ == "__main__":
    main()
