"""
Merge a trained LoRA adapter into its base model and export one submission-ready
artifact (F2, FIX-2).

Why this exists: SFT/GRPO save adapter-only checkpoints. The organisers' benchmark
and the Modal serving endpoint need ONE complete model directory -- merged weights,
tokenizer, generation config -- with nothing else to load at inference time.

What it does:
  1. loads the base model (bf16 by default),
  2. attaches the trained LoRA adapter,
  3. merge_and_unload() so the deltas become real weights,
  4. saves model + tokenizer + a frozen generation config + an audit manifest.

The frozen generation config pins non-thinking-mode decoding so the served model
answers deterministically and cannot drift into free-form sampling:
temperature/top_p/top_k disabled, greedy decode.

Runs on CPU or GPU; memory is just the base model. For Qwen3-4B bf16 that is ~8GB.

Usage:
  python -m atr.export_merge \
      --base Qwen/Qwen3-4B --adapter artifacts/grpo-4b/final --out artifacts/final-4b
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MergeConfig:
    base: str = "Qwen/Qwen3-4B"
    adapter: str = ""                     # path to the PEFT checkpoint to merge
    out: str = "artifacts/final-merged"
    dtype: str = "bfloat16"
    device: str = "cpu"                   # cpu is safest for a merge; no VRAM needed
    # --- frozen inference behaviour (PS: non-thinking mode) ---
    disable_thinking: bool = True         # strip <think> via chat template kwarg default
    max_new_tokens: int = 512


def merge_export(cfg: MergeConfig) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not cfg.adapter:
        raise SystemExit("--adapter is required (path to a PEFT/LoRA checkpoint)")

    torch_dtype = getattr(torch, cfg.dtype)
    print(f"[merge] loading base {cfg.base} ({cfg.dtype}, {cfg.device}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base, torch_dtype=torch_dtype, device_map=cfg.device)

    print(f"[merge] attaching adapter {cfg.adapter} ...", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, cfg.adapter)

    print("[merge] merging weights ...", flush=True)
    merged = model.merge_and_unload()
    merged.eval()

    tok = AutoTokenizer.from_pretrained(cfg.base)
    if cfg.disable_thinking:
        # Qwen3 honours this template kwarg; bake it in as the served default.
        tok.chat_template_kwargs_default = {"enable_thinking": False} if hasattr(
            tok, "chat_template_kwargs_default") else None

    gen = GenerationConfig(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    merged.generation_config = gen

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving -> {out}", flush=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    gen.save_pretrained(out)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": cfg.base,
        "adapter": str(Path(cfg.adapter).resolve()),
        "dtype": cfg.dtype,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": cfg.max_new_tokens,
            "mode": "non-thinking",
        },
        "notes": "Merged submission artifact. Load with vLLM/HF directly; "
                 "no adapter required at inference.",
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_params = sum(p.numel() for p in merged.parameters())
    size_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"[merge] done: {n_params/1e9:.2f}B params, {size_gb:.2f} GB on disk")
    return manifest


def cli():
    p = argparse.ArgumentParser()
    d = MergeConfig()
    for f, v in d.__dict__.items():
        p.add_argument(f"--{f.replace('_', '-')}",
                       type=(lambda x: x.lower() == "true") if isinstance(v, bool) else type(v),
                       default=v)
    a = p.parse_args()
    merge_export(MergeConfig(**vars(a)))


if __name__ == "__main__":
    cli()
