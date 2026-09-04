"""
Supervised fine-tuning on filtered tool-use trajectories.

Two things here are not boilerplate and are worth reading before you change
anything:

1. LOSS MASKING. Labels are set only on assistant-turn tokens (see
   chatml.assistant_spans). Tool results are masked out. If you train on tool
   results, the model learns to *write* tool output, and at inference it will
   happily hallucinate a plausible <tool_response> and answer from it instead of
   calling the tool. This is the single most common way a tool-use SFT run
   produces a model that looks great on loss and fails every executable task.

2. NO PACKING, and length-bucketed batches instead. Packing concatenates
   unrelated conversations across the attention window; for multi-turn agentic
   data that leaks one task's tool results into another task's context and
   teaches exactly the wrong thing about what "context" means.

Runs anywhere: `--lora` fits Qwen3-1.7B on a 24GB card and Qwen3-4B on 1xA100;
full fine-tune of 4B wants 1xH100 80GB with 8-bit optimiser, or 2x with ZeRO-2.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    data: str = "data/sft.jsonl"   # committed set; artifacts/ is gitignored
    out_dir: str = "artifacts/sft-run"
    max_len: int = 4096
    epochs: float = 2.0
    lr: float = 1e-5
    lora_lr: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 0
    eval_frac: float = 0.02
    save_steps: int = 200
    log_steps: int = 10
    # Per-step loss weighting to fix multi-hop collapse. Every <tool_call> turn is
    # a "step" that builds the chain (the anchor search, the next hop, ...). These
    # intermediate tool calls used to be washed out: the old ramp weighted the
    # LAST assistant turn most, which is just the trivial <final_answer>. Instead:
    #   * step_weight (>1) boosts each tool-call step, with hop 1 (the crucial
    #     anchor that must start from the real entity) boosted most.
    #   * answer_weight (<1) de-emphasises the final-answer turn.
    step_weight: float = 2.0
    answer_weight: float = 0.5


def build_dataset(cfg: SFTConfig, tok):
    from torch.utils.data import Dataset
    from ..agent.chatml import assistant_spans

    rows = [json.loads(l) for l in Path(cfg.data).read_text().splitlines() if l.strip()]
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    examples = []
    dropped = 0
    for r in rows:
        ids, spans = assistant_spans(r["messages"], r.get("tools"), encode)
        if len(ids) > cfg.max_len or not spans:
            dropped += 1
            continue
        asst_texts = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        labels = [-100] * len(ids)
        weights = [0.0] * len(ids)
        step_i = 0
        for turn_i, (s, e) in enumerate(spans):
            asst = asst_texts[turn_i]
            if "<tool_call>" in asst:
                step_i += 1
                if cfg.step_weight > 1.0:
                    w = 1.0 + (cfg.step_weight - 1.0) / step_i
                else:
                    w = 1.0
            elif "<final_answer>" in asst:
                w = cfg.answer_weight
            else:
                w = 1.0
            for i in range(s, min(e, len(ids))):
                labels[i] = ids[i]
                weights[i] = w
        examples.append({"input_ids": ids, "labels": labels, "weights": weights,
                         "n_turns": len(spans), "n_steps": step_i,
                         "meta": r.get("meta", {})})

    print(f"[data] {len(examples)} examples kept, {dropped} dropped (too long / no assistant turn)")
    if examples:
        tl = [len(e["input_ids"]) for e in examples]
        sup = [sum(1 for x in e["labels"] if x != -100) for e in examples]
        print(f"[data] tokens/example: mean {sum(tl)/len(tl):.0f} max {max(tl)}  |  "
              f"supervised tokens: mean {sum(sup)/len(sup):.0f} "
              f"({100*sum(sup)/sum(tl):.1f}% of all tokens)")

    class DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    random.Random(cfg.seed).shuffle(examples)
    n_eval = max(1, int(len(examples) * cfg.eval_frac)) if cfg.eval_frac else 0
    return DS(examples[n_eval:]), (DS(examples[:n_eval]) if n_eval else None)


def collate(batch, pad_id: int):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "weights": []}
    for b in batch:
        pad = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["weights"].append(b["weights"] + [0.0] * pad)
    return {k: torch.tensor(v, dtype=torch.float if k == "weights" else torch.long)
            for k, v in out.items()}


class WeightedTrainerMixin:
    """Token-weighted cross-entropy so the per-step weights (step_weight/answer_weight)
    actually do something."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        import torch
        import torch.nn.functional as F
        weights = inputs.pop("weights")
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        w = weights[:, 1:]
        mask = tgt != -100
        loss_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.clamp(min=0).reshape(-1), reduction="none").view(tgt.shape)
        loss = (loss_tok * mask * w).sum() / (mask * w).sum().clamp(min=1e-6)
        return (loss, out) if return_outputs else loss


def main(cfg: SFTConfig):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds, eval_ds = build_dataset(cfg, tok)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="sdpa")
    model.config.use_cache = False

    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets), task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    class T(WeightedTrainerMixin, Trainer):
        pass

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lora_lr if cfg.lora else cfg.lr,
        lr_scheduler_type="cosine",
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.save_steps,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
    )
    trainer = T(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    trainer.save_model(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[done] saved to {cfg.out_dir}")


def cli():
    p = argparse.ArgumentParser()
    d = SFTConfig()
    for f, v in d.__dict__.items():
        if isinstance(v, tuple):
            continue
        p.add_argument(f"--{f.replace('_', '-')}",
                       type=(lambda x: x.lower() == "true") if isinstance(v, bool) else type(v),
                       default=v)
    a = p.parse_args()
    cfg = SFTConfig(**{k: v for k, v in vars(a).items()})
    main(cfg)


if __name__ == "__main__":
    cli()
