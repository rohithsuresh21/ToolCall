"""
Generation backends. All of them expose one method:

    generate(list_of_message_lists, tools, **sampling) -> list[str]

Batched by construction, because both GRPO rollouts and eval are throughput
bound and a per-example loop is the difference between a 20-minute eval and a
3-hour one.

Available:
  MockBackend    -- no model. A scripted/heuristic agent used to test the whole
                    pipeline on CPU. Also doubles as a sanity ceiling: if the
                    mock does not score ~1.0 on the dev set, a verifier is wrong,
                    not the model.
  HFBackend      -- transformers, for small-batch debugging and LoRA eval.
  VLLMBackend    -- offline vLLM, for rollouts and full evals.
  OpenAIBackend  -- any OpenAI-compatible /v1/chat/completions endpoint: the
                    teacher during distillation, or your own served checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 768
    stop: list[str] = field(default_factory=lambda: ["</tool_call>", "</final_answer>"])

    def as_dict(self) -> dict:
        return dict(temperature=self.temperature, top_p=self.top_p,
                    max_tokens=self.max_tokens, stop=list(self.stop))


class Backend:
    name = "base"
    use_chatml = True
    enable_thinking = False
    tok = None

    def _render(self, messages: list[dict], tools: list[dict] | None) -> str:
        """Single rendering path shared by every model backend.

        `use_chatml=True` (default) uses this repo's renderer, the same one the
        SFT tokeniser uses -- train/serve consistency by construction. Set it to
        False only if you must match a serving stack you do not control, and if
        you do, re-tokenise your SFT data the same way.
        """
        if self.use_chatml:
            from .chatml import render
            return render(messages, tools, add_generation_prompt=True)
        return self.tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)

    # Whether generation was cut at a stop string; the loop re-appends it so the
    # transcript stays well-formed for the next turn and for SFT export.
    def generate(self, batch: Sequence[list[dict]], tools: list[dict] | None = None,
                 sp: SamplingParams | None = None, ids: list[str] | None = None) -> list[str]:
        """`ids` carries the per-episode task_id. Real backends ignore it; MockBackend
        uses it so that two tasks sharing a prompt but not a world stay distinct."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
class MockBackend(Backend):
    """
    A deterministic stand-in agent. It is NOT a model: it reads a per-task
    `oracle_plan` (a list of tool calls the task generator already knows) and
    replays it, optionally with injected mistakes. That makes it perfect for
    two jobs: proving the harness end-to-end on CPU, and generating controlled
    ablations ("what does 80% argument accuracy score end to end?").
    """

    name = "mock"

    def __init__(self, plans: dict[str, list[dict]] | None = None,
                 answers: dict[str, str] | None = None,
                 corrupt: Callable[[str, int, dict], dict | None] | None = None,
                 wrong_answer: str = "The answer is 424242.",
                 degrade_on_error: bool = True):
        self.plans = plans or {}
        self.answers = answers or {}
        self.corrupt = corrupt
        self.wrong_answer = wrong_answer
        self.degrade_on_error = degrade_on_error
        # episodes whose plan was disrupted -> they must NOT still produce the
        # gold answer, otherwise ablations measure nothing (the mock would be
        # answering from a lookup table rather than from tool results).
        self._degraded: set[str] = set()

    def _key(self, messages: list[dict]) -> str:
        for m in messages:
            if m["role"] == "user":
                return m["content"]
        return ""

    def generate(self, batch, tools=None, sp=None, ids=None) -> list[str]:
        out = []
        ids = ids or [None] * len(batch)
        for messages, tid in zip(batch, ids):
            key = tid if tid is not None else self._key(messages)
            plan = self.plans.get(key, [])
            step = sum(1 for m in messages if m["role"] == "assistant")

            # an unexpected tool error means the planned data never arrived
            if self.degrade_on_error and self.corrupt is not None:
                planned_errors = sum(1 for c in plan[:step] if "__expect_error__" in c)
                seen_errors = sum(1 for m in messages
                                  if m["role"] == "tool" and '"error"' in m["content"])
                if seen_errors > planned_errors:
                    self._degraded.add(key)

            if step < len(plan):
                call = {k: v for k, v in plan[step].items() if not k.startswith("__")}
                if self.corrupt:
                    c = self.corrupt(key, step, dict(call))
                    if c is None:
                        self._degraded.add(key)
                        out.append(f"<final_answer>{self._answer(key)}</final_answer>")
                        continue
                    if c != call:
                        self._degraded.add(key)
                    call = c
                out.append("<tool_call>" + json.dumps(call) + "</tool_call>")
            else:
                out.append(f"<final_answer>{self._answer(key)}</final_answer>")
        return out

    def _answer(self, key: str) -> str:
        if key in self._degraded:
            return self.wrong_answer
        return self.answers.get(key, "unknown")


# ---------------------------------------------------------------------------
class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, adapter: str | None = None, device: str = "auto",
                 dtype: str = "bfloat16", enable_thinking: bool = False, use_chatml: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.enable_thinking = enable_thinking
        self.use_chatml = use_chatml
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def generate(self, batch, tools=None, sp=None, ids=None) -> list[str]:
        import torch
        sp = sp or SamplingParams()
        texts = [self._render(m, tools) for m in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=sp.max_tokens,
                do_sample=sp.temperature > 0, temperature=max(sp.temperature, 1e-5),
                top_p=sp.top_p, top_k=sp.top_k, pad_token_id=self.tok.pad_token_id)
        outs = []
        for i in range(len(batch)):
            new = gen[i][enc["input_ids"].shape[1]:]
            outs.append(_truncate_at_stop(self.tok.decode(new, skip_special_tokens=True), sp.stop))
        return outs


# vLLM only accepts a max_lora_rank from this set, and it must be >= the adapter's
# own r. Anything else raises at engine construction.
_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256)


def lora_rank_of(adapter: str | None) -> int | None:
    """The `r` a PEFT adapter was trained with, or None if it cannot be read.

    Read rather than hardcoded so a rank change does not silently reintroduce the
    failure this exists to prevent: vLLM's max_lora_rank defaults to 16 while
    `SFTConfig.lora_r` is 32 (and the 4B script trains r=64), so serving any of
    our adapters under the default blows up at engine construction. Hardcoding 32
    would fix today and break the next time someone edits --lora-r.
    """
    if not adapter:
        return None
    cfg = Path(adapter) / "adapter_config.json"
    if not cfg.is_file():
        return None                      # hub id, or a merged (non-PEFT) checkpoint
    try:
        r = json.loads(cfg.read_text(encoding="utf-8")).get("r")
        return int(r) if r else None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _max_lora_rank_for(adapter: str | None, default: int = 16) -> int:
    """Smallest vLLM-supported bucket that fits the adapter."""
    r = lora_rank_of(adapter)
    if r is None:
        return default
    for bucket in _VLLM_LORA_RANKS:
        if bucket >= r:
            return bucket
    raise ValueError(f"adapter rank r={r} exceeds vLLM's largest supported "
                     f"max_lora_rank ({_VLLM_LORA_RANKS[-1]})")


# ---------------------------------------------------------------------------
class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_id: str, adapter: str | None = None, enable_thinking: bool = False,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 8192,
                 use_chatml: bool = True, max_lora_rank: int | None = None, **kw):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        lora_kw = {}
        if adapter:
            rank = max_lora_rank if max_lora_rank is not None else _max_lora_rank_for(adapter)
            lora_kw["max_lora_rank"] = rank
            detected = lora_rank_of(adapter)
            print(f"[vllm] adapter {adapter}: r={detected if detected is not None else '?'} "
                  f"-> max_lora_rank={rank}"
                  + ("" if detected is not None else
                     "  (adapter_config.json unreadable; using the default -- pass "
                     "max_lora_rank= explicitly if the adapter is r>16)"))
        self.llm = LLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_lora=bool(adapter),
                       **lora_kw, **kw)
        self.adapter = adapter
        self.enable_thinking = enable_thinking
        self.use_chatml = use_chatml

    def generate(self, batch, tools=None, sp=None, ids=None) -> list[str]:
        from vllm import SamplingParams as VSP
        sp = sp or SamplingParams()
        texts = [self._render(m, tools) for m in batch]
        vsp = VSP(temperature=sp.temperature, top_p=sp.top_p, top_k=sp.top_k,
                  max_tokens=sp.max_tokens, stop=sp.stop, n=1,
                  include_stop_str_in_output=True)
        kw = {}
        if self.adapter:
            from vllm.lora.request import LoRARequest
            kw["lora_request"] = LoRARequest("adapter", 1, self.adapter)
        outs = self.llm.generate(texts, vsp, **kw)
        return [o.outputs[0].text for o in outs]


# ---------------------------------------------------------------------------
class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_workers: int = 8, timeout: float = 180.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                             api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
                             timeout=timeout)
        self.model = model
        self.max_workers = max_workers

    def _one(self, messages, sp: SamplingParams) -> str:
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=sp.temperature,
                    top_p=sp.top_p, max_tokens=sp.max_tokens, stop=sp.stop or None)
                txt = r.choices[0].message.content or ""
                return _reattach_stop(txt, sp.stop)
            except Exception as e:
                if attempt == 3:
                    return f"<final_answer>BACKEND_ERROR: {e}</final_answer>"
                time.sleep(2 ** attempt)
        return ""

    def generate(self, batch, tools=None, sp=None, ids=None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        sp = sp or SamplingParams()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda m: self._one(m, sp), batch))


# ---------------------------------------------------------------------------
def _truncate_at_stop(text: str, stops: list[str]) -> str:
    best = len(text)
    hit = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and i < best:
            best, hit = i, s
    return text[:best] + (hit or "") if hit else text


def _reattach_stop(text: str, stops: list[str]) -> str:
    """Providers strip the stop string; the parser wants it back."""
    for s in stops:
        opener = s.replace("</", "<")
        if opener in text and s not in text:
            return text + s
    return text


def load_backend(spec: str, **kw) -> Backend:
    """spec like 'vllm:Qwen/Qwen3-1.7B', 'hf:Qwen/Qwen3-4B', 'openai:gpt-4.1', 'mock'."""
    if spec == "mock":
        return MockBackend(**kw)
    kind, _, model = spec.partition(":")
    return {"hf": HFBackend, "vllm": VLLMBackend, "openai": OpenAIBackend}[kind](model, **kw)
