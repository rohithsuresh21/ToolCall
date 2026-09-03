"""
Multi-turn GRPO over whole tool-use trajectories, upgraded with:

  * GiGPO step-level credit assignment (F3): episode advantages stay group-relative,
    but every assistant TURN additionally gets a micro-advantage from anchor-state
    grouping -- turns whose rendered conversation prefix is shared across the G
    same-seed rollouts are compared directly. Critic-free, no extra rollouts, and
    it degrades exactly to plain GRPO when states never repeat.
  * DAPO dynamic sampling (F4): over-sample task groups and keep only the ones
    whose rollouts actually produced signal (non-zero advantage), refilling until
    the batch is live or a generation-batch cap is hit.
  * Overlong handling (F5): episodes truncated mid-generation are soft-penalised
    in the reward and loss-masked here.
  * Multiplicative tool-efficiency (F7): within a successful set, fewer calls rank
    higher (OTC-PO style). Additive penalties still handle failed episodes.
  * Best-checkpoint tracking (F6): whenever the held-out canary runs, the highest
    dev_f1 checkpoint is kept in `<out>/best`. Selection is on dev_f1 alone: the
    official eval is 2/3/4-hop retrieval scored by token-F1 against a gold string,
    so F1 is the score and there is no second axis worth trading it against.
    dev_success and dev_necessity are logged as diagnostics only.

Units (unchanged from the original design):

  * one sample  = one full episode (several assistant turns)
  * one group   = G episodes on the SAME task seed (same world, same gold answer)
  * advantage   = episode-relative + omega * anchor-state-relative, broadcast to
                  that turn's assistant tokens only

Everything that is not an assistant token is masked out of the objective.

Practical notes:
  * Start from the SFT checkpoint, never from base.
  * Keep the group on one task seed.
  * Watch `frac_dead_groups` AND `gen_batches`: if dynamic sampling constantly
    needs its cap, the task mix is too hard/easy for the policy -- re-weight the
    curriculum rather than raising the learning rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from ..agent.backends import Backend, SamplingParams
from ..agent.chatml import assistant_spans
from ..agent.loop import LoopConfig, Trajectory, run_episodes
from ..tasks.generator import DEFAULT_MIX, dev_set, generate
from ..tasks.schema import Task
from ..tasks.verifiers import score as score_traj
from ..tools.adapter import get_registry
from .reward import RewardConfig, compute_reward, group_advantages, scale_by_efficiency


@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    adapter: str | None = None            # start from the SFT LoRA
    out_dir: str = "artifacts/grpo-run"
    env: str = "builtin"

    # --- sampling ---
    group_size: int = 8                   # G rollouts per task
    tasks_per_step: int = 8               # live groups targeted per optimiser step
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20                       # was silently dropped before FIX-2
    max_new_tokens: int = 512
    max_steps_per_episode: int = 10
    max_len: int = 4096
    rollout_chunk: int = 8                # max rollout episodes per generate() call
                                          # (bounds VRAM; no chunking -> CUDA OOM)
    repeat_guard: int = 3                 # F14: env feedback during training rollouts;
                                          # official-style runs should use 0 (see configs/eval.json)

    # --- optimisation ---
    steps: int = 300
    lr: float = 1e-6                      # full FT; use ~2e-5 with LoRA
    lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28           # asymmetric ("clip-higher") keeps exploration alive
    inner_epochs: int = 1
    micro_batch: int = 2
    kl_beta: float = 0.03                 # low beta; the SFT checkpoint is the anchor
    std_normalise: bool = True
    grad_clip: float = 1.0

    # --- F3 GiGPO ---
    gigpo: bool = True
    gigpo_omega: float = 0.30             # weight of the step-level micro advantage

    # --- F4 dynamic sampling ---
    dynamic_sampling: bool = True
    batch_multiplier: int = 3             # first pass samples this many extra tasks
    max_gen_batches: int = 4              # hard cap on refill rounds per step

    # --- F5 overlong ---
    mask_truncated: bool = True           # drop mid-generation-cut episodes from the gradient

    # --- F7 efficiency ---
    efficiency_lambda: float = 0.15       # 0 disables the multiplicative scaling

    # --- Lane B ---
    void_turn_filter: bool = True         # SimpleTIR: drop episodes with a no-op turn
    log_entropy: bool = True              # per-token action entropy; disable if it OOMs

    # --- curriculum ---
    task_mix: dict = field(default_factory=lambda: dict(DEFAULT_MIX))
    curriculum: bool = True               # stage difficulty easy -> default -> hard
    # curriculum_feedback closes the loop from the (already-computed) dead-group
    # diagnostic back into sampling: when the rolling frac_dead_groups exceeds
    # dead_group_threshold, _stage_mix pulls the curriculum position f back
    # toward CURRICULUM_EASY instead of ramping purely on step count.
    #
    # Default ON: it automates exactly the manual remedy the design notes already
    # prescribe ("above ~0.5 the mix is too hard, re-weight the curriculum toward
    # difficulty 2-3"), and the pullback is inert below the threshold, so a healthy
    # run is bit-identical to the step-only ramp. Set false to reproduce a run made
    # before this knob existed, or to compare against a step-only baseline.
    curriculum_feedback: bool = True
    dead_group_window: int = 5            # rolling window of frac_dead_groups fed back
    dead_group_threshold: float = 0.5     # same number the README advises a human to watch
    seed_start: int = 1_000_000
    seed_span: int = 400_000
    seed: int = 0
    log_every: int = 1
    save_every: int = 50
    eval_every: int = 50                  # >0 runs the held-out dev canary every N steps
    eval_per_type: int = 3                # dev tasks per family when the canary runs


# ---------------------------------------------------------------------------
class _PolicyBackend(Backend):
    """Generation that shares weights with the module being optimised."""

    name = "policy"

    def __init__(self, model, tok, cfg: GRPOConfig):
        self.model, self.tok, self.cfg = model, tok, cfg
        self.use_chatml = True
        # Rollouts are generated in sub-batches of at most this many episodes.
        # Without chunking, _PolicyBackend.generate feeds the WHOLE live set
        # (esp. with dynamic_sampling's batch_multiplier) into one generate()
        # call, whose KV cache + logits blow well past the VRAM budget
        # (observed CUDA OOM at ~44 GiB on a 47 GiB RTX 6000). Chunking bounds
        # the peak activation footprint regardless of group_size/tasks_per_step.
        self.rollout_chunk = max(1, getattr(cfg, "rollout_chunk", 8))

    def generate(self, batch, tools=None, sp=None, ids=None):
        import torch
        sp = sp or SamplingParams()
        outs: list[str] = []
        chunk = self.rollout_chunk
        for start in range(0, len(batch), chunk):
            chunk_batch = batch[start:start + chunk]
            texts = [self._render(m, tools) for m in chunk_batch]
            self.tok.padding_side = "left"
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            # Rollouts must run in clean inference mode. During training the
            # policy is in train() with gradient checkpointing ON, both of
            # which corrupt sampling: train dropout degrades output and
            # grad-checkpointing forces use_cache=False, which together produce
            # gibberish instead of tool calls. Dropout is only ever active
            # because we are mid-training; here we temporarily switch to eval +
            # no-checkpoint for a correct, cached forward, then restore the
            # exact previous state.
            was_training = self.model.training
            cache = getattr(self.model.config, "use_cache", None)
            gc_enabled = getattr(self.model, "_is_gradient_checkpointing", None)
            if was_training:
                self.model.eval()
            if gc_enabled:
                self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            try:
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc, max_new_tokens=sp.max_tokens, do_sample=sp.temperature > 0,
                        temperature=max(sp.temperature, 1e-5), top_p=sp.top_p,
                        top_k=getattr(sp, "top_k", None),
                        pad_token_id=self.tok.pad_token_id)
            finally:
                self.model.config.use_cache = cache
                if gc_enabled:
                    self.model.gradient_checkpointing_enable()
                if was_training:
                    self.model.train()
            from .grpo import _cut  # local import keeps this file self-contained
            for i in range(len(chunk_batch)):
                new = gen[i][enc["input_ids"].shape[1]:]
                outs.append(_cut(self.tok.decode(new, skip_special_tokens=True), sp.stop))
        return outs


def _cut(text: str, stops: Sequence[str]) -> str:
    best, hit = len(text), None
    for s in stops:
        i = text.find(s)
        if 0 <= i < best:
            best, hit = i, s
    return text[:best] + hit if hit else text


# ---------------------------------------------------------------------------
def sync_vllm_weights(llm, model) -> bool:
    """
    Push updated HF weights into a live vLLM engine so rollouts stay on-policy.

    This touches vLLM internals and the path moves between releases, so it is
    guarded. If it returns False, either (a) use the HF rollout path, which is
    slower but always correct, or (b) checkpoint to disk every N steps and
    restart the engine.
    """
    try:
        sd = model.state_dict()
        if hasattr(model, "merge_and_unload"):
            sd = model.merge_and_unload().state_dict()
        worker = llm.llm_engine.model_executor.driver_worker
        worker.model_runner.model.load_weights(((k, v) for k, v in sd.items()))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[grpo] vLLM weight sync unavailable ({type(e).__name__}: {e}); "
              f"falling back to HF rollouts")
        return False


# ---------------------------------------------------------------------------
class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.registry = get_registry(cfg.env)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if cfg.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.adapter, is_trainable=True)
        elif cfg.lora:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        self.model.cuda() if torch.cuda.is_available() else None
        self.model.gradient_checkpointing_enable()
        self.model.train()

        self.ref = None
        if cfg.kl_beta > 0:
            self.ref = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            if cfg.adapter:
                # KL must measure drift from the SFT policy, not from base (FIX-1).
                from peft import PeftModel
                self.ref = PeftModel.from_pretrained(self.ref, cfg.adapter)
            if torch.cuda.is_available():
                self.ref.cuda()
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr,
            betas=(0.9, 0.95), weight_decay=0.0)
        self.backend = _PolicyBackend(self.model, self.tok, cfg)
        self.reward_cfg = RewardConfig()
        self.history: list[dict] = []
        self.best: dict | None = None          # F6: best canary result so far
        # rolling frac_dead_groups fed into _stage_mix by curriculum_feedback;
        # maintained in train(), capped at dead_group_window entries
        self.dead_frac_window: list[float] = []

    # -- 1. sample -------------------------------------------------------
    # Difficulty here is CHAIN LENGTH -- the only axis left once the eval is
    # 2/3/4-hop retrieval. Easy leans on 2-hop, hard on 4-hop, and _stage_mix
    # interpolates through cfg.task_mix in the middle.
    #
    # These previously listed compute/db_lookup/db_aggregate/doc_lookup/multi_hop/
    # multi_hop_discount/distractor/recovery -- eight families deleted in the 8->5
    # tool reduction, still carrying positive weight. _lerp_mix unions its keys, so
    # the first 30% of any curriculum run drew from them and generate() raised
    # KeyError on the first unlucky draw. generate() now rejects such a mix up
    # front with a message that names the offending families.
    # Both tiers sum to exactly 1.0. _lerp_mix renormalises, so a tier summing to
    # less than 1 still lerps to the right SHAPE, but the endpoint at f=0 is then
    # not the literal tier -- which is what the curriculum-feedback test asserts.
    CURRICULUM_EASY = {
        "musique_2hop": 0.60, "musique_3hop": 0.30, "musique_4hop": 0.10,
    }
    CURRICULUM_HARD = {
        "musique_2hop": 0.20, "musique_3hop": 0.35, "musique_4hop": 0.45,
    }

    @staticmethod
    def _lerp_mix(a: dict, b: dict, t: float) -> dict:
        keys = set(a) | set(b)
        raw = {k: (1 - t) * a.get(k, 0.0) + t * b.get(k, 0.0) for k in keys}
        s = sum(raw.values()) or 1.0
        return {k: v / s for k, v in raw.items()}

    def _rolling_dead_mean(self) -> float:
        """Mean frac_dead_groups over the feedback window (empty window -> 0)."""
        w = self.dead_frac_window or [0.0]
        return sum(w) / len(w)

    def _curriculum_fraction(self, step: int) -> float:
        """Effective curriculum position f fed into _stage_mix.

        With curriculum_feedback ON and the rolling mean of frac_dead_groups above
        dead_group_threshold, f is pulled back toward CURRICULUM_EASY (f -> 0)
        proportionally to how far over the threshold the recent dead-group fraction
        sits -- so the sampler retreats toward easier tasks instead of ramping into
        CURRICULUM_HARD while the policy is producing mostly dead groups. With the
        knob OFF (default), or with the rolling mean below the threshold, this is
        exactly the step-only ramp (step / steps), so existing behaviour is
        provably unchanged.
        """
        f = step / max(1, self.cfg.steps)
        if not self.cfg.curriculum_feedback:
            return f
        mean = self._rolling_dead_mean()
        if mean <= self.cfg.dead_group_threshold:
            return f
        overshoot = (mean - self.cfg.dead_group_threshold) / max(
            1e-9, self.cfg.dead_group_threshold)
        pullback = min(1.0, overshoot)
        return f * (1.0 - pullback)

    def _stage_mix(self, step: int) -> dict:
        """easy (f<0.3) -> default mix (0.3-0.65) -> hard (f>0.7), linear in between."""
        f = self._curriculum_fraction(step)
        if f < 0.30:
            return self._lerp_mix(self.CURRICULUM_EASY, self.cfg.task_mix, f / 0.30)
        if f < 0.70:
            return dict(self.cfg.task_mix)
        t = min(1.0, (f - 0.70) / 0.15)
        return self._lerp_mix(self.cfg.task_mix, self.CURRICULUM_HARD, t)

    def sample_tasks(self, n: int | None = None) -> list[Task]:
        n = n or self.cfg.tasks_per_step
        base = self.rng.randrange(self.cfg.seed_start,
                                  self.cfg.seed_start + self.cfg.seed_span)
        mix = self._stage_mix(len(self.history) + 1) if self.cfg.curriculum else self.cfg.task_mix
        return generate(n, seed_start=base, mix=mix,
                        rng_seed=self.rng.randrange(1 << 30))

    # -- 1b. rollout -----------------------------------------------------
    def rollout(self, tasks: Sequence[Task]) -> list[dict]:
        """G episodes per task; returns one record per episode."""
        expanded: list[Task] = []
        for t in tasks:
            for g in range(self.cfg.group_size):
                d = t.to_dict()
                d["task_id"] = f"{t.task_id}#g{g}"
                expanded.append(Task.from_dict(d))

        sp = SamplingParams(temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                            top_k=self.cfg.top_k,
                            max_tokens=self.cfg.max_new_tokens)
        loop_cfg = LoopConfig(max_steps=self.cfg.max_steps_per_episode,
                              repeat_guard=self.cfg.repeat_guard)
        trajs = run_episodes(expanded, self.registry, self.backend, loop_cfg, sp)

        out = []
        for t, j in zip(expanded, trajs):
            card = score_traj(t, j)
            r, parts = compute_reward(t, card, self.reward_cfg)
            # SimpleTIR: a turn that produced neither a tool call nor a final
            # answer ends the episode as "empty_turn" -- that episode is void.
            void = self.cfg.void_turn_filter and j.stop_reason == "empty_turn"
            out.append({"group": t.task_id.split("#g")[0], "task": t, "traj": j,
                        "card": card, "reward": r, "parts": parts, "void": bool(void)})
        return out

    # -- 2. advantages ---------------------------------------------------
    @staticmethod
    def _prefix_hashes(r: dict) -> list[str]:
        """One hash per assistant turn: the conversation prefix the model saw
        BEFORE emitting that turn. Identical prefixes = identical anchor state."""
        msgs = r["traj"].messages
        hashes, L = [], 2                     # system + user lead the transcript
        for step in r["traj"].steps:
            if L <= len(msgs):
                blob = json.dumps(msgs[:L], sort_keys=True, default=str)
                hashes.append(hashlib.md5(blob.encode()).hexdigest())
            L += 1 + len(step.tool_calls)
        return hashes

    def assign_advantages(self, records: list[dict]) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in records:
            if not r["void"]:
                groups.setdefault(r["group"], []).append(r)

        lam = self.cfg.efficiency_lambda
        dead = 0
        for gid, rs in groups.items():
            rewards = [r["reward"] for r in rs]
            if lam > 0:                        # F7 multiplicative efficiency
                rewards = scale_by_efficiency(
                    rewards,
                    [r["card"].num_calls for r in rs],
                    [bool(r["card"].success) for r in rs],
                    lam)
            ep = group_advantages(rewards, std_normalise=self.cfg.std_normalise)

            micro: list[list[float]] = [[0.0] * len(r["traj"].steps) for r in rs]
            if self.cfg.gigpo:
                anchors: dict[str, list[tuple[int, int]]] = {}
                for i, r in enumerate(rs):
                    for k, h in enumerate(self._prefix_hashes(r)):
                        anchors.setdefault(h, []).append((i, k))
                for members in anchors.values():
                    if len(members) < 2:
                        continue               # no peers at this state -> no signal
                    adv = group_advantages([rewards[i] for i, _ in members],
                                           std_normalise=self.cfg.std_normalise)
                    for (i, k), a in zip(members, adv):
                        micro[i][k] = self.cfg.gigpo_omega * a

            live_group = False
            for i, r in enumerate(rs):
                ta = [ep[i] + m for m in micro[i]]
                r["turn_advantages"] = ta
                r["advantage"] = ep[i]
                if any(abs(x) > 1e-8 for x in ta):
                    live_group = True
            r_live = live_group
            for r in rs:
                r["live"] = r_live
            if not live_group:
                dead += 1

        return {"n_groups": len(groups), "dead_groups": dead,
                "frac_dead_groups": round(dead / max(1, len(groups)), 3),
                "frac_void_episodes": round(
                    sum(r["void"] for r in records) / max(1, len(records)), 3)}

    # -- 3. tensorise ----------------------------------------------------
    def encode(self, records: list[dict]) -> list[dict]:
        import torch
        encode = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        tools = self.registry.schemas()
        out = []
        for r in records:
            if not r.get("live"):
                continue                       # dead/void group -> no gradient anyway
            if self.cfg.mask_truncated and r["card"].detail.get("truncated"):
                continue                       # F5: cut-off generation teaches noise
            msgs = r["traj"].messages
            ids, spans = assistant_spans(msgs, tools, encode)
            if not spans or len(ids) > self.cfg.max_len:
                continue
            mask = [0] * len(ids)
            tadv = [0.0] * len(ids)
            tas = r.get("turn_advantages") or []
            for j, (s, e) in enumerate(spans):
                a = tas[j] if j < len(tas) else 0.0
                for i in range(s, min(e, len(ids))):
                    mask[i] = 1
                    tadv[i] = a
            if sum(mask) == 0:
                continue
            out.append({"input_ids": torch.tensor(ids), "action_mask": torch.tensor(mask),
                        "token_adv": torch.tensor(tadv), "n_action": sum(mask)})
        return out

    # -- 4. logprobs -----------------------------------------------------
    def _logprobs(self, model, input_ids, attn, want_entropy=False):
        import torch
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1]
        tgt = input_ids[:, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        out = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        if not want_entropy:
            return out
        p = lp.exp()
        ent = -(p * lp).sum(-1)               # [B, L-1] full-distribution entropy
        del p, lp
        return out, ent

    def _pad(self, items):
        import torch
        n = max(len(x["input_ids"]) for x in items)
        pad = self.tok.pad_token_id
        ii = torch.full((len(items), n), pad, dtype=torch.long)
        am = torch.zeros((len(items), n), dtype=torch.long)
        ac = torch.zeros((len(items), n), dtype=torch.long)
        tv = torch.zeros((len(items), n), dtype=torch.float)
        for i, x in enumerate(items):
            L = len(x["input_ids"])
            ii[i, :L] = x["input_ids"]
            am[i, :L] = 1
            ac[i, :L] = x["action_mask"]
            tv[i, :L] = x["token_adv"]
        dev = next(self.model.parameters()).device
        return ii.to(dev), am.to(dev), ac.to(dev), tv.to(dev)

    # -- 5. one optimisation step ---------------------------------------
    def optimise(self, items: list[dict]) -> dict:
        import torch
        if not items:
            return {"loss": 0.0, "n_items": 0}
        mb = self.cfg.micro_batch
        chunks = [items[i:i + mb] for i in range(0, len(items), mb)]

        # behaviour-policy logprobs (exact: the policy has not moved yet)
        olds = []
        ent_sum, ent_tok = 0.0, 0
        with torch.no_grad():
            for ch in chunks:
                ii, am, ac, _ = self._pad(ch)
                if self.cfg.log_entropy:
                    old, ent = self._logprobs(self.model, ii, am, want_entropy=True)
                    m = ac[:, 1:].float()
                    ent_sum += float((ent * m).sum())
                    ent_tok += int(m.sum())
                    del ent
                else:
                    old = self._logprobs(self.model, ii, am)
                olds.append(old.detach())

        refs = [None] * len(chunks)
        if self.ref is not None:
            with torch.no_grad():
                refs = []
                for ch in chunks:
                    ii, am, ac, _ = self._pad(ch)
                    refs.append(self._logprobs(self.ref, ii, am).detach())

        total_action = sum(x["n_action"] for x in items)
        stats = {"loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "n_items": len(items)}
        if self.cfg.log_entropy:
            stats["entropy"] = round(ent_sum / max(1, ent_tok), 4)

        for _ in range(self.cfg.inner_epochs):
            self.opt.zero_grad(set_to_none=True)
            for ci, ch in enumerate(chunks):
                ii, am, ac, tv = self._pad(ch)
                logp = self._logprobs(self.model, ii, am)
                mask = ac[:, 1:].float()
                adv = tv[:, 1:]                    # per-token advantage (GiGPO-aware)
                old = olds[ci]
                ratio = torch.exp(logp - old)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps_high) * adv
                pg = -torch.min(s1, s2)
                loss = (pg * mask).sum() / max(1, total_action)

                if refs[ci] is not None:
                    d = refs[ci] - logp
                    kl = torch.exp(d) - d - 1.0          # k3, always >= 0
                    loss = loss + self.cfg.kl_beta * (kl * mask).sum() / max(1, total_action)
                    stats["kl"] += float((kl * mask).sum().detach()) / max(1, total_action)

                loss.backward()
                stats["loss"] += float(loss.detach())
                with torch.no_grad():
                    clipped = ((ratio < 1 - self.cfg.clip_eps) |
                               (ratio > 1 + self.cfg.clip_eps_high)).float()
                    stats["clip_frac"] += float((clipped * mask).sum()) / max(1, total_action)

            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
            self.opt.step()
            stats["grad_norm"] = float(gn)
        return stats

    # -- 6. driver -------------------------------------------------------
    def collect_batch(self) -> tuple[list[dict], dict]:
        """F4 dynamic sampling: refill until enough live groups or the cap hits."""
        target = self.cfg.tasks_per_step
        pool: list[dict] = []
        seen: set[str] = set()
        gen_batches, discarded = 0, 0
        while gen_batches < max(1, self.cfg.max_gen_batches):
            n = target * (self.cfg.batch_multiplier if gen_batches == 0 else 2)
            tasks = self.sample_tasks(min(n, self.cfg.seed_span))
            records = self.rollout(tasks)
            self.assign_advantages(records)
            fresh = []
            for gid in dict.fromkeys(r["group"] for r in records):
                grp = [r for r in records if r["group"] == gid]
                if any(r.get("live") for r in grp):
                    if gid not in seen:
                        fresh.append(grp)
                else:
                    discarded += 1
            for grp in fresh:
                if len(seen) >= target:
                    break
                seen.add(grp[0]["group"])
                pool.extend(grp)
            gen_batches += 1
            if not self.cfg.dynamic_sampling or len(seen) >= target:
                break
        info = {"gen_batches": gen_batches, "discarded_groups": discarded,
                "live_groups": len(seen)}
        return pool, info

    @staticmethod
    def canary_accept(cand: dict, best: dict | None) -> bool:
        """F6 checkpoint acceptance: highest dev_f1 wins, full stop.

        Token-F1 against the gold string IS the judge's score, and the official
        eval is 2/3/4-hop retrieval only -- no no_tool family, and the judge never
        sees tool calls. So there is nothing here to guard against: selecting on
        anything but dev_f1 would optimise a metric that is not scored.

        This deliberately replaces an earlier dev_success veto. That guard existed
        to catch a policy buying F1 by calling tools on no_tool tasks (such an
        episode scores F1 1.0 when the answer is right, so F1 alone cannot see it).
        With no_tool absent from the official eval, the regression it protected
        against is not a regression. dev_success stays in the canary log as a
        DIAGNOSTIC -- useful for reading a run, never an input to selection."""
        return best is None or cand["dev_f1"] > best["dev_f1"]

    def run_dev_canary(self) -> dict:
        """Held-out canary on a small balanced dev set.

        dev_f1 is the only number selection reads (see canary_accept). dev_success
        and dev_necessity come back alongside it as DIAGNOSTICS -- they are worth
        having in history.jsonl when reading a run, and they are not inputs to any
        decision. Note the dev set is balanced across all five families while the
        official eval is 2/3/4-hop only, so dev_f1 here is not the judge's number;
        it is a consistent internal proxy for ranking checkpoints against each
        other. The final-selection pass is where the mix should match the judge."""
        from ..eval.harness import aggregate, evaluate, load_eval_config
        ecfg = load_eval_config()
        tasks = dev_set(n_per_type=self.cfg.eval_per_type)
        sp = SamplingParams(temperature=ecfg["temperature"],
                            max_tokens=ecfg["max_new_tokens"])
        loop_cfg = LoopConfig(max_steps=ecfg["max_steps"],
                              repeat_guard=ecfg["repeat_guard"])
        cards, _ = evaluate(tasks, self.backend, env=self.cfg.env, cfg=loop_cfg,
                            sp=sp, progress=False)
        rep = aggregate(cards)
        return {"dev_f1": rep["overall"]["final_f1"],
                "dev_success": rep["overall"]["success"],
                "dev_necessity": rep["overall"]["necessity_ok"]}

    def train(self):
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=str))

        for step in range(1, self.cfg.steps + 1):
            t0 = time.time()
            records, dinfo = self.collect_batch()
            ginfo = self.assign_advantages(records)     # idempotent; refresh stats
            items = self.encode(records)
            ostats = self.optimise(items)

            # feed the live dead-group diagnostic into the rolling window used by
            # curriculum_feedback; keep at most dead_group_window entries
            self.dead_frac_window.append(ginfo.get("frac_dead_groups", 0.0))
            if len(self.dead_frac_window) > max(1, self.cfg.dead_group_window):
                self.dead_frac_window.pop(0)

            feedback = self.cfg.curriculum_feedback
            dg_mean = self._rolling_dead_mean()
            pulled_back = (feedback and dg_mean > self.cfg.dead_group_threshold)

            rewards = [r["reward"] for r in records] or [0.0]
            row = {
                "step": step,
                "reward_mean": round(statistics.fmean(rewards), 4),
                "reward_max": round(max(rewards), 4),
                "success": round(sum(r["card"].success for r in records) / max(1, len(records)), 4),
                "final_f1": round(statistics.fmean([r["card"].final_f1 for r in records]), 4) if records else 0.0,
                "final_correct": round(sum(r["card"].final_correct for r in records) / max(1, len(records)), 4),
                "format_strict": round(sum(r["card"].format_strict for r in records) / max(1, len(records)), 4),
                "avg_calls": round(statistics.fmean([r["card"].num_calls for r in records]), 2) if records else 0.0,
                "trained_episodes": ostats["n_items"],
                "clip_frac": round(ostats.get("clip_frac", 0.0), 4),
                "grad_norm": round(ostats.get("grad_norm", 0.0), 3),
                **({"entropy": ostats["entropy"]} if "entropy" in ostats else {}),
                "kl": round(ostats.get("kl", 0.0), 5),
                "secs": round(time.time() - t0, 1),
                **dinfo,
                **{k: v for k, v in ginfo.items() if k != "frac_void_episodes"},
                "frac_void_episodes": ginfo.get("frac_void_episodes", 0.0),
                "dead_group_rolling_mean": round(dg_mean, 4),
                "curriculum_pulled_back": bool(pulled_back),
            }
            if self.cfg.eval_every and step % self.cfg.eval_every == 0:
                row.update(self.run_dev_canary())
                print(f"[grpo] canary step {step}: dev_f1={row['dev_f1']} "
                      f"dev_success={row['dev_success']} "
                      f"(train f1 {row['final_f1']})", flush=True)
                cand = {"step": step, "dev_f1": row["dev_f1"],
                        "dev_success": row["dev_success"],
                        "reward_mean": row["reward_mean"]}
                if self.canary_accept(cand, self.best):
                    self.best = cand
                    self.model.save_pretrained(out / "best")
                    self.tok.save_pretrained(out / "best")
                    (out / "best.json").write_text(json.dumps(cand, indent=2))
                    print(f"[grpo] new BEST checkpoint at step {step}", flush=True)
            self.history.append(row)
            if step % self.cfg.log_every == 0:
                print("[grpo] " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            with (out / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

            if self.cfg.save_every and step % self.cfg.save_every == 0:
                self.model.save_pretrained(out / f"step-{step}")
                self.tok.save_pretrained(out / f"step-{step}")

        self.model.save_pretrained(out / "final")
        self.tok.save_pretrained(out / "final")
        if self.best is not None:
            print(f"[grpo] done -> {out / 'final'}  (best canary dev_f1 "
                  f"{self.best['dev_f1']:.3f} at dev_success {self.best['dev_success']:.3f}, "
                  f"step {self.best['step']} -> {out / 'best'})")
        else:
            print(f"[grpo] done -> {out / 'final'}")


def cli():
    p = argparse.ArgumentParser()
    d = GRPOConfig()
    for f, v in asdict(d).items():
        if isinstance(v, dict):
            continue
        p.add_argument(f"--{f.replace('_', '-')}",
                       type=(lambda x: x.lower() == "true") if isinstance(v, bool)
                       else (str if v is None else type(v)), default=v)
    a = p.parse_args()
    GRPOTrainer(GRPOConfig(**vars(a))).train()


if __name__ == "__main__":
    cli()
