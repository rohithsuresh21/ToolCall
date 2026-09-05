# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ATR (Agentic Tool Reasoning): a pipeline that teaches a small Qwen3 model (1.7B for
development, 4B for the final artifact) to answer MuSiQue-style multi-hop questions by
chaining calls to **one** tool — a BM25 `search` over a seeded candidate passage set.
Task generation, tools, agent loop, eval and data build are **stdlib-only and CPU-only**;
the third-party deps in `requirements.txt` are needed only for training/serving
(`atr/train/*`, `atr/export_merge.py`, `scripts/modal_app.py`) and for the optional LLM
naturalization path.

## Commands

The `scripts/*.sh` files are bash and call `python3`. On Windows, use the underlying module
invocations directly (`python -m atr.cli ...`) — they are what the scripts wrap.

```bash
# sanity — run FIRST and after ANY change to tools, verifiers or the generator
python -m atr.cli eval --dev --n-per-type 6 --backend oracle   # MUST print 100% success
python -m atr.cli ablate --dev --n-per-type 10                 # per-skill end-to-end value

# tests (plain scripts, no pytest; run one by running its file)
python tests/test_pipeline.py          # end-to-end: determinism, oracle=1.0, masking, reward order
python tests/test_parser.py            # tool-call parsing + repair table
python tests/test_fix2.py              # FIX-2 invariants incl. dead-code assertions
python tests/test_answer_f1.py         # token-F1 judge metric + reward/verifier wiring
python tests/test_shortcut_filter.py   # shortcut + prefix-leak filters, train/dev route holdout
python tests/test_naturalize.py        # naturalization with a mock LLM (offline)
python tests/test_grpo_resume.py       # GRPO checkpoint/resume/wall-clock stop (CPU, no model)
python tests/audit_sft.py data/sft.jsonl  # built-set audit: psychic queries, unretrieved/early answers, label conflicts
python tests/verify_dataset.py         # needs artifacts/naturalized_passages_scaled.json
python tests/verify_naturalize_live.py # needs artifacts/naturalized_passages.json

# data
python -m atr.cli gen --n 500 --out artifacts/tasks.jsonl
python -m atr.cli collect --n 40 --backend oracle --samples-per-task 1 --out artifacts/raw_oracle.jsonl
python -m atr.cli build artifacts/raw_*.jsonl --rebalance --max-per-task 1 --max-per-shape 2000 --out artifacts/sft_candidate.jsonl
# ...then audit and promote to the COMMITTED training set (scripts/10_build_data.sh does both):
python tests/audit_sft.py artifacts/sft_candidate.jsonl && mv artifacts/sft_candidate.jsonl data/sft.jsonl

# judge task set: 54 public rows, rebuilt from the hub, byte-identical every run
python scripts/make_judge_tasks.py            # -> data/judge_tasks.jsonl

# GPU stages (bash scripts; env vars MODEL/ADAPTER/OUT override defaults)
scripts/20_sft.sh   scripts/30_grpo.sh   scripts/40_eval.sh   scripts/60_merge_export.sh

# GRPO inside a fixed reservation: stop cleanly at 3h30m, then continue next session
MAX_SECONDS=12600 bash scripts/30_grpo.sh
RESUME=artifacts/grpo-1p7b/final MAX_SECONDS=12600 bash scripts/30_grpo.sh
```

Tests exit non-zero on failure and print `PASS`/`FAIL` per check. `test_pipeline.py`,
`test_parser.py`, `test_fix2.py` and `test_answer_f1.py` are the CPU gate for any change.
(`test_shortcut_filter.py` and `test_naturalize.py` have no `sys.path` bootstrap of their own,
so run those as `PYTHONPATH=. python tests/...`.)

`--backend` takes `oracle | mock | hf:MODEL | vllm:MODEL | openai:MODEL`
(`atr/agent/backends.py:268`). `atr.train.sft` and `atr.train.grpo` generate their CLI flags
automatically from the `SFTConfig` / `GRPOConfig` dataclass fields — add a field and the flag
appears; booleans parse as the literal strings `true`/`false`.

## Architecture

**Seeded world → executable gold.** `build_world(seed)` (`atr/tools/world.py:99`) is a pure
function of the seed and builds an encyclopedic entity graph with Wikipedia-style passages.
`atr/tasks/generator.py` reads *the same world the tool reads*, so every task carries a
computed `gold` plus an `oracle_plan` (the reference `search` sequence). Nothing is
hand-labelled and nothing is frozen. **SFT-build seeds `0..500k`, dev seeds `900k+`, GRPO
rollout seeds `1M..1.4M`** (`GRPOConfig.seed_start`/`seed_span`) — keep all three disjoint or
gold answers leak. Note `test_pipeline.py` only checks the SFT-build range against dev; the
GRPO range is disjoint by construction and nothing asserts it. `generate()` draws from the train route pool and
`dev_set()` from the dev pool, so held-out route *shapes* also never appear in training.

**The oracle is the harness's own unit test — for the VERIFIERS only.** `oracle_backend`
replays each `oracle_plan` through the real registry. If it is not 100%, a verifier is broken and
every downstream number — including the RL reward — is wrong. That is why `00_sanity.sh` runs
before everything else. But it is **structurally blind to the data**: `MockBackend` replays the
plan and then emits `oracle_answer` from a lookup table, and never reads a tool result. It
therefore scores 100% whether or not a single retrieved passage contained the answer.
`atr/tasks/retrievability.py` is the missing half — it replays plans through the real registry
and checks the returned text. `assert_answer_retrievable` runs automatically inside
`eval --backend oracle` (disable with `--no-retrieval-check`) and in `test_pipeline.py`. It is
what caught the terminal-read bug below; run it whenever routes, passages or `_LEAF_ATTR` change.

**An L-hop route plans L+1 searches: L to walk the chain, one terminal read.** Walking the
relation chain ends on `chain[L-1]`'s passage, which NAMES the leaf but carries none of the
leaf's attributes — and the asked-for attribute *is* the answer. With L searches the gold string
was simply absent from the episode for ~50% of train tasks and 78% of the held-out 4-hop dev
shape (a `person` leaf, which nothing co-retrieves), and every one still scored 100% under the
oracle. The leaf's name is legitimate query material by that point — the previous hop just
returned it — so the read is not a second psychic search. `difficulty` stays the hop count;
`len(oracle_plan)` is `difficulty + 1`. Nothing derives a budget from `difficulty`, so call-bloat
limits and reward shaping scale off `len(oracle_plan)` automatically.

**A question string must carry one gold answer.** The world is a pure function of the seed but
the question never names the seed, and the entity pools are small (10 people, 8 orgs, 6
cities/countries), so the same sentence recurs across worlds with a different answer each time.
The pre-fix 5850-record build had 798 of 2600 distinct questions carrying conflicting labels
(worst: 49 answers for one question) across 68.6% of records — unlearnable by construction.
`FilterConfig.dedupe_by_question` keeps the first occurrence and drops the rest. `max_per_task`
does *not* catch this: `task_id` carries the seed, so each collision is a different task.
Disambiguating the prompt instead was rejected deliberately — the judge's questions carry no
world marker, so it would train a format the model never sees at eval.
`tests/audit_sft.py <path>` checks a built set for all four defects (psychic first query,
unretrieved answer, conflicting labels, prefix leakage) and works on any jsonl, including one
built elsewhere. It exits non-zero when any of them is present.

**The shortcut filter has two axes, and the single-search one only sees the first.**
`_is_shortcut_solvable` is MuSiQue disconnection filtering: fire ONE `search` on the full
question and reject if the gold answer comes back. That catches the chain that collapses to a
single query, and nothing else. The other leak is per-hop: BM25 returns whole passages and
co-retrieves neighbours, so the gold string routinely turns up in the top-k of a call *before*
the terminal read — the chain is truncatable, a model that stops early is still scored right,
and that is what it learns. Measured on unfiltered chains: **51.9% of 4-hop and 41.6% of both
2- and 3-hop** train tasks were answerable in fewer hops than their label claimed, and only a
minority of those were visible to the single-search test. `_is_prefix_leaky` replays the plan's
own pre-terminal calls and rejects on any hit; after it, all three families sit at 0%.

Two things make it cheap. Leakiness is a property of **(route, attribute)**, not of the route —
a country's population leaks through its capital's passage (same number by construction), its
official language does not — so `gen_musique` walks the leaf's other terminal attributes before
abandoning the route. And the train pool holds 8 routes per length, each with several terminal
attributes. Train yield is therefore **unchanged at 100% of seeds for every hop length**; the
cost lands on the *build*, where the surviving questions are less varied and `dedupe_by_question`
drops more (6000 seeds: 2457 → 1951 records, −20.6%, mix still exactly 40/30/30). The dev pool
has one route per length, so 12–28% of dev seeds become unmintable; `dev_set` retries seeds, so
the dev set still fills.

**Retrieval checks read parsed `title` + `text`, never the raw tool_response string.** The
rendered block also carries `doc_id` and the BM25 `score` float, and `norm_text` deletes
punctuation — so gold `1949` substring-matches `"score": 5.1949` and a clean set reports a
phantom early hit. `audit_sft._hit_texts` parses the payload; `_is_prefix_leaky` and
`_is_shortcut_solvable` read `title` and `text` together, because the title is what leaks when
the gold answer *is* an entity name.

**One seam for tools: `ToolRegistry`.** `atr/tools/registry.py` validates arguments against the
JSON schema, dispatches, and appends to `world.call_log`, which is what the verifiers and
metrics read back. Model-caused failures (unknown tool, bad args, `ToolError`) come back as
`{"error": ..., "message": ...}` payloads and **never raise** — recovering from them is a
trained skill. Argument coercion (e.g. `"3"` → `3`) happens but is *recorded*, so eval reports
`args_ok` separately from `args_strict_ok`. `atr/tools/adapter.py::get_registry` is the only
file to edit when the organizers' real tools arrive; `world.py` and `builtin.py` are the only
files that get replaced.

**The loop decides nothing.** `atr/agent/loop.py` parses, executes, returns — it never picks a
tool, rewrites arguments or plans. The only environment-side feedback is tool error payloads
and an optional repeat-guard (`repeat_guard=0` disables it). Episodes advance in lockstep, one
batched `generate()` per turn, because GRPO needs G rollouts per task per step.

**One renderer for training and inference: `atr/agent/chatml.py`.** Do not swap in
`tokenizer.apply_chat_template` — Qwen3's template strips `<think>` blocks from *previous*
assistant turns, silently desynchronising multi-turn training from multi-turn inference.
`assistant_spans()` is the masking contract: loss covers assistant content plus its
`<|im_end|>` and the newline `render()` writes after it — nothing else. (That trailing newline
is inert at inference: generation stops on `<|im_end|>`.) Verified over all 1951 records of
`data/sft.jsonl` with the real Qwen3 tokenizer: 0 prefix-stability violations and the
`min(e, len(full_ids))` clamp never fires. Training on tool results teaches the model to hallucinate
`<tool_response>` blocks instead of calling the tool — that looks fine on the loss curve and
scores zero on executable tasks.

**Parsing is lenient, export is canonical.** `atr/agent/parser.py` accepts a wide repair table
(markdown fences, `parameters` alias, single quotes, truncated JSON, python-call syntax) and
records `repairs` / `strict_format`. `atr/data/build_sft.py` then rewrites every exported
assistant turn into exactly one canonical form (`<think>` + `<tool_call>`, or
`<final_answer>`), which is what makes `format_strict` at eval time a measurement of the model
rather than of the parser's leniency.

**The family mix is hop-only, and the mix is the single source of truth.** The organizers
confirmed the official eval is 2/3/4-hop retrieval only -- no `no_tool`, no `unanswerable` -- so
`DEFAULT_MIX` is 40/30/30 across the three hop families and the other two carry weight `0.0`.
Their generators stay registered in `GENERATORS` on purpose: the code records what the harness
can mint, and the spec is one number away from changing back. `active_families()` derives the
live set from the mix, and **`generate()` and `dev_set()` both read it** -- `dev_set` used to
iterate `GENERATORS` directly, so zeroing a weight silenced a family in training while the dev
set (and therefore the GRPO canary) kept drawing it. `generate()` also rejects a mix naming a
family with no generator, which is how the stale GRPO curriculum mix (eight families deleted in
the 8->5 tool reduction, still weighted) is now caught up front instead of as a mid-run KeyError.

**Routes are enumerated from the relation graph, not hand-listed.** `_ROUTES_TRAIN` /
`_ROUTES_DEV_ONLY` hold every type-valid relation sequence whose leaf kind has a terminal
attribute and which resolves without revisiting an entity in >=90% of worlds -- 9 at each of
lengths 2, 3 and 4, split 8 train / 1 dev. `_resolve_chain` enforces the acyclicity guard on the
resolved ENTITY ids, which is the only check that catches
`person_org>org_city>city_country>country_city`: four distinct relation names that revisit the
same city. Relation-keyword checks miss it, and it was labelled 4-hop while being a 2-hop
question. Before this, ~60% of 3/4-hop tasks were such loops, `by_difficulty` was measuring a
fake population, and rejection.py was silently discarding them as `repeated_call`.

**`organisation.founder` is load-bearing.** It was declared in `world.py` and left `None`, so the
`org -> person` edge did not exist. Without it every acyclic route sinks into `city <-> country`,
which are mutual inverses here (every city IS its country's capital), capping the graph at
exactly ONE acyclic 4-chain and ZERO 5-chains. Populating it gives 9 routes at each length.
`_passage()` already rendered the attribute and already listed it in `links`, so no downstream
change was needed.

**Family-scoped assertions must fail loudly on an empty family.** `all()` over an empty list is
`True`, so a check scoped to one family silently turns green the moment that family stops being
minted. `test_pipeline.py` and `verify_dataset.py` gate such checks on `active_families()`, assert
non-emptiness when the family is active, and assert absence plus print an explicit `SKIP`/`NOTE`
when it is not.

**Verification is per-objective, not pass/fail.** `atr/tasks/verifiers.py` fills a `ScoreCard`
(`atr/tasks/schema.py:56`) with `success` plus separate necessity / selection / args /
recovery / format axes and a single most-informative `failure_mode`. When `success` stops
moving, the breakdown tells you which data batch to build next. `success` is deliberately
stricter than "answer correct": calling a tool on a `no_tool` task fails even when the answer
is right.

**The headline metric is `final_f1`, not `success`.** The official judge scores SQuAD/MuSiQue
token-level F1 against a short gold string, so `answer_f1` (`atr/tasks/schema.py`) reproduces that
normaliser exactly -- punctuation deleted rather than split on, articles dropped, MULTISET token
intersection -- and `atr/train/reward.py` scales `w_final_correct` by it. `success` and
`final_correct` stay as internal booleans for the per-objective breakdown. F1 precision is now the
only brevity pressure on the final answer: the old character-count verbosity penalty was removed
because it double-counted the same signal in units the judge does not use. `p_no_answer` is NOT
redundant with it -- a pure-F1 objective collapses into answer-avoidance, and that penalty is the
mitigation. F6 best-checkpoint selection (`GRPOTrainer.canary_accept`) is **pure `dev_f1`**:
`cand["dev_f1"] > best["dev_f1"]`, nothing else. The official eval is 2/3/4-hop retrieval scored
by token-F1 against a gold string -- there is no `no_tool` family in it and the judge never sees
tool calls -- so F1 is the score and there is no second axis worth trading it against. An earlier
`dev_success` veto was removed for that reason: it guarded a metric that is not scored, and at a
15-task canary it could freeze `best/` at an early low-F1 checkpoint on one lucky measurement.
`dev_success` and `dev_necessity` are still logged by the canary as **diagnostics** -- read them
when reading a run, never feed them into a decision.

The canary's dev set is balanced across the ACTIVE families, and since `active_families()` is
now the judge's 2/3/4-hop only, the canary mix IS the judge's mix -- `test_answer_f1.py` pins
that (all canary tasks are hop families). It is still not the judge's NUMBER: the canary draws
synthetic dev worlds, not the public MuSiQue rows, and `eval_per_type` is 3, so `dev_f1` is a
consistent internal proxy for ranking checkpoints against each other. (This paragraph used to
say the canary was balanced across all five families; that was true before `dev_set()` started
reading `active_families()`, and re-zeroing a family's weight would restore the mismatch.)

**`target_mix` fails loudly on a starved family.** A family with a positive target share and
zero kept trajectories used to drop out of the feasibility `min()` and let the mix renormalise
silently -- asking for 40/30/30 and getting 57/43/0 looked like success. `filter_and_balance` now
raises. Set a share to `0.0` to build without a family deliberately.

**A GRPO checkpoint is the adapter PLUS `trainer_state.pt`, and nothing else is a
resume point.** `save_pretrained` writes weights. Continuing from weights alone restarts
AdamW with zero moments, re-draws the task seeds already trained on, resets the curriculum
position (`_stage_mix` reads `len(self.history)`) and forgets which checkpoint was winning —
none of which raises, and none of which appears in `history.jsonl`. So `_save_checkpoint`
writes optimiser state, the step counter, `best`, the history rows, the python+torch RNG
states and the dead-frac window alongside every `step-N/` and `final/`, and
`--resume-from` on a directory without that file **refuses** rather than silently cold-starting
(it points you at `--adapter`, which is the flag that legitimately starts fresh from weights).
On resume the policy continues from the checkpoint while the KL reference still anchors on
`--adapter`, the SFT policy: re-anchoring it on the resumed checkpoint would let drift compound
across restarts, which is what FIX-1 removed. `--max-seconds` checks the clock BEFORE each step
and stops one step short of the budget, because a step costs ~140s and the reservation does not
care that we were mid-`optimise()`. `best/` stays weights-only on purpose — it is a selection
artifact, not a resume point. On resume `history.jsonl` is rewritten from the checkpoint's own
history: it is opened in append mode, so the aborted tail of the previous session would
otherwise sit in the file as duplicate step numbers that every downstream reader mis-plots.
`tests/test_grpo_resume.py` covers all of it on CPU with a stand-in model.

**Filtering on `success` alone is the classic mistake.** `atr/data/rejection.py` also rejects
call-bloat, repeats and over-represented shapes, while explicitly *protecting* trajectories
that hit a tool error and recovered — naive success-filtering under-samples those because
error paths are longer and rarer.

**Reward shaping only matters within a group** (`atr/train/reward.py`): GRPO normalises
advantages across the G rollouts of one task, so a term that fires identically on all G
contributes exactly nothing. `w_final_correct` gives partial credit on failed tasks to keep
hard groups from producing zero gradient. There is deliberately **no reward for using a tool**
— adding one reintroduces tool calls on `no_tool` tasks. Watch `frac_dead_groups` in the GRPO
logs: above ~0.5 the mix is too hard, so re-weight the curriculum toward difficulty 2–3 rather
than raising the learning rate.

**Frozen eval config.** `configs/eval.json` is read by `load_eval_config()` and shared by the
CLI, the GRPO dev canary, `scripts/modal_app.py` and `scripts/judge_eval.py`, so internal numbers
match official scoring conditions. Explicit CLI flags override it (`atr/cli.py::_frozen`); those
flags default to `None` precisely so "not passed" stays distinguishable from an explicit value.

Two ways this claim was false and is now enforced. `judge_eval.py` read none of it -- its own
argparse defaults (temperature 0.2, max_steps 8) meant the one script scoring the judge's data
sampled differently from every other eval in the repo. And `top_p` was declared in the config and
passed by nobody: every `SamplingParams(...)` built from the frozen dict listed `temperature` and
`max_tokens` only, so the dataclass default 0.8 silently won over the configured 1.0 in the CLI,
the canary, modal and judge alike. **A knob in this file that no call site forwards is worse than
no knob**, because it reads as a controlled variable in every report. If you add a field here,
add it to `_frozen()` and to all four call sites.

**vLLM's `max_lora_rank` is read from the adapter, never assumed.** It defaults to 16 and raises
at engine construction when the adapter's `r` exceeds it, so serving any adapter this repo trains
(`SFTConfig.lora_r` 32; `50_sft_4b.sh` r=64) under the default fails outright -- `40_eval.sh`, the
vLLM dev eval, could not run against a trained adapter at all. `VLLMBackend` now derives it via
`lora_rank_of()` reading the adapter's own `adapter_config.json`, rounded up to a supported bucket
(8/16/32/64/128/256), with an explicit `max_lora_rank=` override for a hub id or merged checkpoint
whose config it cannot read. Hardcoding 32 would have fixed today and broken silently the next
time someone edited `--lora-r`; `tests/test_lora_rank.py` pins that, and needs neither vLLM nor a GPU.

**One matcher for the score: `answer_f1`.** The repo has three answer comparisons and they
disagree by multiples on the same string -- `match_answer` (substring containment; drives the
internal `success` boolean), `norm_text` equality (strictly harsher than the judge), and
`answer_f1` (token F1, the judge's metric). `judge_eval.py` used to headline the first two and
import the third nowhere, which is how an `acc_exact` on the 54 real MuSiQue rows came to be
compared against a `final_f1` on the 60 synthetic dev tasks and reported as a "5.67x backend
ratio" between HF and vLLM. It was never a backend effect: the two numbers came from different
scripts running different task sets under different metrics. **Only `answer_f1` may be quoted as
a score**; `acc_exact` survives in `judge_eval.py` as an explicitly labelled diagnostic.

**Passage naturalization is offline and opt-in.** `atr/data/naturalize.py` rewrites passage
prose via an LLM under fact-preservation checks and writes a cache keyed `"seed:doc_id"`. It
**never** runs inside `build_world()` or at train/eval time — the cache is minted once
(`scripts/70_naturalize_passages.sh`, or `atr.data.naturalize_local` /
`naturalize_retrieved_local` against a local Ollama), then loaded via `--cache <path>` →
`load_naturalized_loader()` → `build_world(seed, text_loader=...)`. Entities, attributes and
gold answers are unaffected by design; `test_naturalize.py` and `verify_naturalize_live.py`
assert that isolation.

## Conventions

- `artifacts/` is gitignored except `artifacts/sft_sample.jsonl`. Committed data lives in
  `data/sft.jsonl` and `data/judge_tasks.jsonl`. SFT records are `{messages, tools, meta}`
  JSONL, UTF-8.
- **Training inputs come from `data/`, never `artifacts/`, and the gate enforces it.**
  `20_sft.sh` and `50_sft_4b.sh` source `scripts/lib_data_gate.sh` and call
  `require_clean_dataset`, which runs `tests/audit_sft.py` and refuses to start on
  `DEFECTS PRESENT`, on a path under gitignored `artifacts/`, or when the audit cannot run at
  all (a broken interpreter must not read as clean data). `10_build_data.sh` builds to
  `artifacts/sft_candidate.jsonl` and promotes to `data/sft.jsonl` only after the same audit
  passes. This exists because a stale pre-fix `artifacts/sft.jsonl` shadowed the committed clean
  set and was trained on: 40.4% / 46.0% / 51.7% of its 2/3/4-hop records were answerable before
  the terminal read. `eval --backend oracle` scores such a set 100% by construction, so the
  audit is the only thing that can catch it.
- Tests are plain scripts with a `check(cond, label)` helper and `sys.exit(1)` on failure —
  match that style rather than introducing pytest.
- Module docstrings carry the design rationale ("why", not "what"), and several encode
  invariants the tests assert. `test_fix2.py` asserts that specific dead code stays deleted,
  so reintroducing a removed symbol breaks it on purpose.
