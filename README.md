# ATR — Agentic Tool Reasoning for Small Language Models

> **Agentic Tool Reasoning (ATR)** is a self-contained pipeline that takes a small
> language model (`Qwen/Qwen3-1.7B` for development, `Qwen/Qwen3-4B` for the
> final system) and teaches it to **solve tasks by using one tool**: a BM25
> retrieval search over a candidate passage set, chained hop by hop to answer
> MuSiQue-style multi-hop questions.

This repository is the student edition of the problem statement: *make a small
model reason with tools, then ship it*. Everything runs on a laptop first (no GPU
until you are ready to bill one), is seeded and deterministic, and — most
importantly — every design decision is written down *next to the code that makes
it*, with the *why* attached.

```
atr/
  tools/      world.py  builtin.py  registry.py  sandbox.py  adapter.py   <- the environment
  agent/      parser.py  chatml.py  prompt.py  backends.py  loop.py
  tasks/      schema.py  generator.py  verifiers.py                        <- tasks + executable scoring
  data/       teacher.py  rejection.py  build_sft.py                       <- distillation pipeline
  train/      sft.py  reward.py  grpo.py  reward_audit.py
  eval/       harness.py
  cli.py
scripts/      00_sanity → 40_eval, plus modal_app.py  (bash pipeline)
tests/        test_pipeline.py  test_parser.py  test_fix2.py
```

---

## Start here (no GPU needed)

```bash
python -m atr.cli eval --dev --n-per-type 6 --backend oracle    # must print 100%
python -m atr.cli ablate --dev --n-per-type 10                  # what each skill is worth
```

The first command is the harness's own unit test. The **oracle** replays each
task's reference plan against the real tool. If it does not score 100%, a
**verifier** is broken — and every number downstream, including your RL reward,
is lying to you. Run it after every change to the tools or the task generator.

```bash
python -m atr.cli gen --n 500 --out artifacts/tasks.jsonl       # mint tasks
python -m atr.cli build artifacts/raw_oracle.jsonl --rebalance  # filter -> SFT jsonl
python tests/test_pipeline.py                                   # full pipeline suite
```

---

## The task: multi-hop retrieval with one tool

The benchmark (MuSiQue-style reading comprehension) collapses the tool set to a
**single action**: BM25 retrieval over a *provided* candidate set of
Wikipedia-style passages. There is no calculator, no database, no web search and
no page fetch. The whole problem is:

> **Given a nested multi-hop question and a set of candidate passages, retrieve
> the right passage one hop at a time and compose the answer.**

A 3-hop question looks like this:

```
What is the official language of the country that contains the city where the
company that employs the person who created Glass Rivers is headquartered?
```

Answering it requires **sequential, dependent retrieval**: the query for hop *2*
cannot even be written until the passage retrieved by hop *1* has been read. That
dependency — the number of hops — is exactly what small models fall off, so it is
exactly what the curriculum is built around.

`atr/tasks/generator.py` mints these questions programmatically. Nothing is
hand-labelled: every task ships with a **gold answer computed from the same world
the tool reads**, plus an **`oracle_plan`** — the reference sequence of `search`
calls. That makes the verifiers *executable* rather than string-matched against a
frozen answer key.

### Task families

| family | difficulty | tier | what it tests |
|---|---|---|---|
| `musique_2hop` | 2 | 2 | a 2-link chain; one search resolves it |
| `musique_3hop` | 3 | 3 | two dependent searches traverse a 3-link chain |
| `musique_4hop` | 4 | 4 | three dependent searches traverse a 4-link chain |
| `unanswerable` | 2 | 1 | the information is genuinely absent; the model must look, then *abstain* |
| `no_tool` | 0 | 0 | answerable from the model's own knowledge; any tool call is a necessity failure |

`no_tool` and `unanswerable` are cheap insurance against the two things small
models do by default: **call a tool for everything**, and **invent an answer
rather than abstain**. The default mix weights the set toward multi-hop work
(where the score is won) while keeping those negatives at a healthy share:

| family | share |
|---|---|
| `musique_2hop` | 22% |
| `musique_3hop` | 24% |
| `musique_4hop` | 18% |
| `unanswerable` | 11% |
| `no_tool` | 25% |

---

## The five design decisions that matter

**1. The environment is executable and seeded, not a fixture.**
`build_world(seed)` is a pure function of the seed — it deterministically builds
an encyclopedic world (countries, cities, people, organisations, works) with
interlinked Wikipedia-style passages. The generator reads the *same* world the
tool reads, so gold answers are computed, never frozen. Train on seeds `0..500k`,
hold out `900k+`. Nothing leaks, and you can mint 50k tasks for free.

**2. Verification is per-objective, not pass/fail.**
`success` is the headline, but `verifiers.py` reports tool-necessity, tool
selection, argument quality, error recovery and format strictness separately.
When success stops moving, the breakdown tells you which data batch to build
next. That feedback loop is the whole point of this repo.

**3. One renderer for training and inference (`agent/chatml.py`).**
Qwen3's own chat template strips ` thinking` blocks from *previous* assistant
turns, silently desynchronising multi-turn tool-use training from multi-turn
inference. Both paths go through the same small renderer, eliminating the largest
source of unexplained point loss in agentic fine-tunes.

**4. Loss is masked to assistant tokens only.**
Tool results are masked out of the SFT loss and out of the GRPO objective. Train
on tool results and the model learns to *write* `<tool_response>` blocks — at
inference it will hallucinate a plausible tool result and answer from it instead
of calling the tool. That failure looks fine on the loss curve and scores zero on
executable tasks.

**5. Nothing outside the model decides anything.**
The loop parses, executes, and returns. It never picks a tool, rewrites
arguments, or plans. The only environment-side feedback is tool error payloads
(normal behaviour) and an optional repeat-guard that tells the model it just made
the same call three times — set `repeat_guard=0` if the official benchmark
forbids even that.

---

## The pipeline

```bash
scripts/00_sanity.sh                              # verify the harness (must be 100%)
TEACHER=oracle scripts/10_build_data.sh            # collect -> filter -> SFT jsonl
scripts/20_sft.sh                                 # LoRA SFT on Qwen3-1.7B
scripts/30_grpo.sh                                # multi-turn GRPO on the SFT checkpoint
scripts/40_eval.sh                                # held-out dev set (seeds 900k+)
scripts/60_merge_export.sh                        # merge LoRA -> submission artifact
```

### Data (`atr/data/`)

Three trajectory sources, mixed on purpose:

| source | cost | what it teaches | what it cannot teach |
|---|---|---|---|
| oracle replay | free | format, tool choice, argument shape | deliberation — it has no reasoning |
| teacher rollout | $$ | how to *decide* on hard multi-hop tasks | on-policy phrasing for a small model |
| self-sampling | GPU | on-policy reasoning in the student's own idiom | anything it cannot already do sometimes |

Round 1 is oracle-heavy to fix format. From round 2, the marginal value is in
self-sampled rejection sampling: keep only verified-successful rollouts from the
current checkpoint and retrain.

**Filtering on `success` alone is the first mistake everyone makes.** A
successful trajectory that flailed through redundant calls teaches flailing; one
that repeated a call teaches loops. `rejection.py` rejects call-bloat, repeats,
unrequested side effects and over-represented shapes — while explicitly
*protecting* trajectories that hit a tool error and recovered, which naive
success-filtering under-samples because error paths are longer and rarer.

### RL (`atr/train/grpo.py`)

One **sample = one episode** with real tool execution interleaved; reward only
exists once the verifier runs. One **group = G episodes on the same task seed**;
advantage = group-relative reward broadcast to every assistant token.

Two knobs worth understanding before anything else:

- **`w_final_correct` (partial credit).** On a 4-hop task, a group where all G
  rollouts score zero produces zero gradient — the classic dead group. Partial
  credit for a correct answer on a failed task keeps hard tasks in the curriculum
  instead of silently dropping them.
- **`frac_dead_groups` in the logs.** Above ~0.5, the task mix is too hard for
  the current policy. Re-weight the curriculum toward difficulty 2–3 rather than
  raising the learning rate — the latter is how you get a policy that collapses
  onto one tool.

There is deliberately **no reward for using a tool**. Reward tool use and you get
tool use on the `no_tool` tasks — the exact behaviour we are trying to remove.

---

## Why end-to-end score falls off so fast

`python -m atr.cli ablate` degrades a *perfect* agent one axis at a time on the
dev set:

| per-step degradation | task success |
|---|---|
| perfect agent | 100.0% |
| argument errors @ 10% | 84.5% |
| wrong tool @ 10% | 77.3% |
| gives up early @ 20% | 65.5% |

Per-step accuracy compounds. 85% per step over four dependent steps is 52%
end-to-end. On a multi-hop set a 5-point gain in *argument accuracy* is worth more
than anywhere else you could spend the same effort — the sub-metrics are the
actual currency, not the headline.

---

## Suggested experiment order

1. **Baseline.** `40_eval.sh` on base Qwen3-1.7B and 4B. Get the real numbers
   before assuming where the loss is.
2. **Format only.** Oracle-replay SFT, ~2k examples. `format_strict` should reach
   ~99%. If not, the bug is in the export canonicalisation, not the model.
3. **Necessity.** Add `no_tool` + `unanswerable` to ~36% of the mix. The largest
   single-batch win available, and nearly free.
4. **Deliberation.** Teacher rollouts on `musique_3hop` / `musique_4hop` only.
   Watch `by_difficulty` d3/d4, not the overall number.
5. **On-policy.** Self-sampled rejection sampling, 4–8 samples per task, best-of-N.
6. **GRPO.** Only now, and only from the SFT checkpoint.
7. **Scale up.** Re-run the winning recipe on Qwen3-4B. Expect the ordering of
   decisions to transfer; expect the absolute numbers not to.

Track `by_difficulty` and `failure_modes` every run. The overall number moves
slowly and tells you nothing about what to do next.

---

## Plugging in the official tools

Edit `atr/tools/adapter.py` — one function, one branch. Build a `ToolRegistry`
whose `ToolSpec.fn(world, **kwargs)` forwards to the organizers' executor. Two
invariants the adapter must preserve, because metrics depend on them:

1. Model-caused failures come back as `{"error": ..., "message": ...}` dicts,
   never as raised exceptions. Recovery is a trained skill; a crash is not.
2. The registry appends to `world.call_log`. If their environment has its own
   session object, wrap it in a shim exposing `.call_log` and `.sent_messages`.

Everything else — loop, verifiers, data pipeline, SFT, GRPO — addresses tools only
through the registry and needs no changes. `world.py` and `builtin.py` are the
only two files that get replaced.

---

## Known limitations

- Numeric answer matching accepts *any* number in the reply within tolerance.
  `match_answer(strict=True)` refuses shotgun answers and untagged replies, but
  tighten further if the official scorer is stricter.
- The judge scores by **token-F1**; this repo scores by exact / normalised match.
  Treat the two as correlated, not identical, until calibrated on the benchmark's
  own examples.
- `sync_vllm_weights` touches vLLM internals and is version-fragile. The HF
  rollout path is slower and always correct; use it until the GRPO curve is
  trustworthy.
- Re-tokenising a sampled trajectory for GRPO logprobs can differ from the exact
  sampled token ids at rare BPE boundaries. Standard practice, small effect, but a
  real approximation.
- The stand-in tool is *plausible*, not the real one. Treat every absolute number
  from this repo as a relative measurement between checkpoints.

---

## License

This project is provided for research and educational use. See each file header
for design rationale.
