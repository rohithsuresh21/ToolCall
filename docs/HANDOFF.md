# HANDOFF — synthesised reasoning + real-MuSiQue chains

Branch: `v3-canonical-queries`. Written 2026-09-11. Read this cold; it assumes only
`CLAUDE.md`.

## The diagnosis this work came from

A failed 4-hop judge trajectory. Question: *"Where does the second largest city in
the state where Yuma's Library District is located hold NASCAR races?"* The model's
three queries were:

    "Yuma Arizona Library District"
    "Yuma County Arizona"
    "Yuma County Arizona largest city"

Three minor rephrasings orbiting the prompt's own entity, all returning the
**identical three documents**. The answer — Tucson, described in the corpus as
"second largest in the state after Phoenix" — was in the **top-3 of call 1** and was
never used.

Every step carried `"thinking": ""`. The model emits tool calls with no reasoning
between them, so nothing carries a discovered entity out of one tool result and
into the next query. That is a **carry failure, not a retrieval failure**, and it is
exactly what our SFT data teaches: oracle trajectories are bare `<tool_call>` blocks.
The reference plan is correct *because the generator already knew the chain*, so the
plan never has to show the step where the chain is discovered.

The fix: synthesise the missing step. We know the chain at generation time, so before
each call emit a `<think>` block naming what the previous result revealed and what the
next hop needs, varied across a few templates so it is not memorised verbatim.

## Judge scores for comparison (54 real MuSiQue rows, `answer_f1`)

| checkpoint | final_f1 | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| base 1.7B | 5.1% | | | |
| SFT 1.7B | 30.2% | | | |
| GRPO 1.7B, 25 steps | 29.7% | | | |
| SFT 4B | **44.6%** | 75.4 | 37.4 | 21.1 |

The 4B row is the one to beat. The hop gradient (75 → 37 → 21) is the shape this
work targets.

## Four questions answered before implementing (all measured, not assumed)

Measured with the real `Qwen/Qwen3-1.7B` tokenizer on 120 records sampled from the
committed `data/sft.jsonl`.

1. **Does `chatml.py` / `assistant_spans` handle `<think>` + a tool call in one
   assistant turn, and does the loss mask cover both?** Yes to both. Both tags live
   inside a single assistant message's `content`, so `render()` emits one
   `<|im_start|>assistant\n…<|im_end|>\n` block and `assistant_spans` returns **one
   span covering the whole thing**. Decoded span, verbatim:

   ```
   '<think>The result names Arizona. I need the second largest city in Arizona, so I
   search for that next.</think>\n<tool_call>{"name": "search", "arguments":
   {"query": "Cobalt Energy founder"}}</tool_call><|im_end|>\n'
   ```

   Prefix-stability violations: **0 / 599 spans**, with and without `<think>`. The
   `min(e, len(full_ids))` clamp never fires.

2. **Does the parser accept that shape?** Yes. `parse_turn(..., allow_fallbacks=False)`
   returns `thinking` populated, one tool call, `repairs=[]`, `errors=[]`. Same for a
   `<think>` + `<final_answer>` turn.

3. **Tokens/example.** Supervised fraction rises **7.60% → 14.97%**, i.e. 105 → 225
   supervised tokens/example, total 1383 → 1503 tokens/example. Roughly 2.1x the
   supervised signal for a 9% longer sequence. (7.60% reproduces the ~7.7% you quoted.)

4. **Does `format_strict` still pass?** Yes — `strict_format` is `not errors and all
   calls strict`, and neither is affected by a `<think>` block. Confirmed True on both
   turn shapes.

## What is DONE and committed

- **`atr/data/reasoning.py`** (new). Synthesises the `<think>` text for both sources.
  Templates chosen by `sha1(task_id|step)` — deterministic, so a rebuild is
  byte-identical; three renderings per position. Two invariants documented in the
  module docstring: a block may name only what the episode has already seen, and no
  template opens a sentence with an entity (see the `prose=True` note below).
- **`atr/data/real_chains.py`** (new). Resolves MuSiQue's `#N` placeholders into
  executable, leak-free chains. Gates: substitution occurs in the evidence, entity-like,
  not-the-subject (`subject_rule`, `subset` default / `overlap` available), unicode
  folding, every hop retrieves new support, gold in the final hop and nowhere earlier,
  hop-1 query writable from the question. Includes `verify_index()`, which asserts the
  fast BM25 ranks identically to `builtin._search` (**160/160** on the current pool).
- **`atr/tasks/schema.py`**: added `psychic_caps(query, prompt, prose=False)` +
  `_NON_ENTITY_CAPS`. ONE definition shared by the build-time gate and the auditor.
- **`atr/data/build_sft.py`**: `_rationale` now calls `reasoning.think_for_call`
  instead of returning a fixed sentence; the answering turn gets `think_for_answer`.
  Indexed by **call**, not by step. Still gated on `ExportConfig.oracle_rationale`.
- **`tests/audit_sft.py`**: new **axis 5, psychic `<think>`** — a block naming a proper
  noun absent from the prompt and from every *strictly earlier* tool response. Folded
  into the CLEAN/DEFECTS verdict. Axis 1 now uses the shared `psychic_caps`.
- **`scripts/make_musique_chain_tasks.py`** (new) → writes `data/musique_chain_tasks.jsonl`.
- **`scripts/11_build_combined.py`** (new). Builds the combined set. The two sources are
  filtered and balanced **separately** and only then concatenated at an explicit ratio,
  because `target_mix` balances by task *type* and both sources use the same three type
  names — pooling them would let whichever source is more plentiful silently fill each
  hop family.
- **`data/musique_chain_tasks.jsonl`** (new, committed). 2009 rows,
  sha256 `d15e17de…`.

### Real-chain yield (top_k=3, defaults)

**2009 / 3975 = 50.5%**, hop mix **1138 / 536 / 335**. Rejections per axis (SUPERSEDED -- the hop-2+ gate below takes this to 1935 / 48.7%, 1132 / 521 / 282):

| axis | n | % |
|---|---|---|
| substitution_not_in_evidence | 1342 | 33.8% |
| no_new_evidence | 456 | 11.5% |
| gold_leaks_early | 122 | 3.1% |
| psychic_first_query | 46 | 1.2% |

Two notes on numbers that differ from the earlier measurement:

- Reproducing the measurement exactly (subset rule, no first-query gate) gives **2055
  (51.7%), 1167/544/344** — the ~2050/51.6%/1164-545-342 you had, within 4 records.
- **The unicode-folding fix changes 0 of 7524 substitutions on this corpus.** I kept it
  (it is one `normalize` call and the failure it prevents is silent) but it is insurance
  against a name distribution, not a fix for one. The thing I *think* you meant by
  "shares a content token with hop N's own query" measures **8.00% of all substitutions
  / 2.41% of substitutions in the leak-free set** — neither is 4.5%. Rejecting on it
  (`subject_rule="overlap"`) costs only 21 records, but it also rejects correct answers
  that legitimately reuse a question word (MuSiQue nests place names: "Yuma" → "Yuma
  County"), which is why `subset` is the default. **Please sanity-check that I read
  your 4.5% correctly** — if you meant something else, `ChainConfig.subject_rule` is
  where it goes.

### One real finding worth knowing

The `psychic_caps` "function words are not entities" fix is load-bearing. Counting a
real sub-question's leading `What` / `Where` / `Which` as an invented entity rejected
**404 of 2055** otherwise-clean chains — 20% of the yield — for a word that names
nothing. This is the same trap `_oracle_entity` fell into by anchoring at `^`.

## What was LEFT, and what happened to it (2026-09-11, second session)

All five items below are DONE and committed. CLAUDE.md now carries the durable version
of this; what follows is the diff-level record.

### 1. Hop-2+ psychic queries — FIXED

`ChainConfig.require_writable_later_queries` (default on) checks every hop 2..L's
resolved query with `psychic_caps` against the question PLUS the norm_text prose of
every strictly earlier hit, with its own rejection key `psychic_query_hopN`. Hop 1's
gate is untouched and still fires at the end of `_resolve`, so a row failing both is
counted under the later-hop key (that is the check that fires first).

Yield **2009 → 1935 of 3975 (50.5% → 48.7%)**, hop mix **1138/536/335 → 1132/521/282**.
The gate costs 74 rows and **53 of them are 4-hop** — the thinnest family pays most,
because a longer chain has more hops that can assume a name. New rejection table:

| axis | n | % |
|---|---|---|
| substitution_not_in_evidence | 1310 | 33.0% |
| no_new_evidence | 451 | 11.3% |
| gold_leaks_early | 119 | 3.0% |
| psychic_query_hop2+ | 115 | 2.9% |
| psychic_first_query | 45 | 1.1% |

`data/musique_chain_tasks.jsonl` rebuilt: 1935 rows, sha256 `ae7f5348…`.

All three smoke-build offenders are now rejected, and all three come back when the
flag is off — `tests/test_real_chains.py::test_later_hop_gate_is_the_thing_that_rejects`
pins that both ways, which is what distinguishes a working gate from a row that was
already failing something else. Non-obvious: the SAME hop-2 query is psychic in one
row and clean in another, because each real row owns its own 20 passages, so hop 1
returns different prose.

### 2–3. The two datasets — BUILT, both CLEAN

Full builds, `--n 6000`, not the smoke.

| | records | 2-hop | 3-hop | 4-hop | audit |
|---|---|---|---|---|---|
| `data/sft_r0.jsonl` | 2280 | 912 | 684 | 684 | CLEAN |
| `data/sft_r35.jsonl` | 3508 | 1630 | 1010 | 868 | CLEAN |

`r35` = the identical 2280 synthetic + 1228 real (35.0%, asked 35.0%). Real slice
un-rebalanced as recommended: **718 / 326 / 184**. The 2280 synthetic records are
**byte-identical between the two files** — at 0.35 the real side is the binding
constraint, so the synthetic slice is taken whole rather than sub-sampled. So the
ablation isolates exactly one variable, and `data/sft.jsonl` (the untouched v3 set,
no `<think>`) is a third arm isolating the reasoning.

One number worth knowing before you pick the mix: 0.35 spends only **184 of the 282**
real 4-hop rows. Taking all of them means `--real-frac 0.46`, which is the entire real
pool.

Both sets pass `scripts/lib_data_gate.sh::require_clean_dataset`, so
`DATA=data/sft_r35.jsonl bash scripts/50_sft_4b.sh` runs without an override.

### 4. Sequence lengths — nothing is dropped

Measured with the real `Qwen/Qwen3-4B` tokenizer through `assistant_spans`, i.e. the
exact path `atr/train/sft.build_dataset` takes, reproducing its drop rule
(`len(ids) > max_len or not spans`).

| set | mean | max | p90 | p99 | supervised | > 4096 |
|---|---|---|---|---|---|---|
| `data/sft.jsonl` (no reasoning) | 1360 | 1723 | 1653 | 1691 | 103 (7.59%) | **0** |
| `data/sft_r0.jsonl` | 1476 | 1872 | 1793 | 1837 | 220 (14.87%) | **0** |
| `data/sft_r35.jsonl` | 1573 | 3707 | 1937 | 2929 | 197 (12.50%) | **0** |

Nothing is dropped by `--max-len 4096` and nothing exceeds it; the longest record in
either new set is 3707 tokens, and that tail is entirely real 4-hop (real mean 1754 /
max 3707 vs synthetic 1476 / 1872) — MuSiQue's 20 candidate passages are longer than a
synthetic world's. The `r35` supervised FRACTION is lower than `r0`'s (12.50% vs
14.87%) purely because real records are longer, not because they are less supervised.

### 5. Test, docs, gate — DONE

`tests/test_real_chains.py` (new, 13 checks, house style): index fidelity 25/25 against
the shipped BM25; one pinned rejection per axis plus a pinned acceptance; the three
smoke regressions pinned both ways; every query in the committed pool re-checked for
writability; pool plan shape (`len(oracle_plan) == len(route)`, no terminal read — the
synthetic `+1` rule must never be pointed here); `tidy_query` never changes a BM25 token
over all 11,500 source queries, with a non-vacuity check; determinism over a 300-row
slice; and the reasoning invariants — never names `chain[k+1]`, and no template opens a
sentence with a bare entity.

**That last one failed when first written, and the template was wrong, not the test.**
`_SYN_FIRST[1]` was `"Starting point: {head}. …"` — `psychic_caps(prose=True)` treats a
colon as a sentence boundary and skips what follows, so a bare name parked there is
invisible to the auditor. It never produced a defect (a synthetic head entity is always
named by the prompt), but the invariant is precisely what the auditor's blind spot is
traded against. Changed to `"My starting point is {head}. …"`; both datasets were
rebuilt after the change, which is why the numbers above are from the second build.
`{goal}` remains the documented exemption — it expands to the hop's own query, which
`real_chains` has already gated.

Also: `audit_sft.py`'s docstring now documents five axes (it said "three"), the data
gate's refusal message lists the fifth, and `schema.psychic_caps` pointed at a
`tests/test_reasoning.py` that was never written — it points at `test_real_chains.py`.

CPU gate green: `test_pipeline`, `test_parser`, `test_fix2`, `test_answer_f1`,
`test_shortcut_filter`, `test_naturalize`, `test_grpo_resume`, `test_source_split`,
`test_lora_rank`, `test_real_chains`, `test_curriculum_feedback`, `test_planb` all exit
0, and `eval --dev --backend oracle` is 100% at every difficulty.

## The 8.00% question, answered

**The reading I used:** a chosen `#N` substitution *shares at least one content token
with the query of the hop it was drawn from*, where "content token" is
`real_chains.content_tokens` — unicode-folded, function words dropped, tokens of 1–2
chars dropped. That is the `subject_rule="overlap"` predicate exactly. The `subset`
default is the weaker sibling: reject only when EVERY token of the candidate is already
in that query.

**The 8.00% is not reproducible and I no longer stand behind it.** Re-measured on the
current tree by recovering every chosen substitution from the resolved queries:
**114 / 4230 = 2.70%** over all attempted rows, **76 / 3020 = 2.52%** over kept rows.
The second is the "2.41% in the leak-free set" from the first session, moved by the new
gate. The first is not 8.00%, and the gap is a denominator: 7525 is the count of `#N`
placeholders in all 3975 raw plans, including every hop of every row that was rejected
before reaching it — substitutions that were never chosen. 4230 is the count actually
chosen. A rate whose numerator counts decisions and whose denominator counts
opportunities is not a rate. **So: ~2.5–2.7% under the reading above, and neither your
4.5% nor my 8.00% is a number this corpus produces.** If 4.5% came from a different
predicate, `ChainConfig.subject_rule` is still where it goes.

**Does the alternative change the built set materially? No — and its direction is
wrong.** Full-pool comparison, both with the new hop-2+ gate on:

| | kept | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| `subset` (default) | 1935 | 1132 | 521 | 282 |
| `overlap` | 1911 | 1120 | 511 | 280 |

32 rows only in `subset`, 8 only in `overlap`, 1903 shared — and **46 of the shared
rows resolve to a DIFFERENT name**, which is the part that matters more than the count.
Traced live:

    2hop__128478_11424
      hop 1: "What city is WAYV located?"          -> Atlantic City
      subset  hop 2: "How many households were there in Atlantic City during the 2010 … Census?"
      overlap hop 2: "How many households were there in New Jersey during the 2010 … Census?"

`overlap` refuses "Atlantic City" because `city` is shared with hop 1's query, and
substitutes "New Jersey" instead — a WRONG resolution that still clears the occurs gate,
because New Jersey is named in the same passage. MuSiQue nests place names constantly,
so this is the common case, not a corner.

The unicode folding, re-checked with the function-word filter held constant, changes the
overlap verdict on **0 of 4230** chosen substitutions — the first session's "0 of 7524"
reproduces. It stays in as insurance against a name distribution, not as a fix for one.
(A first pass at this comparison said 15; that probe dropped short tokens but not
function words, so it was measuring `the`, not diacritics.)

## What is actually left

- **Run the ablation.** Three arms exist and are one flag apart: `data/sft.jsonl`
  (no reasoning), `data/sft_r0.jsonl` (reasoning, synthetic only), `data/sft_r35.jsonl`
  (+35% real). Read `final_f1` PER HOP on the 54-row judge probe, not the headline — the
  4B's 75.4 / 37.4 / 21.1 gradient is the thing under test, and 184 real 4-hop
  trajectories may simply be too few to move the last number. That is a measurement, not
  a mixing problem, and no `--real-frac` fixes it.
- **`--real-frac 0.46`** is the point at which the real pool is exhausted, if the 4-hop
  arm looks starved.
- The two run-tooling issues in CLAUDE.md's "Open issues" section are untouched: the
  committed SSH password still needs ROTATING (deleting the lines is not sufficient),
  and `60_pipeline_tomorrow.sh` still passes `--eval-every 50` alongside `--save-every 5`.
