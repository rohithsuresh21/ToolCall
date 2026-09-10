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

**2009 / 3975 = 50.5%**, hop mix **1138 / 536 / 335**. Rejections per axis:

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

## What is LEFT

### 1. Hop-2+ psychic queries in real chains — **do this first**

The smoke build audits **DEFECTS PRESENT**: 3 of 325 records (all 4-hop) trip the new
psychic-`<think>` axis. They are **genuine**, not template noise:

    4hop3__166346_… "That gives me Zermatt. Now: Where are the villages of Wengen and Zermatt located?"
    4hop3__5752_…   "Now I have United States. The next step: Federal Detention Center, United States >> country?"
    4hop3__31642_…  "The result names Lockerbie. Next: All Saints Church, Lockerbie >> located in …?"

`Wengen`, `Federal Detention Center` and `All Saints Church` appear in MuSiQue's own
sub-question but in neither the question nor any earlier retrieved passage. The build-time
gate (`ChainConfig.require_writable_first_query`) only checks **hop 1**; hops 2+ escape it.
The `<think>` text is the same string as the query, so this is a psychic *query* that the
new audit axis surfaced — the reasoning did not invent it.

**Fix:** in `real_chains._resolve`, extend the writability check to every hop — a query's
`psychic_caps` must be empty against *prompt + text of all strictly earlier hits*. The
per-hop hit texts are already in scope as `hop_hits` / `idx.blob`. Then re-measure yield
(expect a few percent below 2009) and update the docstring's MEASURED YIELD block and the
table above. Keep it a named `ChainConfig` flag with its own rejection key, per the
"one key per axis" convention.

### 2. The combined build

Not yet run at full size — only a 300-seed smoke build. After fix 1:

```bash
python scripts/make_musique_chain_tasks.py
python scripts/11_build_combined.py --n 6000 --real-frac 0.35 --no-promote
# inspect the audit, then drop --no-promote to write data/sft.jsonl
```

### 3. A test for `real_chains.py`

`tests/test_real_chains.py` does not exist yet. Match the house style (plain script,
`check(cond, label)`, `sys.exit(1)`; no pytest). It should cover:
- `verify_index()` == 100% (without it every yield number describes a re-implementation)
- a pinned known-rejection per axis, the way `test_shortcut_filter.py` pins seed 735 —
  a gate that fires on 1% is indistinguishable from dead code under a sampling test
- `tidy_query()` never changes a BM25 token
- determinism: two `build_chain_tasks` runs are byte-identical
- `reasoning.think_for_call` never names `chain[k+1]` at step k, and **no template opens
  a sentence with an entity** (the `prose=True` audit mode cannot see one there)

### 4. CLAUDE.md updates

Nothing in CLAUDE.md yet describes any of this. Add:
- reasoning synthesis + its two invariants; that `oracle_rationale` is now meaningful
  and what it costs in supervised tokens (7.60% → 14.97%)
- `real_chains.py`: the resolution strategy and each gate's measured cost
- `psychic_caps` as the single shared definition, and the function-word finding
- audit axis 5, and that the audit's headline is now five axes not four (the module
  docstring still says "the three defects" / "The three axes")
- that `data/musique_chain_tasks.jsonl` inherits judge-disjointness from
  `musique_train_tasks.jsonl` because it only rewrites plans and never resamples

## Recommended mix — and the open question

**My recommendation: `--real-frac 0.35`, with the real slice left un-rebalanced.**

Reasoning. Real prose is the gap the judge score is measuring — synthetic 2-hop sits near
100% while the real 2-hop rows sit at 46.4%, and the whole synthetic-to-real delta is what
the 4B's 75/37/21 gradient is made of. But real supply is **2009 records against ~2280
synthetic**, and it is thinnest exactly where the model is weakest: **only 335 real 4-hop**
(16.7% of the real pool) versus 684 synthetic 4-hop. At 0.35 you get roughly 1080 synthetic
+ 580 real, which spends most of the real 4-hop supply while keeping synthetic as the
volume source for the hop families where it is cheap.

The open question I could not settle for you: **do not rebalance the real slice to 40/30/30.**
Doing so caps the whole real slice at `335 / 0.30 ≈ 1116` records and throws away ~800
real 2-hop rows to buy nothing — the 4-hop count is fixed at 335 either way. Taking the real
pool as-is (57/27/17) and letting the synthetic side carry the hop balance gets you every
real 4-hop row AND more real prose. That is what `--real-hop-mix ""` (the default) does.

The real risk is the opposite one and it is not addressable by mixing: 335 4-hop real
trajectories may simply be too few to move 4-hop, and the honest test is an ablation —
build at 0.0 / 0.35 / 0.60 and read `final_f1` per hop on the judge probe. Worth one GPU
afternoon before committing to a mix.

## State of the tree

CPU gate green after the changes: `test_pipeline`, `test_parser`, `test_fix2`,
`test_answer_f1` all PASS, and the committed `data/sft.jsonl` still audits **CLEAN** under
the new five-axis auditor (it has no `<think>` blocks, so axis 5 is 0 — that is a
consistency check, not evidence the axis works; the smoke build is what exercises it).

`data/sft.jsonl` is **untouched** — still the 2280-record v3 set. Nothing here has been
promoted into the training path.
