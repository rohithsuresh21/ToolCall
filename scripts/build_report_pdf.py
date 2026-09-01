# Builds a simple, professional summary PDF of the work done on the ATR
# (agentic-tool-reasoning) pipeline: shortcut filter, train/dev route holdout,
# LLM passage naturalization, and verification.
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)

OUT = "ATR_Work_Summary.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=6,
                    textColor=colors.HexColor("#1a3555"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10,
                    spaceAfter=4, textColor=colors.HexColor("#2a4d73"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.5, leading=13)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=6, spaceAfter=2)
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=8.5,
                      backColor=colors.HexColor("#f2f2f2"))
META = ParagraphStyle("META", parent=BODY, fontSize=8, textColor=colors.gray)

def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=10, value="\u2022") for t in items],
        bulletType="bullet", start="\u2022", leftIndent=10, spaceAfter=3)

story = []

# ---- Title block ----
title = Paragraph("ATR Pipeline &mdash; Work Summary", H1)
story.append(title)
story.append(Paragraph("Single-tool (BM25 MuSiQue) agentic tool-use training contract &mdash; "
                       "data-quality hardening, route holdout, and passage naturalization",
                       ParagraphStyle("sub", parent=BODY, fontSize=10, textColor=colors.gray,
                                      spaceAfter=2)))
story.append(Paragraph("Prepared by the engineering agent &middot; verified locally (Windows / CPU)",
                       META))
story.append(Spacer(1, 6))

# ---- 1. Objective ----
story.append(Paragraph("1. Objective", H2))
story.append(Paragraph(
    "Build and validate a data pipeline for training an agent to solve multi-hop "
    "questions by chaining BM25 <i>search</i> tool calls. The current SFT model sits at a "
    "<b>20% TASK SUCCESS</b> baseline against the internal dev set; the immediate target is "
    "to lift this before GRPO. The work here focuses on <b>training-data quality</b>: removing "
    "questions that can be answered without chaining, holding out structural shapes for "
    "honest evaluation, and making synthetic passages look more natural without changing the "
    "underlying facts.", BODY))
story.append(Spacer(1, 2))

# ---- 2. Objective of the three-part change ----
story.append(Paragraph("2. What was changed (three parts)", H2))

story.append(Paragraph("Part 1 &mdash; Shortcut (disconnection) filter", H3 := ParagraphStyle(
    "H3local", parent=H2, fontSize=11, spaceBefore=6, spaceAfter=2)))
story.append(Paragraph(
    "A multi-hop question is a <i>disconnected shortcut</i> when a single, un-chained BM25 "
    "search can already surface the gold answer &mdash; the exact lazy move a model makes "
    "instead of reasoning. We now detect this with the <b>same <i>search</i> tool the agent "
    "uses</b>: fire one search with the full question text and check whether the returned "
    "passages already contain the gold value. Such questions are rejected at generation time. "
    "The filter is off by default for fast/offline work and is observable via a "
    "<font face='Courier'>_SHORTCUT_STATS</font> counter.", BODY))
story.append(Paragraph(
    "Measured on a 500-task train sample: <b>52 of 359</b> candidate multi-hop chains "
    "(&asymp;<b>14.5%</b>) were flagged as shortcut-solvable and rejected &mdash; a real, "
    "non-trivial leak rather than dead code.", BODY))
story.append(Paragraph("How it works (function flow)", H3))
story.append(bullets([
    "<b>gen_musique()</b> builds the final <font face='Courier'>prompt</font> for each candidate "
    "chain; if <font face='Courier'>filter_shortcuts=True</font> it increments "
    "<font face='Courier'>_SHORTCUT_STATS[\"checked\"]</font> then calls "
    "<font face='Courier'>_is_shortcut_solvable(w, prompt, gold)</font>.",
    "<b>_is_shortcut_solvable()</b>: <font face='Courier'>get_registry(\"builtin\")</font> &rarr; "
    "<font face='Courier'>reg.call(w, \"search\", {\"query\": prompt, \"top_k\": 3})</font> &mdash; "
    "fires the <b>real BM25 search tool</b> with the full question text, returning top-3 passages. "
    "If 0 results &rarr; not a shortcut. It computes <font face='Courier'>want = "
    "_norm_find(gold)</font> (the normalized expected answer).",
    "<b>Substring check</b>: if the normalized answer appears in any of the 3 returned passages' "
    "normalized text &rarr; returns True (shortcut).",
    "<b>Back in gen_musique</b>: on True it increments "
    "<font face='Courier'>_SHORTCUT_STATS[\"rejected\"]</font> and <font face='Courier'>continue</font> "
    "s (skips this chain, tries the next route). If no chain survives it returns None and the "
    "caller falls back to a no_tool task.",
]))

story.append(Paragraph("Part 2 &mdash; Train / dev route-shape holdout", H3))
story.append(Paragraph(
    "A <b>shape</b> is the <i>fixed skeleton</i>: a comma-separated sequence of relations that "
    "defines how many hops and which general relations to chain (e.g. "
    "org_city &rarr; city_country means &ldquo;find the org's city, then that city's "
    "country&rdquo;). It is a <b>label only</b> &mdash; it does not fix any specific content. "
    "Underneath a shape, the concrete entities are free to vary: any person, company, city or "
    "country from the world's pools can fill each slot, so a single shape generates many "
    "distinct trajectories with different questions and answers. The <i>only</i> constraint is "
    "that each relation fixes the <b>kind</b> of entity per slot (a city slot takes any city, "
    "a person slot takes any person) &mdash; the <b>instances</b> are what vary.", BODY))
story.append(Paragraph(
    "Example (real questions, same 3-hop dev shape work_person &rarr; person_org &rarr; "
    "org_city): &ldquo;who is the author whose company is headquartered in Meridian City?&rdquo;, "
    "&ldquo;which author writes for the company based in Nyrmont?&rdquo;, &ldquo;name the author "
    "at the firm located in Lacerta&rdquo;. Same skeleton, different people/companies/cities, "
    "different answers each time.", BODY))
story.append(Spacer(1, 2))

route_rows = [
    ["Hops", "Train shapes", "Dev-only (held-out) shapes"],
    ["2-hop", "3 shapes", "1 shape: org_city &rarr; city_country"],
    ["3-hop", "3 shapes", "1 shape: work_person &rarr; person_org &rarr; org_city"],
    ["4-hop", "2 shapes", "1 shape: work_person &rarr; person_org &rarr; org_city &rarr; city_country"],
]
rt = Table(route_rows, colWidths=[22*mm, 60*mm, 92*mm])
rt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4d73")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(rt)
story.append(Spacer(1, 3))
story.append(Paragraph(
    "Totals: <b>train = 8 shapes</b> (3 + 3 + 2), <b>dev = 3 held-out shapes</b> (1 + 1 + 1). "
    "Dev evaluation only ever sees held-out shapes, so success on dev is a genuine "
    "generalization signal. A test in <font face='Courier'>test_shortcut_filter.py</font> proves "
    "the train pool never emits a dev-only shape and the dev set uses exactly the held-out "
    "shapes.", BODY))
story.append(Paragraph("How it works (function flow)", H3))
story.append(bullets([
    "Two module-level dicts in <font face='Courier'>generator.py</font>: "
    "<font face='Courier'>_ROUTES_TRAIN</font> and <font face='Courier'>_ROUTES_DEV_ONLY</font>, "
    "each keyed by hop count <font face='Courier'>{2: [...], 3: [...], 4: [...]}</font>. Each "
    "entry is a list of step-key sequences; each sequence is one shape.",
    "<b>Selection inside gen_musique()</b>: "
    "<font face='Courier'>pool = (_ROUTES_TRAIN if route_pool == \"train\" else "
    "_ROUTES_DEV_ONLY)[hops]</font>, then <font face='Courier'>rng.shuffle(routes)</font> and "
    "it iterates the shapes in random order, returning the first chain that resolves and passes "
    "the shortcut filter.",
    "<b>Train path</b>: <font face='Courier'>generate()</font> &rarr; "
    "<font face='Courier'>gen_*_hop(..., route_pool=\"train\")</font> (default) &rarr; only train "
    "shapes.",
    "<b>Dev path</b>: <font face='Courier'>dev_set()</font> calls "
    "<font face='Courier'>fn(..., route_pool=\"dev\")</font> &rarr; only dev-only shapes.",
    "<b>Verification</b>: the test rebuilds the shape a task used from its "
    "<font face='Courier'>oracle_plan</font> (mapping each plan query's trailing keyword back to "
    "a relation via <font face='Courier'>_kw_to_step()</font>) and asserts dev = the held-out "
    "shapes and train never matches a dev shape.",
]))
story.append(Spacer(1, 2))

story.append(Paragraph("Dataset composition (what the whole dataset contains)", H3))
story.append(Paragraph(
    "The dataset is <b>not</b> only country/city chains. It is a controlled mix of five families "
    "(verified: musique_2hop, musique_3hop, musique_4hop, no_tool, unanswerable):", BODY))
story.append(bullets([
    "<b>musique_2hop / 3hop / 4hop</b> &mdash; the chained questions across 8 train + 3 dev "
    "shapes (person / company / city / country + attributes).",
    "<b>no_tool</b> &mdash; a question answerable from one passage alone; correct behavior is "
    "zero tool calls. A 304-item bank covering gold kinds all_of, any_of, numeric, text.",
    "<b>unanswerable</b> &mdash; the fact genuinely does not exist in the corpus (e.g. fake "
    "products like &ldquo;Zephyr Vale&rdquo;); the model must abstain, not hallucinate.",
]))
story.append(Spacer(1, 2))

story.append(Paragraph("Part 3 &mdash; LLM passage naturalization (facts untouched)", H3))
story.append(Paragraph(
    "Synthetic passages currently use fixed per-kind templates (e.g. every country passage has "
    "the same three sentences). Part A rewrites them into varied, Wikipedia-like prose using an "
    "LLM &mdash; <b>offline only</b>, never inside training or evaluation. Two hard invariants "
    "are enforced by post-check: (1) every fact the original template surfaced must still appear "
    "in the new text (otherwise retry, fall back to the original after a cap); (2) naturalization "
    "only ever replaces <font face='Courier'>doc.text</font> &mdash; it never touches the entity "
    "<font face='Courier'>attrs</font> dict, so gold answers and every verifier are "
    "byte-identical. Usage: <font face='Courier'>scripts/70_naturalize_passages.sh</font> mints "
    "a seed-keyed cache, then <font face='Courier'>build_world(seed, text_loader=...)</font> "
    "loads it opt-in.", BODY))
story.append(Paragraph("How it works (function flow)", H3))
story.append(bullets([
    "<b>naturalize_passages(world, llm_client)</b> &mdash; entry point. Loops every "
    "<font face='Courier'>world.documents</font>, calls <font face='Courier'>naturalize_passage</font> "
    "per doc, writes back only <font face='Courier'>d[\"text\"]</font>, returns a stats dict "
    "(docs / naturalized / fell_back).",
    "<b>naturalize_passage(passage, llm_client)</b> &mdash; retry loop (max 3): "
    "<font face='Courier'>_build_prompt(passage)</font> builds the rewrite instruction (with "
    "FACTS json + the current templated passage); <font face='Courier'>_call_llm(client, prompt)</font> "
    "gets the rewrite (client is a callable or an object with "
    "<font face='Courier'>.complete(prompt)</font>); <font face='Courier'>_facts_present()</font> "
    "is the safety post-check. If all facts survived, accept and return.",
    "<b>Fact-presence check</b>: <font face='Courier'>_surfaced_facts()</font> decides which facts "
    "must survive (those that already appear in the original text, via "
    "<font face='Courier'>_format_for_check()</font>, plus the title); "
    "<font face='Courier'>_text_presence()</font> does a formatting-robust substring match "
    "(strips non-alphanumerics so 1,097,000 == 1097000).",
    "<b>Fallback</b>: if all 3 retries fail, ship the original templated text and mark "
    "<font face='Courier'>naturalized=False</font> &mdash; never ships broken prose.",
    "<b>Scoring isolation</b>: <font face='Courier'>out = dict(passage)</font> then only "
    "<font face='Courier'>out[\"text\"]</font> changes; the <font face='Courier'>facts</font>/"
    "<font face='Courier'>attrs</font> dicts (the sole source of gold answers) are untouched.",
    "<b>Opt-in wiring</b>: <font face='Courier'>build_world(seed, text_loader=None)</font> &mdash; "
    "when a loader is given it calls <font face='Courier'>text_loader(seed, d[\"doc_id\"])</font> "
    "and, if a string returns, sets <font face='Courier'>d[\"text\"]</font> and marks "
    "<font face='Courier'>naturalized=True</font>. The offline script writes "
    "<font face='Courier'>{\"seed:doc_id\": text}</font> to a cache; "
    "<font face='Courier'>load_naturalized_loader(path)</font> reads it back into a loader.",
]))
story.append(Spacer(1, 2))
story.append(Paragraph("Seed &amp; cache &mdash; the key idea", H3))
story.append(bullets([
    "<b>Seed</b> = an integer that makes world generation deterministic: "
    "<font face='Courier'>build_world(seed)</font> always yields the same entities and passages. "
    "Training generation uses seeds 0&ndash;500k; dev uses 900k+.",
    "<b>Range</b> = which worlds need naturalized passages. The cache is keyed by "
    "<font face='Courier'>seed:doc_id</font>, and <font face='Courier'>build_world</font> only "
    "substitutes naturalized prose for seeds present in the cache. So a seed range is required "
    "to tell the script which worlds to preprocess; a seed not in the cache simply gets templated "
    "prose (a clean per-seed fallback).",
    "<b>Full range (one-time preprocessing)</b>: for each seed, build the world, rewrite every "
    "passage via the local LLM, run the fact-preservation post-check, and store accepted rewrites "
    "in <font face='Courier'>artifacts/naturalized_passages.json</font>.",
    "<b>What the cache does</b>: it is exactly what "
    "<font face='Courier'>load_naturalized_loader(CACHE)</font> reads so training/eval can use "
    "naturalized prose with identical facts/gold. No cache = templated prose; full cache = the "
    "hardened data used to train toward the target. It runs offline once, then just gets loaded.",
]))
story.append(Paragraph("In simple words", H3))
story.append(Paragraph(
    "<b>Deterministic</b> = the same input always gives the same output &mdash; no coin-flips "
    "between runs. Put in the same seed and you always get the exact same entities, names, "
    "numbers and passages; a different seed gives a different but equally repeatable world. "
    "Think of the seed like a <b>recipe-card number</b>: seed 42 always bakes the same cake, "
    "seed 99 always bakes a different, fixed one.", BODY))
story.append(Paragraph(
    "A <b>seed fixes the randomness</b>; it does <b>not</b> measure &ldquo;how much&rdquo;. The "
    "generator has a fixed pseudo-random sequence, and the seed simply picks <i>which</i> starting "
    "point &mdash; so a seed selects <i>which</i> random world you get, not an amount of "
    "randomness. Same seed = same world (reproducible); one seed apart = a different, "
    "independent-looking world. The variety you cover is set by how many seeds you include, not "
    "by any single seed value: more seeds = more distinct worlds, because each is a different "
    "draw. A seed is just an ID that locks in one specific, reproducible world. A "
    "<b>range</b> tells the script <i>which</i> of those locked-in worlds to naturalize, because "
    "the cache stores one entry per seed:doc_id &mdash; a seed not in the cache simply keeps its "
    "original templated prose. A <b>complete cache</b> means every world used for training or eval "
    "has varied, natural prose with its facts unchanged. All of this runs once, offline and free "
    "via local Ollama, and is then just loaded at train time.", BODY))
story.append(Spacer(1, 2))

# ---- 3. Verification ----
story.append(Paragraph("3. Verification (all green)", H2))
story.append(Paragraph("Oracle evaluation against the dev set (6/type):", BODY))
story.append(Spacer(1, 2))
eval_rows = [
    ["Metric", "Result"],
    ["TASK SUCCESS", "100.0%"],
    ["final answer correct", "100.0%"],
    ["tool selection / args match oracle", "100.0% / 100.0%"],
    ["tool-necessity decision", "100.0%"],
    ["format strict / parseable", "100.0% / 100.0%"],
]
et = Table(eval_rows, colWidths=[95*mm, 60*mm])
et.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4d73")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
]))
story.append(et)
story.append(Spacer(1, 6))
story.append(Paragraph("Regression test suites &mdash; all passing:", BODY))
story.append(Spacer(1, 2))
story.append(bullets([
    "<b>test_pipeline.py</b> &mdash; ALL PASS (generation determinism, oracle 100%, toolset "
    "single-tool, SFT export, advantage ordering).",
    "<b>test_parser.py</b> &mdash; 0 failures (19 parser cases).",
    "<b>test_fix2.py</b> &mdash; ALL PASS (verifiers, efficiency, GRPO, cleanup).",
    "<b>test_curriculum_feedback.py</b> &mdash; ALL PASS (GRPO curriculum-feedback feature).",
]))
story.append(Spacer(1, 4))

story.append(Paragraph("New test: test_shortcut_filter.py &mdash; 7/7 PASS", H3))
story.append(Spacer(1, 2))
short_rows = [
    ["Test", "Verifies"],
    ["test_detection_not_vacuous", "Filter flags at least one shortcut chain (not dead code)."],
    ["test_not_overflagging", ">= 50% of genuine chains are NOT flagged."],
    ["test_filter_excludes_rejected_shape", "A flagged chain is excluded when filtering is on."],
    ["test_filter_observable_rejection_rate", "Rejection counter non-zero with a sane rate."],
    ["test_dev_uses_dev_only_shapes", "Dev set uses exactly the held-out shapes."],
    ["test_train_never_uses_dev_only_shapes", "Train generation never emits a dev-only shape."],
    ["test_shapes_are_adjacent_and_disjoint", "Dev shapes are not duplicated in train."],
]
st = Table(short_rows, colWidths=[78*mm, 77*mm])
st.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4d73")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(st)
story.append(Spacer(1, 4))

story.append(Paragraph("New test: test_naturalize.py &mdash; 7/7 PASS", H3))
story.append(Spacer(1, 2))
nat_rows = [
    ["Test", "Verifies"],
    ["test_facts_preserved_on_success", "Good rewrite keeps every fact AND varies the text."],
    ["test_dropped_fact_is_rejected_and_falls_back", "Dropped fact rejected; falls back after retries."],
    ["test_api_never_fabricates_facts", "Only text/title touched; facts dict never mutated."],
    ["test_numfmt_matches_commas", "Number formatting robust (1097000 == 9,800,000)."],
    ["test_scoring_isolation", "attrs / gold / oracle answers identical after naturalization."],
    ["test_loader_opt_in_wiring", "build_world(text_loader=...) swaps prose only."],
    ["test_loader_none_when_missing", "Missing/empty cache -> loader is None (opt-in)."],
]
nt = Table(nat_rows, colWidths=[78*mm, 77*mm])
nt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4d73")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(nt)
story.append(Spacer(1, 4))

story.append(Paragraph("Live naturalization (free, local, verified)", H3))
story.append(Paragraph(
    "Ran the pipeline against a local <b>Ollama</b> server (qwen2.5-coder:7b) on one world "
    "(seed 0, 38 passages): <b>38 / 38 naturalized, 0 fell back</b>, and independent re-check "
    "of the cache with <font face='Courier'>_facts_present</font> showed <b>zero fact-loss</b>. "
    "Cache written to <font face='Courier'>artifacts/naturalized_passages.json</font>. No API "
    "key or credits required.", BODY))
story.append(Spacer(1, 4))

# ---- 4. Files touched ----
story.append(Paragraph("4. Primary files", H2))
story.append(bullets([
    "<font face='Courier'>atr/tasks/generator.py</font> &mdash; shortcut filter "
    "(<font face='Courier'>_is_shortcut_solvable</font>), train/dev route-pool split "
    "(<font face='Courier'>_ROUTES_TRAIN</font>/<font face='Courier'>_ROUTES_DEV_ONLY</font>), "
    "threaded <font face='Courier'>filter_shortcuts</font>/<font face='Courier'>route_pool</font>.",
    "<font face='Courier'>atr/tools/world.py</font> &mdash; <font face='Courier'>build_world(...)</font> "
    "opt-in <font face='Courier'>text_loader</font> for naturalized passages.",
    "<font face='Courier'>atr/data/naturalize.py</font> (new) &mdash; fact-preserving passage "
    "rewrite + cache loader.",
    "<font face='Courier'>atr/data/naturalize_local.py</font> (new) &mdash; free, local runner "
    "that feeds the naturalize pipeline through Ollama's OpenAI-compatible endpoint (no key).",
    "<font face='Courier'>scripts/70_naturalize_passages.sh</font> (new) &mdash; offline "
    "naturalization to a seed-keyed cache.",
    "<font face='Courier'>tests/test_shortcut_filter.py</font>, "
    "<font face='Courier'>tests/test_naturalize.py</font> (new).",
]))
story.append(Spacer(1, 4))

# ---- 5. Models used ----
story.append(Paragraph("5. Models used", H2))
story.append(bullets([
    "<b>SFT / reinforcement (the agent model):</b> <font face='Courier'>Qwen/Qwen3-1.7B</font> "
    "(&ldquo;only move to 4B once the 1.7B curve has flattened&rdquo;).",
    "<b>Part A naturalization LLM (live, free, local):</b> <font face='Courier'>qwen2.5-coder:7b</font> "
    "served by a local <b>Ollama</b> server (http://localhost:11434/v1), an OpenAI-compatible "
    "endpoint. No API key and no credits required &mdash; fully offline and on-machine. Run via "
    "<font face='Courier'>python -m atr.data.naturalize_local</font>.",
    "<b>Fallback option</b>: the same pipeline also works with any OpenAI-compatible provider "
    "(e.g. gpt-4.1-mini) if an API key is available; tests use a mock so no key is needed.",
]))

# ---- 6. Next steps ----
story.append(Paragraph("6. Next steps", H2))
story.append(bullets([
    "Reproduce the <b>20% &rarr; target</b> SFT training on these hardened data, then run GRPO.",
    "Optionally run naturalization with an LLM key to produce the live fact-preservation "
    "retry-rate numbers (tests already use a mock, no key needed).",
    "Report the held-out dev success to confirm lifting above the 20% baseline.",
]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Scope note: repository is a git repo on branch main with an unrelated pre-existing "
    "dirty file (requirements-colab.txt) deliberately left untouched. This summary reflects "
    "work verified locally on Windows/CPU.</i>", META))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=16*mm, bottomMargin=16*mm,
                        title="ATR Pipeline Work Summary")
doc.build(story)
print("wrote", OUT)
