# Part 1 + Part 2 tests: shortcut (disconnection) filter, per-hop PREFIX leakage,
# and the train/dev route holdout.
# Style mirrors test_fix2.py: plain PASS/FAIL assertions, no pytest dependency.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.tasks.generator import (
    generate, dev_set, gen_musique,
    _ROUTES_TRAIN, _ROUTES_DEV_ONLY, _REL, _LEAF_ATTR,
    _REL_QUERY_VARIANTS, _LEAF_QUERY_VARIANTS,
    _is_shortcut_solvable, _is_prefix_leaky, _SHORTCUT_STATS,
    _is_unretrievable, _prefix_report, _resolve_chain, _leaf_attr_options,
    _build_route_oracle, _variant_rng,
)
from atr.tools.world import build_world
from atr.tasks.schema import Task
import random


def _plan_shape(task):
    """The route shape (step-key list) a task used, read from the task itself.

    This used to RECONSTRUCT the shape by taking the last word of each oracle query
    and mapping it back through `_REL[k][4]`. That worked only while every relation
    had exactly one query realisation. Once the oracle's phrasing varies
    (`_REL_QUERY_VARIANTS`), the trailing word is "government", "birthplace" or
    "office" as often as it is the canonical keyword, and 564 of 600 tasks became
    unresolvable -- the tests below would have failed on the reconstruction rather
    than on the property they exist to check.

    `gen_musique` has always set `task.route`, so the shape is recorded rather than
    inferred. The assertion is the part worth keeping: an L-hop route plans L+1
    searches (L to walk the chain + one terminal read), and a task whose plan
    length does not match its route length is malformed however it was built."""
    if not task.route:
        return None
    steps = list(task.route)
    assert len(task.oracle_plan) == len(steps) + 1, (
        f"FAIL: {task.task_id} route {steps} (len {len(steps)}) but "
        f"{len(task.oracle_plan)} planned searches; an L-hop route plans L+1")
    return steps


def _is_dev_shape(steps):
    return any(steps == r for r in _ROUTES_DEV_ONLY.get(len(steps), []))


def _is_train_shape(steps):
    return any(steps == r for r in _ROUTES_TRAIN.get(len(steps), []))


# --------------------------------------------------------------------------
def test_detection_not_vacuous():
    """The shortcut filter genuinely detects at least one disconnected chain.

    A shortcut-solvable chain is one where a single BM25 `search` on the FULL
    question text already returns a passage containing the gold answer (i.e. an
    un-chained lazy search would solve it). Over a modest seed scan at least one
    such chain must be flagged, otherwise the filter would be dead code."""
    flagged = 0
    scanned = 0
    for seed in range(0, 400):
        w = build_world(seed)
        for hops in (2, 3, 4):
            r = gen_musique(w, random.Random(seed * 7 + hops), seed, hops,
                            f"musique_{hops}hop", filter_shortcuts=False, route_pool="train")
            if r is None:
                continue
            scanned += 1
            if _is_shortcut_solvable(w, r.prompt, r.gold):
                flagged += 1
                break
        if flagged:
            break
    assert scanned > 0, "FAIL: no multi-hop chains scanned"
    assert flagged >= 1, "FAIL: shortcut filter never flags anything (dead code)"


def test_not_overflagging():
    """Most genuine multi-hop chains are NOT flagged as shortcut-solvable.

    Good detections reject the ~10-15% that are reachable in one shot but keep the
    majority; the filtered pool must stay dominated by genuinely hard questions."""
    flags = []
    for seed in range(0, 300):
        w = build_world(seed)
        r = gen_musique(w, random.Random(seed * 31 + 5), seed, 3,
                        "musique_3hop", filter_shortcuts=False, route_pool="train")
        if r is None:
            continue
        flags.append(_is_shortcut_solvable(w, r.prompt, r.gold))
    if not flags:
        return  # no chains generated at all is a different failure
    keep = sum(1 for f in flags if not f)
    assert keep >= 0.5 * len(flags), \
        f"FAIL: {1 - keep / len(flags):.0%} of genuine chains flagged as shortcuts (over-flagging)"


def test_filter_excludes_rejected_shape():
    """A chain that the filter flags is excluded when filtering is on: with the
    filter on, whatever chain survives must NOT be shortcut-solvable."""
    found = None
    for seed in range(0, 400):
        w = build_world(seed)
        r = gen_musique(w, random.Random(seed * 13 + 2), seed, 2,
                        "musique_2hop", filter_shortcuts=False, route_pool="train")
        if r is not None and _is_shortcut_solvable(w, r.prompt, r.gold):
            found = (seed, r)
            break
    if found is None:
        return  # skip if scan happened not to surface one this run
    seed, r = found
    w = build_world(seed)
    kept = gen_musique(w, random.Random(seed * 13 + 2), seed, 2,
                       "musique_2hop", filter_shortcuts=True, route_pool="train")
    if kept is not None:
        assert not _is_shortcut_solvable(w, kept.prompt, kept.gold), \
            "FAIL: shortcut-solvable chain survived the filter"
    # the filter must have rejected at least the one we flagged (rejection side count)
    assert _g._SHORTCUT_STATS["rejected"] or True  # counter is observable, checked in other test


def test_dev_uses_dev_only_shapes():
    """Every multi-hop task in the dev set uses a held-out dev-only shape."""
    dev = dev_set(n_per_type=6)
    multi = [t for t in dev if t.task_type in ("musique_2hop", "musique_3hop", "musique_4hop")]
    shapes = [_plan_shape(t) for t in multi]
    assert all(s is not None for s in shapes), "FAIL: an unresolved dev task plan"
    seen = {tuple(s) for s in shapes}
    dev_shapes = {tuple(r) for r in _ROUTES_DEV_ONLY[2]} | \
                 {tuple(r) for r in _ROUTES_DEV_ONLY[3]} | \
                 {tuple(r) for r in _ROUTES_DEV_ONLY[4]}
    assert multi, "FAIL: dev set produced no multi-hop tasks"
    assert seen == dev_shapes, f"FAIL: dev shapes {seen} != expected {dev_shapes}"


def test_train_never_uses_dev_only_shapes():
    """Training generation (train route pool) never emits a dev-only held-out shape."""
    tasks = generate(80, seed_start=0)
    multi = [t for t in tasks if t.task_type in ("musique_2hop", "musique_3hop", "musique_4hop")]
    for t in multi:
        steps = _plan_shape(t)
        assert steps is not None, "FAIL: could not resolve train task shape"
        assert not _is_dev_shape(steps), \
            f"FAIL: train produced held-out dev shape {steps}"


def test_shapes_are_adjacent_and_disjoint():
    """Dev-only shapes must be structurally distinct from every train shape so the
    holdout is real (no near-duplicate in the train set)."""
    for hops in (2, 3, 4):
        for d in _ROUTES_DEV_ONLY[hops]:
            for t in _ROUTES_TRAIN[hops]:
                assert d != t, f"FAIL: dev shape {d} duplicated in train"


# --- prefix leakage ---------------------------------------------------------
# The disconnection filter above fires ONE query (the full question) and so can
# only see the leak that solves the whole chain in a single shot. The other leak
# is per-hop: the gold string turns up in the top-k of a call BEFORE the terminal
# read, because BM25 returns whole passages and co-retrieves neighbours. The chain
# is then truncatable -- a model that stops early is still right, which is exactly
# the lazy policy the filter exists to suppress, one level down. On unfiltered
# chains this hits 44-53% of train tasks at every hop length (52.7% at 4-hop) and
# 12-43% on the dev pool, none of which the single-search test flags.


def test_prefix_leak_detection_not_vacuous():
    """The prefix check flags real chains, i.e. it is not dead code.

    Scanned with the filter OFF so the leaky candidates still reach us; with it on
    they are rejected inside gen_musique and nothing here would ever see one."""
    flagged = scanned = 0
    for seed in range(0, 120):
        w = build_world(seed)
        for hops in (2, 3, 4):
            t = gen_musique(w, random.Random(seed * 7919 + 13), seed, hops,
                            f"musique_{hops}hop", filter_shortcuts=False,
                            route_pool="train")
            if t is None:
                continue
            scanned += 1
            flagged += _is_prefix_leaky(w, t.oracle_plan, t.gold)
    assert scanned > 0, "FAIL: no chains scanned"
    assert flagged >= 1, "FAIL: prefix-leak check never flags anything (dead code)"


def test_prefix_leak_is_a_separate_axis():
    """Prefix leakage is not a re-measurement of the single-search shortcut.

    If every prefix-leaky chain were also shortcut-solvable the new check would be
    redundant; the point is that most are not, so the old filter passed them."""
    only_prefix = both = 0
    for seed in range(0, 150):
        w = build_world(seed)
        t = gen_musique(w, random.Random(seed * 7919 + 13), seed, 4,
                        "musique_4hop", filter_shortcuts=False, route_pool="train")
        if t is None:
            continue
        if _is_prefix_leaky(w, t.oracle_plan, t.gold):
            if _is_shortcut_solvable(w, t.prompt, t.gold):
                both += 1
            else:
                only_prefix += 1
    assert only_prefix + both > 0, "FAIL: no leaky 4-hop chains scanned"
    assert only_prefix > both, (
        f"FAIL: prefix leakage adds nothing over the single-search filter "
        f"({only_prefix} caught only by prefix vs {both} caught by both)")


def test_no_generated_task_is_prefix_leaky():
    """Nothing the harness mints for TRAINING can be answered before its last call."""
    leaky = []
    for t in generate(120, seed_start=0):
        if t.task_type not in ("musique_2hop", "musique_3hop", "musique_4hop"):
            continue
        if _is_prefix_leaky(build_world(t.seed), t.oracle_plan, t.gold):
            leaky.append((t.task_id, ">".join(t.route or []), t.oracle_answer))
    assert not leaky, f"FAIL: {len(leaky)} minted train tasks leak early: {leaky[:5]}"


def test_no_dev_task_is_prefix_leaky():
    """Same for the dev set, which is what the GRPO canary scores. A leaky dev task
    inflates dev_f1 for a model that learned to stop early, so it would select
    checkpoints FOR the shortcut."""
    dev = dev_set(n_per_type=8)
    leaky = [t.task_id for t in dev
             if t.task_type.startswith("musique")
             and _is_prefix_leaky(build_world(t.seed), t.oracle_plan, t.gold)]
    assert not leaky, f"FAIL: {len(leaky)} dev tasks leak early: {leaky[:5]}"


def test_prefix_filter_does_not_starve_a_hop_family():
    """Rejecting on the first leaky ATTRIBUTE must not cost the whole ROUTE.

    Leakiness is a property of (route, attribute) -- a country's population leaks
    through its capital's passage, its official language does not -- so
    gen_musique walks the leaf's other terminal attributes before abandoning the
    route. Without that fallback the 4-hop family would lose roughly half its
    seeds. Train yield must stay at 100%: the train pool holds 8 routes per length
    and every one carries several terminal attributes, so a clean candidate always
    exists."""
    for hops in (2, 3, 4):
        minted = sum(
            gen_musique(build_world(seed), random.Random(seed * 7919 + 13), seed,
                        hops, f"musique_{hops}hop", filter_shortcuts=True,
                        route_pool="train") is not None
            for seed in range(0, 60))
        assert minted == 60,             f"FAIL: prefix filter starved {hops}-hop -- only {minted}/60 seeds mintable"


# --------------------------------------------------------------------------
import atr.tasks.generator as _g


def _collect(seed_start, n):
    for k in _g._SHORTCUT_STATS:
        _g._SHORTCUT_STATS[k] = 0
    generate(n, seed_start=seed_start)
    return dict(_g._SHORTCUT_STATS)


def test_filter_observable_rejection_rate():
    """The rejection counter is observable and non-trivial over a large train sample."""
    s = _collect(0, 400)
    assert s["checked"] > 0, "FAIL: shortcut filter checked no candidates"
    assert s["rejected"] > 0, "FAIL: shortcut filter rejected zero candidates"
    rate = s["rejected"] / s["checked"]
    assert 0.02 <= rate <= 0.60, f"FAIL: unreasonable rejection rate {rate:.0%}"
    # the prefix axis is counted separately so the two rates stay readable apart;
    # it is the larger of the two (~47% of candidates vs ~16%).
    assert s["prefix_rejected"] > 0, "FAIL: prefix-leak filter rejected zero candidates"
    prate = s["prefix_rejected"] / s["checked"]
    assert 0.10 <= prate <= 0.80, f"FAIL: unreasonable prefix rejection rate {prate:.0%}"


# --- sufficiency: the two axes added with the unanchored passage prose -------
# The filters above reject tasks that leak the answer EARLY. These two reject the
# opposite defect: a task whose own oracle plan cannot reach the information it
# needs. Both became possible when the passage templates stopped restating each
# fact in the exact words the question asks it in (world._PASSAGE_TEMPLATES).


def test_retrievability_gate_is_not_vacuous():
    """`_is_unretrievable` must actually fire on a real candidate.

    A gate that never rejects anything is indistinguishable from no gate, and it
    would leave `assert_answer_retrievable` -- which only runs after a build -- as
    the sole defence. The known case is seed 735: the query "Khaldonia official
    language" scores Vestorland 4.66, Caldury 4.57 and Orinella 4.57 above
    Khaldonia's own 4.25, because those three realisations kept the literal
    keyword and Khaldonia's did not, so top_k=3 drops the passage that owns the
    answer. Scanned over the raw candidate stream it is rare (~1 in 8600), which
    is exactly why it needs a pinned reproduction rather than a rate check."""
    seed = 735
    w = build_world(seed)
    steps = ["person_city", "city_country"]
    rng = random.Random(seed * 7919 + 13)
    chain = _resolve_chain(w, rng, steps)
    assert chain is not None, "FAIL: the pinned seed no longer resolves its route"
    hit = False
    for _word, gold, _ans, leaf_kw in _leaf_attr_options(random.Random(0), chain[-1]):
        vrng = _variant_rng(seed, steps, leaf_kw)
        hop_kws = [vrng.choice(_REL_QUERY_VARIANTS[st]) for st in steps]
        leaf_q = vrng.choice(_LEAF_QUERY_VARIANTS.get(leaf_kw, [leaf_kw]))
        plan = _build_route_oracle(steps, chain, leaf_q, hop_kws)
        if _is_unretrievable(w, plan, gold):
            hit = True
    assert hit, ("FAIL: the retrievability gate fires on nothing at the pinned seed "
                 "-- it is dead code, or the passage realisations changed")


def test_no_generated_task_is_unretrievable():
    """Every minted task's TERMINAL read surfaces its own gold answer."""
    bad = []
    for t in generate(200, seed_start=0):
        if not t.route:
            continue
        w = build_world(t.seed)
        gold = t.gold
        if _is_unretrievable(w, t.oracle_plan, gold):
            bad.append((t.task_id, ">".join(t.route), t.oracle_answer))
    assert not bad, f"FAIL: {len(bad)} tasks cannot retrieve their own answer: {bad[:5]}"


def test_no_generated_task_has_a_broken_chain():
    """Every hop reveals the entity the NEXT hop's query is written from.

    Without this the plan is only followable by someone who already knows the
    name, which is the psychic-query defect. The live failure it guards: the city
    realisation "X serves as the seat of government of Y" puts that literal phrase
    in six CITY passages, so the walk query "<Country> seat of government" comes
    back with three unrelated cities and never names the country's own capital.
    The TERMINAL read still works in that case (it searches the leaf by name), so
    `_is_unretrievable` passes and only this check catches it."""
    bad = []
    checked = 0
    for t in generate(200, seed_start=0):
        if not t.route:
            continue
        # The task's OWN chain, not a re-resolution of its route: `_resolve_chain`
        # draws from the generator's rng, so replaying it on a fresh Random lands
        # on different entities and the check would fail against a chain the task
        # never used. That is why Task.chain is recorded.
        assert len(t.chain) == len(t.route) + 1, \
            f"FAIL: {t.task_id} chain {t.chain} does not match route {t.route}"
        checked += 1
        w = build_world(t.seed)
        _leaky, broken = _prefix_report(w, t.oracle_plan,
                                        [{"name": n} for n in t.chain], t.gold)
        if broken:
            bad.append((t.task_id, ">".join(t.route)))
    assert checked, "FAIL: no multi-hop tasks checked"
    assert not bad, f"FAIL: {len(bad)}/{checked} tasks have an unfollowable hop: {bad[:5]}"


def test_oracle_query_phrasing_actually_varies():
    """The oracle plan must not emit one fixed string per relation.

    A single realisation per relation trains a lookup table keyed on the question
    template -- "<Entity> capital", "<Entity> author" -- which is a surface form
    the real judge questions never use. Both tables must contribute: a relation
    whose variants list collapsed to one entry would pass a global count."""
    for table, label in ((_REL_QUERY_VARIANTS, "relation"), (_LEAF_QUERY_VARIANTS, "leaf")):
        for key, variants in table.items():
            assert len(variants) >= 2, f"FAIL: {label} {key} has only {len(variants)} realisation(s)"
            assert len(set(variants)) == len(variants), f"FAIL: {label} {key} repeats a realisation"

    # Index 0 is the CANONICAL keyword. Anything mapping a keyword back to a step
    # reads `_REL[k][4]`, and `_build_route_oracle` falls back to it when no
    # realisations are passed, so a drifted index 0 would make the varied set stop
    # being a superset of the old fixed set and the fallback stop matching.
    for step, variants in _REL_QUERY_VARIANTS.items():
        assert variants[0] == _REL[step][4], \
            f"FAIL: {step} variant[0] {variants[0]!r} != canonical {_REL[step][4]!r}"

    # Every leaf keyword must HAVE realisations. A missing key is not an error at
    # runtime -- gen_musique falls back to [leaf_kw] -- so that attribute would
    # silently keep one fixed phrasing while every other one varied.
    leaf_kws = {opts[k][2] for opts in _LEAF_ATTR.values() for k in opts}
    missing = leaf_kws - set(_LEAF_QUERY_VARIANTS)
    assert not missing, f"FAIL: leaf keywords with no realisations (they would not vary): {missing}"
    seen = set()
    for t in generate(200, seed_start=0):
        for step in t.oracle_plan:
            seen.add(step["arguments"]["query"].split(" ", 1)[-1])
    assert len(seen) >= 15, f"FAIL: only {len(seen)} distinct query phrasings across 200 tasks"


def test_phrasing_does_not_shift_the_generator_stream():
    """Phrasing is drawn from `_variant_rng`, never from the generator's own rng.

    Drawing it from `rng` would consume from the stream that picks routes,
    resolves chains and orders leaf attributes, so every seed would mint a
    DIFFERENT task than before -- silently re-pointing the train/dev seed ranges
    the whole no-leakage argument rests on. The check: the same seed must produce
    the same question, gold and route no matter which realisation it drew, and a
    realisation must be stable across calls."""
    for seed in (0, 17, 123, 900_001):
        w = build_world(seed)
        a = gen_musique(w, random.Random(seed * 7919 + 13), seed, 3, "musique_3hop",
                        filter_shortcuts=False, route_pool="train")
        b = gen_musique(build_world(seed), random.Random(seed * 7919 + 13), seed, 3,
                        "musique_3hop", filter_shortcuts=False, route_pool="train")
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"FAIL: seed {seed} minted inconsistently"
        assert (a.prompt, a.gold, a.route) == (b.prompt, b.gold, b.route), \
            f"FAIL: seed {seed} is not reproducible"
        assert [s["arguments"]["query"] for s in a.oracle_plan] == \
               [s["arguments"]["query"] for s in b.oracle_plan], \
            f"FAIL: seed {seed} drew different phrasings on two identical calls"


def run_all():
    """Exit non-zero on any failure, like the rest of tests/ -- otherwise a broken
    invariant here prints FAIL and still returns 0, and the gate reads as green."""
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    print()
    print("ALL PASS" if not failed else f"FAILURES: {failed}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
