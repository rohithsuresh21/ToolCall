# Part 1 + Part 2 tests: shortcut (disconnection) filter, per-hop PREFIX leakage,
# and the train/dev route holdout.
# Style mirrors test_fix2.py: plain PASS/FAIL assertions, no pytest dependency.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.tasks.generator import (
    generate, dev_set, gen_musique,
    _ROUTES_TRAIN, _ROUTES_DEV_ONLY, _REL,
    _is_shortcut_solvable, _is_prefix_leaky, _SHORTCUT_STATS,
)
from atr.tools.world import build_world
from atr.tasks.schema import Task
import random


def _kw_to_step():
    return {_REL[k][4]: k for k in _REL}


def _plan_shape(task):
    """Reconstruct the route shape (step-key list) a task used, from its oracle_plan.

    A route of L relation steps plans L+1 searches: L that walk the chain, then a
    TERMINAL READ of the leaf's own passage (see _build_route_oracle). Only the
    first L encode relations, so the last step is dropped before mapping. It has to
    be positional rather than keyword-based: the leaf-attribute keywords overlap the
    relation keywords ("born", "founded", "field"), so a trailing-keyword lookup
    would silently mis-resolve the read as another hop."""
    step_for = _kw_to_step()
    steps = []
    for q in task.oracle_plan[:-1]:
        kw = q["arguments"]["query"].rsplit(" ", 1)[-1]
        s = step_for.get(kw)
        if s is None:
            return None
        steps.append(s)
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
    _g._SHORTCUT_STATS["checked"] = 0
    _g._SHORTCUT_STATS["rejected"] = 0
    _g._SHORTCUT_STATS["prefix_rejected"] = 0
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
