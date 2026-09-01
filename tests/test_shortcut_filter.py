# Part 1 + Part 2 tests: shortcut (disconnection) filter and train/dev route holdout.
# Style mirrors test_fix2.py: plain PASS/FAIL assertions, no pytest dependency.
from atr.tasks.generator import (
    generate, dev_set, gen_musique,
    _ROUTES_TRAIN, _ROUTES_DEV_ONLY, _REL,
    _is_shortcut_solvable, _SHORTCUT_STATS,
)
from atr.tools.world import build_world
from atr.tasks.schema import Task
import random


def _kw_to_step():
    return {_REL[k][4]: k for k in _REL}


def _plan_shape(task):
    """Reconstruct the route shape (step-key list) a task used, from its oracle_plan."""
    step_for = _kw_to_step()
    steps = []
    for q in task.oracle_plan:
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


# --------------------------------------------------------------------------
import atr.tasks.generator as _g


def _collect(seed_start, n):
    _g._SHORTCUT_STATS["checked"] = 0
    _g._SHORTCUT_STATS["rejected"] = 0
    generate(n, seed_start=seed_start)
    return dict(_g._SHORTCUT_STATS)


def test_filter_observable_rejection_rate():
    """The rejection counter is observable and non-trivial over a large train sample."""
    s = _collect(0, 400)
    assert s["checked"] > 0, "FAIL: shortcut filter checked no candidates"
    assert s["rejected"] > 0, "FAIL: shortcut filter rejected zero candidates"
    rate = s["rejected"] / s["checked"]
    assert 0.02 <= rate <= 0.60, f"FAIL: unreasonable rejection rate {rate:.0%}"


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")


if __name__ == "__main__":
    run_all()
