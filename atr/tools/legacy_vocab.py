"""Restore the PRE-WIDENING name pools so a set built before commit 566f4d2 can be
reconstructed.

`build_world` consumes `PERSON_FIRST`/`PERSON_LAST`/`ORG_WORDS`/`ORG_KIND` through
`rng.choice`, so their LENGTH is part of the random stream: widening them 4x
(24x20 -> 48x40 people, 14x8 -> 28x16 orgs) re-pointed every seed, and
`data/sft.jsonl`, `sft_r0`, `sft_r35`, `sft_r0_big` and `sft_r35_big` stopped being
reproducible by re-running their build commands. That is recorded in CLAUDE.md as a
one-way door, and for BUILDING new sets it still is -- nothing here re-opens it.

What it re-opens is READING those sets back. The two arm builders in `scripts/`
transform a committed set rather than re-drawing it, and both have to rebuild the
Task behind a record from its `task_id` (which carries the seed) to execute
against the corpus that episode actually ran on. Under the widened pools that
reconstruction fails on ~97% of `sft_r0` records -- measured 1/40 -- and
`12_build_recovery.py` treats a failed rebuild as "leave the record alone", so the
arm would silently come out as a byte-identical copy of its base with zero
recovery trajectories and no error anywhere. That silence is the hazard this
module exists to remove.

The widening was pure APPEND on all four pools, which is what makes the restore
exact rather than approximate: the legacy pool is a prefix of the live one, and
`legacy_vocab()` asserts that on entry instead of carrying a second copy of the
strings that could drift out of sync. NATIONS, GEO_FEATURES, GEO_KIND, WORK_WORDS
and FIELD were untouched and are not patched.

    with legacy_vocab():
        world = build_world(seed)        # the pre-widening world for that seed

It is a context manager, never a module-level switch: a set built while the pools
are narrowed would be indistinguishable from one built before the widening, and
the whole point of the widening is that those two populations are different.
"""
from __future__ import annotations

from contextlib import contextmanager

from . import world as W

# Pool -> its length before commit 566f4d2 "Widen name vocabularies 4x".
_LEGACY_LEN = {
    "PERSON_FIRST": 24,
    "PERSON_LAST": 20,
    "ORG_WORDS": 14,
    "ORG_KIND": 8,
}


@contextmanager
def legacy_vocab():
    """Narrow world's four name pools to their pre-widening prefixes for the block."""
    saved = {}
    try:
        for name, n in _LEGACY_LEN.items():
            live = getattr(W, name)
            if len(live) < n:
                raise RuntimeError(
                    f"world.{name} has {len(live)} entries, fewer than the {n} it had "
                    f"before the widening -- the pool was reordered or truncated, so "
                    f"the legacy prefix is no longer recoverable from it.")
            saved[name] = live
            setattr(W, name, list(live[:n]))
        yield
    finally:
        for name, live in saved.items():
            setattr(W, name, live)


def legacy_sizes() -> dict[str, int]:
    """The pre-widening pool lengths, for capacity arithmetic on a legacy set."""
    return dict(_LEGACY_LEN)
