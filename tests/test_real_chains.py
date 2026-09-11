# Tests for atr/data/real_chains.py (MuSiQue #N resolution) and the synthesised
# reasoning that quotes its output, atr/data/reasoning.py.
#
# Style mirrors test_shortcut_filter.py: plain script, PASS/FAIL per check, no
# pytest. Run it as `python tests/test_real_chains.py`.
#
# Why the rejections below are PINNED to specific task_ids rather than sampled.
# Each gate fires on 1-33% of the pool, and the two that matter most fire on ~3%.
# A sampling test over a 3% gate is indistinguishable from dead code -- exactly
# the argument test_shortcut_filter.py makes for pinning seed 735. The three
# 4-hop rows pinned for the hop-2+ writability gate are the ones that actually
# tripped the auditor in the first combined smoke build, so they are a
# regression pin, not an illustration.
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.data import reasoning  # noqa: E402
from atr.data.real_chains import (  # noqa: E402
    ChainConfig, Index, _resolve, build_chain_tasks, is_entity_like,
    load_chain_tasks, load_rows, tidy_query, tokens, verify_index,
)
from atr.tasks.generator import generate  # noqa: E402
from atr.tasks.schema import Task, norm_text, psychic_caps  # noqa: E402

SRC = pathlib.Path("data/musique_train_tasks.jsonl")
POOL = pathlib.Path("data/musique_chain_tasks.jsonl")

_ROWS = None


def rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = {r["task_id"]: r for r in load_rows(SRC)}
    return _ROWS


# One pinned row per rejection axis, plus one that survives all of them. Measured
# 2026-09-11 against data/musique_train_tasks.jsonl (3975 rows).
PINS = {
    "2hop__100331_42157": "substitution_not_in_evidence_hop2",
    "2hop__10188_449977": "no_new_evidence_hop1",
    "2hop__151123_14904": "gold_leaks_at_hop1",
    "2hop__13705_13640": "psychic_first_query",
    "2hop__139851_140916": "psychic_query_hop2",
    "2hop__100543_15014": "",          # accepted
}

# The three rows whose synthesised <think> tripped audit axis 5 in the first
# combined smoke build: MuSiQue's own hop-2 sub-question opens on a proper noun
# ("Wengen", "Federal Detention Center", "All Saints Church, Lockerbie") that
# neither the question nor hop 1's passages contain.
SMOKE_REGRESSIONS = [
    "4hop3__166346_9522_28235_22384",
    "4hop3__5752_362455_28338_160498",
    "4hop3__31642_754527_173395_20507",
]


# --------------------------------------------------------------------------
def test_index_matches_shipped_bm25():
    """The fast Index must rank identically to builtin._search.

    Without this every yield number in the module docstring is a statement about
    a re-implementation rather than about the data."""
    good, checked = verify_index(list(rows().values()), n=25)
    assert checked > 0, "FAIL: verify_index checked nothing"
    assert good == checked, (
        f"FAIL: {good}/{checked} queries rank identically -- the resolution index "
        f"has drifted from the shipped BM25")


def test_pinned_rejections():
    """One pinned row per axis, each rejected for its OWN reason."""
    for tid, want in PINS.items():
        assert tid in rows(), f"FAIL: pinned row {tid} is gone from {SRC}"
        res = _resolve(rows()[tid], ChainConfig())
        if want:
            assert not res.ok and res.reason == want, (
                f"FAIL: {tid} expected rejection {want!r}, got "
                f"ok={res.ok} reason={res.reason!r}")
        else:
            assert res.ok, f"FAIL: {tid} should resolve cleanly, got {res.reason!r}"


def test_later_hop_gate_is_the_thing_that_rejects():
    """The hop-2+ writability gate, not some other axis, is what drops the three
    smoke-build regressions -- and turning it off brings every one of them back.

    A rejection test that only asserts "rejected" cannot tell a working gate from
    a row that was already failing something else."""
    on = ChainConfig()
    off = ChainConfig(require_writable_later_queries=False)
    for tid in SMOKE_REGRESSIONS:
        assert tid in rows(), f"FAIL: pinned row {tid} is gone from {SRC}"
        a, b = _resolve(rows()[tid], on), _resolve(rows()[tid], off)
        assert not a.ok and a.reason.startswith("psychic_query_hop"), (
            f"FAIL: {tid} expected a psychic_query_hopN rejection, got "
            f"ok={a.ok} reason={a.reason!r}")
        assert b.ok, (
            f"FAIL: {tid} is rejected even with the gate off ({b.reason!r}) -- the "
            f"gate is not what removes it, so this row pins nothing")


def test_every_query_in_the_pool_is_writable():
    """The property the gate exists to produce, checked on the committed pool
    rather than on the gate's own return value: no hop's query may name a proper
    noun absent from the question and from every strictly earlier hit."""
    assert POOL.exists(), f"FAIL: {POOL} is missing -- run make_musique_chain_tasks.py"
    bad = []
    for t in load_rows(POOL):
        idx = Index(t["documents"])
        seen = t["prompt"]
        for s in t["oracle_plan"]:
            q = (s["arguments"] or {}).get("query", "")
            caps = psychic_caps(q, seen)
            if caps:
                bad.append((t["task_id"], q, caps))
                break
            seen += " " + " ".join(idx.blob[h] for h in idx.search(q, 3))
    assert not bad, (f"FAIL: {len(bad)} pool rows carry an unwritable query, "
                     f"e.g. {bad[0]}")


def test_pool_plans_are_executable_and_shaped():
    """Committed-pool invariants: no placeholder survives, and a real task's plan
    is `len(route)` searches with NO terminal read -- the synthetic-only
    `len(oracle_plan) == len(route) + 1` rule must never be pointed here."""
    pool = load_rows(POOL)
    assert pool, f"FAIL: {POOL} is empty"
    for t in pool:
        qs = [(s["arguments"] or {}).get("query", "") for s in t["oracle_plan"]]
        assert all("#" not in q for q in qs), f"FAIL: {t['task_id']} kept a placeholder"
        assert len(qs) == len(t["route"]), (
            f"FAIL: {t['task_id']} has {len(qs)} searches for {len(t['route'])} "
            f"evidence ids; a real plan has no terminal read")
        assert len(t["chain"]) == len(qs), (
            f"FAIL: {t['task_id']} chain/plan length mismatch -- reasoning.py "
            f"indexes chain by step and would read off the end")


def test_tidy_query_never_changes_a_bm25_token():
    """tidy_query is cosmetic for retrieval BY CONTRACT: it closes up MuSiQue's
    tokeniser spacing, and if it ever changed a token it would silently re-rank a
    hop. Checked over every raw plan query in the source pool, not a sample."""
    n = 0
    for r in rows().values():
        for s in r["oracle_plan"]:
            q = (s["arguments"] or {}).get("query", "")
            assert tokens(tidy_query(q)) == tokens(q), (
                f"FAIL: tidy_query changed the token stream of {q!r}")
            n += 1
    assert n > 1000, f"FAIL: only {n} queries checked -- the source pool looks wrong"
    # and it must actually DO something, or the guard above is vacuous
    assert tidy_query("Antarctica 's border") == "Antarctica's border"
    assert tidy_query("who colonized #1 ?") == "who colonized #1?"


def test_is_entity_like_rejects_sentence_openers():
    for junk in ("The", "It", "However", "A", "Of the"):
        assert not is_entity_like(junk), f"FAIL: {junk!r} counted as an entity"
    for name in ("Atlantic City", "Beyonce", "Yuma County"):
        assert is_entity_like(name), f"FAIL: {name!r} not counted as an entity"


def test_build_is_deterministic():
    """Two runs over the same slice are byte-identical. A live RNG anywhere in
    resolution would make the committed pool unreproducible and its sha256 a lie."""
    slice_ = list(rows().values())[:300]
    a, sa = build_chain_tasks(slice_, ChainConfig())
    b, sb = build_chain_tasks(slice_, ChainConfig())
    assert sa == sb, "FAIL: rejection stats differ across two identical runs"
    ja = [json.dumps(t, sort_keys=True, ensure_ascii=False) for t in a]
    jb = [json.dumps(t, sort_keys=True, ensure_ascii=False) for t in b]
    assert ja == jb, "FAIL: build_chain_tasks is not deterministic"


# --- the reasoning that quotes those queries -------------------------------
_FIELD = re.compile(r"\{(\w+)\}")
# Fields that expand to a BARE ENTITY NAME. `goal` is deliberately not one: it
# expands to the hop's own query (real) or a relation phrase (synthetic), and
# real_chains gates every query against what the episode has seen, so a
# sentence-initial `goal` is covered by that gate rather than by this rule.
_NAME_FIELDS = {"head", "prev", "leaf", "ans"}


def test_no_template_opens_a_sentence_with_a_bare_entity():
    """audit_sft.py reads <think> blocks with psychic_caps(prose=True), which
    ignores sentence-initial capitals -- in running prose the first word of a
    sentence is capitalised whatever it is. The cost is that a one-word invented
    entity sitting there would be invisible, so no template may put a bare name
    there."""
    tmpls = [(n, t) for n, v in vars(reasoning).items()
             if n.startswith("_") and isinstance(v, list) and v
             and all(isinstance(x, str) for x in v)
             for t in v if _FIELD.search(t)]
    assert len(tmpls) >= 15, f"FAIL: only found {len(tmpls)} templates to check"
    for name, t in tmpls:
        for m in _FIELD.finditer(t):
            if m.group(1) not in _NAME_FIELDS:
                continue
            before = t[:m.start()].rstrip()
            assert before and before[-1] not in ".?!:", (
                f"FAIL: {name} template puts {{{m.group(1)}}} at a sentence-initial "
                f"position, where prose=True cannot see it: {t!r}")


def test_as_subquestion_only_adds_punctuation():
    """The real <think> goal must be the GATED query text and nothing else -- if
    it paraphrased, the build-time writability gate would no longer cover it."""
    for q in ("What country was Signmark from?", "when was the golden nugget built",
              "Federal Detention Center >> country"):
        out = reasoning._as_subquestion(q)
        assert out.rstrip("?") == " ".join(q.split()).rstrip("?"), (
            f"FAIL: _as_subquestion rewrote {q!r} into {out!r}")


def _psychic_think(task, step, seen):
    return [c for c in psychic_caps(reasoning.think_for_call(task, step), seen, prose=True)
            if norm_text(c) and norm_text(c) not in seen]


def test_think_never_names_the_next_entity():
    """Invariant 1 of reasoning.py: the block before step k may name chain[k] --
    put on screen by step k-1 -- and never chain[k+1]."""
    tasks = load_chain_tasks(POOL)[:400] + generate(60, seed_start=0)
    checked = 0
    for task in tasks:
        chain = list(task.chain or [])
        for k in range(len(task.oracle_plan) - 1):
            nxt = chain[k + 1] if k + 1 < len(chain) else ""
            if not nxt or norm_text(nxt) in norm_text(task.prompt):
                continue
            think = reasoning.think_for_call(task, k)
            assert norm_text(nxt) not in norm_text(think), (
                f"FAIL: {task.task_id} step {k} names chain[{k+1}]={nxt!r}: {think!r}")
            checked += 1
    assert checked > 100, f"FAIL: only {checked} steps carried a next-entity to check"


def test_real_think_is_clean_against_what_the_episode_has_seen():
    """End-to-end: synthesise the <think> for every hop of a pool task and check
    it the way tests/audit_sft.py checks a built record -- against the prompt plus
    every strictly earlier hit. This is the axis the hop-2+ gate exists to make
    pass, measured on the reasoning rather than on the query."""
    bad = []
    for t in load_rows(POOL)[:500]:
        task = Task.from_dict(t)
        idx = Index(t["documents"])
        seen = norm_text(t["prompt"])
        for k, s in enumerate(t["oracle_plan"]):
            hit = _psychic_think(task, k, seen)
            if hit:
                bad.append((t["task_id"], k, hit, reasoning.think_for_call(task, k)))
                break
            q = (s["arguments"] or {}).get("query", "")
            seen += " " + " ".join(idx.blob[h] for h in idx.search(q, 3))
    assert not bad, f"FAIL: {len(bad)} pool tasks synthesise a psychic <think>, e.g. {bad[0]}"


def run_all():
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
