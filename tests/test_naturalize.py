# Part A tests: LLM passage naturalization (facts untouched, scoring isolated).
# Uses a mock LLM client -- NO API key, fully offline.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.data.naturalize import (
    naturalize_passage, naturalize_passages, load_naturalized_loader,
    _facts_present, _surfaced_facts, _format_for_check,
)
from atr.tools.world import build_world
import atr.tasks.generator as gen
import random


class MockLLM:
    """Success-path mock: returns a 'naturalized' rewrite that keeps every fact
    (copies every surfaced value verbatim into new prose)."""
    def __init__(self, prompt_reply=None):
        self.prompt_reply = prompt_reply or (lambda p: _copy_facts(p))
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.prompt_reply(prompt)


def _copy_facts(prompt):
    """Extract the facts list from the prompt and weave them into fresh prose."""
    # FACTS is emitted as a JSON dict; reuse every value present in the ORIGINAL text.
    import re, json
    m = re.search(r'CURRENT PASSAGE: \"(.*)\"$', prompt, re.S)
    orig = m.group(1) if m else ""
    name = orig.split()[0] if orig else ""
    new = f"Some fresh varied prose about {name}: it spans "
    m2 = re.search(r'FACTS: (\{.*?\})\n', prompt, re.S)
    facts = json.loads(m2.group(1)) if m2 else {}
    for v in facts.values():
        if v is not None:
            new += f"{v}; "
    if name:
        new += f" and the entity is {name}"
    return new.strip() or orig


class DropFactMock(MockLLM):
    """Adversarial mock: always returns prose with a SURFACED fact removed, so the
    post-check must reject it and eventually fall back to the original text."""
    def complete(self, prompt):
        self.calls += 1
        import re, json
        m2 = re.search(r'FACTS: (\{.*?\})\n', prompt, re.S)
        facts = json.loads(m2.group(1)) if m2 else {}
        # deliberately omit the first surfaced value
        dropped = None
        for v in facts.values():
            if v is not None:
                dropped = v
                break
        return "Naturalized prose without " + (str(dropped) if dropped else "X")


class AddFactMock(MockLLM):
    """Adversarial mock: adds a fact that is NOT in the attribute list, which is
    forbidden even if every true fact survives."""
    def complete(self, prompt):
        self.calls += 1
        # start from a correct copy then inject a fabricated claim
        base = _copy_facts(prompt)
        return base + " It was founded in 1066 A.D. (a fabricated extra fact)."


# --------------------------------------------------------------------------
def test_facts_preserved_on_success():
    w = build_world(123)
    d = dict(w.documents[0])
    nat = naturalize_passage(d, MockLLM())
    assert _facts_present(d, nat["text"]), "FAIL: surfaced fact lost in naturalized text"
    assert nat["text"] != d["text"], "FAIL: text not actually varied"


def test_dropped_fact_is_rejected_and_falls_back():
    w = build_world(123)
    d = dict(w.documents[0])
    nat = naturalize_passage(d, DropFactMock())
    # retries exhausted -> must fall back to original, never ship broken prose
    assert nat["text"] == d["text"], "FAIL: dropped-fact rewrite shipped instead of fallback"
    assert nat.get("naturalized") is False, "FAIL: fallback not marked"


def test_api_never_fabricates_facts():
    """naturalize_passage only ever touches prose -- it must NOT add, drop or edit
    the ``facts`` dict (which is the sole source of gold answers). Even when the
    LLM returns prose with an out-of-list fact, the module's own data is clean."""
    w = build_world(123)
    d = dict(w.documents[0])
    facts_before = dict(d["facts"] or {})
    attrs_before = dict(d.get("attrs") or {})
    nat = naturalize_passage(d, AddFactMock())
    assert nat["facts"] == facts_before, "FAIL: naturalize mutated the facts dict"
    assert nat.get("attrs", {}) == attrs_before, "FAIL: naturalize mutated attrs"
    # only text / title / naturalized markers may differ from the input copy
    changed = {k for k in nat if nat[k] != d.get(k)}
    assert changed <= {"text", "title", "naturalized"}, f"FAIL: naturalize touched {changed}"


def test_numfmt_matches_commas():
    """A population surfaced as 9,800,000 must match the raw int 9800000."""
    assert _format_for_check(9_800_000) == _format_for_check("9,800,000")


def test_scoring_isolation():
    """Naturalization only rewrites doc text -- attrs, gold, oracle answers are
    identical, so every scorer reads the same values either way."""
    seeds = [5, 77, 900]
    for seed in seeds:
        wt = build_world(seed)
        rng = random.Random(seed * 101 + 3)
        t = gen.gen_musique(wt, rng, seed, 2, "musique_2hop", filter_shortcuts=False,
                            route_pool="train")
        if t is None:
            continue
        wn = build_world(seed)
        naturalize_passages(wn, MockLLM())
        # attrs (source of gold) untouched
        for e_t, e_n in zip(wt.entities, wn.entities):
            assert e_t["attrs"] == e_n["attrs"], f"FAIL: seed {seed} attrs changed by naturalization"
        # gold + oracle answer identical (derived from attrs, not prose)
        tn = gen.gen_musique(wn, random.Random(seed * 101 + 3), seed, 2, "musique_2hop",
                             filter_shortcuts=False, route_pool="train")
        assert tn is not None
        assert t.gold == tn.gold, f"FAIL: seed {seed} gold changed"
        assert t.oracle_answer == tn.oracle_answer, f"FAIL: seed {seed} oracle answer changed"


def test_loader_opt_in_wiring():
    """load_naturalized_loader + build_world(text_loader=...) swaps prose only."""
    w = build_world(11)
    d0 = dict(w.documents[0])
    # build a tiny cache with a recognizable fake prose for every doc
    cache_dict = {f"{w.seed}:{dd['doc_id']}": f"FAKED PROSE {i}" for i, dd in enumerate(w.documents)}
    import json, tempfile, os
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f)
    try:
        loader = load_naturalized_loader(path)
        assert loader is not None, "FAIL: loader not produced from valid cache"
        wn = build_world(11, text_loader=loader)
        prose = [dd["text"] for dd in wn.documents]
        assert all("FAKED PROSE" in p for p in prose), "FAIL: loader text not applied"
        # attrs still intact (only prose swapped)
        for e_t, e_n in zip(w.entities, wn.entities):
            assert e_t["attrs"] == e_n["attrs"]
    finally:
        os.remove(path)


def test_loader_none_when_missing():
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)  # non-existent file
    assert load_naturalized_loader(path) is None, "FAIL: missing cache should yield None"
    # empty dict
    import json
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({}, f)
    try:
        assert load_naturalized_loader(path) is None, "FAIL: empty cache should yield None"
    finally:
        os.remove(path)


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
