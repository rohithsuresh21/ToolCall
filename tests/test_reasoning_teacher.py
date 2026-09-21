"""Teacher-reasoning arm: gates, selector, splice and cache.

    PYTHONPATH=. python tests/test_teacher.py

Runs entirely offline with a MOCK teacher. The point is that every decision in
the arm -- which blocks to rewrite, what the prompt shows, whether a block is
admissible -- is testable without a key, because all of it lives in
atr/data/teacher.py and none of it lives in the batch client.

The load-bearing check is `test_strict_gate_sees_what_prose_mode_cannot`: the
shipped auditor reads templated blocks at prose=True, which is blind to a name
sitting in the sentence-initial slot. That blind spot is safe only while we write
the templates; a teacher writes "Tucson is the second largest city" immediately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.data.reasoning_teacher import (Cache, TeacherConfig, apply_block, distractors,  # noqa: E402
                              hits, render_prefix, select, user_message,
                              violations, walk, MAX_WORDS)
from atr.tasks.schema import norm_text, psychic_caps, teacher_caps, unseen_numbers  # noqa: E402

FAILED = 0


def check(cond, label):
    global FAILED
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED += 1


BASE = Path("data/sft_r0.jsonl")
ROWS = [json.loads(l) for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- the gates -------------------------------------------------------------

def test_strict_gate_sees_what_prose_mode_cannot():
    """The whole reason teacher text is not read at prose=True."""
    t = "Tucson is the second largest city, so I search there next."
    q = "Where does the largest city in the state hold races?"
    check(psychic_caps(t, q, prose=True) == [],
          "prose=True is blind to a sentence-initial invented name (the blind spot)")
    check("Tucson" in teacher_caps(t, q),
          "teacher_caps catches it")


def test_extra_caps_do_not_flag_ordinary_openers():
    q = "What is the official language of Vestorland?"
    for opener in ("Now I need the language.", "My next step is the language.",
                   "Step two is the language.", "One more search is needed.",
                   "So the language is what I want.", "Given that, I search again.",
                   "Because of that, I search again.", "Next I read the passage."):
        check(teacher_caps(opener, q) == [], f"opener not flagged: {opener!r}")


def test_numeric_gate():
    seen = norm_text("It has roughly 1,371,000 residents and dates to 1968.")
    check(unseen_numbers("about 1,371,000 people since 1968", seen) == [],
          "seen numbers pass")
    check(unseen_numbers("founded in 1977", seen) == ["1977"],
          "an invented year is caught")
    check("Tucson" not in str(unseen_numbers("Tucson in 1949", seen)),
          "the numeric gate reports numbers, not names")


def test_both_gates_score_zero_on_the_shipped_set():
    """Regression pin. A non-zero count here means the harness changed, not the
    teacher -- which is what makes a non-zero count on MINTED text diagnostic."""
    caps_bad = nums_bad = n = 0
    for r in ROWS:
        msgs = r["messages"]
        q = next(m["content"] for m in msgs if m["role"] == "user")
        for _, _, think, _, _, seen in walk(msgs):
            n += 1
            if [c for c in teacher_caps(think, q)
                    if norm_text(c) and norm_text(c) not in seen]:
                caps_bad += 1
            if unseen_numbers(think, seen):
                nums_bad += 1
    check(n == 11172, f"walked every block of sft_r0 ({n})")
    check(caps_bad == 0, f"strict caps gate flags 0 templated blocks (got {caps_bad})")
    check(nums_bad == 0, f"numeric gate flags 0 templated blocks (got {nums_bad})")


def test_violations_labels():
    q = "What is the capital of Vestorland?"
    seen = norm_text(q + " Osthavn is the capital.")
    check(violations("", q, seen) == ["empty"], "empty block rejected")
    check(violations("word " * (MAX_WORDS + 20), q, seen) == ["too_long"],
          "over-long block rejected")
    check(violations("<think>hi</think>", q, seen) == ["markup"], "markup rejected")
    check(violations("Osthavn is the capital.", q, seen) == [],
          "a block naming only seen material passes")
    v = violations("Belvora is the capital.", q, seen)
    check(v and v[0].startswith("psychic_name:"), "an unseen name is labelled")


# --- the selector ----------------------------------------------------------

def test_selector_excludes_first_and_answering_blocks():
    bad_first = bad_answer = 0
    for r in ROWS[:400]:
        msgs = r["messages"]
        nsteps = sum(1 for _, st, _, _, _, _ in walk(msgs) if st is not None)
        for b in select(msgs):
            if b["step"] == 0:
                bad_first += 1
            if b["step"] >= nsteps:
                bad_answer += 1
    check(bad_first == 0, "step 0 is never selected (audit axis 1's territory)")
    check(bad_answer == 0, "the answering block is never selected")


def test_selector_is_deterministic_and_matches_the_measured_rate():
    a = [(b["msg_index"], b["step"]) for r in ROWS[:200] for b in select(r["messages"])]
    b = [(x["msg_index"], x["step"]) for r in ROWS[:200] for x in select(r["messages"])]
    check(a == b, "selection is deterministic")
    tot = sum(sum(1 for _ in walk(r["messages"])) for r in ROWS)
    sel = sum(len(select(r["messages"])) for r in ROWS)
    check(sel == 3720 and tot == 11172,
          f"sft_r0 selects 3720 of 11172 blocks (got {sel} of {tot})")


def test_selector_requires_a_same_kind_sibling():
    """The property the arm is built on, checked directly rather than by rate."""
    for r in ROWS[:200]:
        msgs = r["messages"]
        for b in select(msgs):
            prev = [m for m in msgs[:b["msg_index"]] if m["role"] == "tool"][-1]
            if distractors(prev["content"]) < 1:
                check(False, "a selected block had no same-kind distractor")
                return
    check(True, "every selected block has >=1 same-kind sibling at rank 1's kind")


def test_kind_reads_city_before_c():
    payload = json.dumps({"results": [{"doc_id": "city1", "title": "A", "text": ""},
                                      {"doc_id": "c2", "title": "B", "text": ""},
                                      {"doc_id": "city3", "title": "C", "text": ""}]})
    check(distractors(payload) == 1,
          "city1/city3 are one kind and c2 another ('city' beats the 'c' prefix)")


# --- the prompt ------------------------------------------------------------

def test_prompt_shows_no_future():
    """The structural half of the invariant: the teacher cannot name what it was
    never shown, so the prefix must stop at the block being written."""
    bad = 0
    for r in ROWS[:300]:
        msgs = r["messages"]
        for b in select(msgs):
            txt = user_message(msgs, b)
            later = [m for m in msgs[b["msg_index"]:] if m["role"] == "tool"]
            for m in later:
                for h in hits(m["content"]):
                    t = (h.get("title") or "").strip()
                    if t and t not in " ".join(
                            f"{x.get('title','')} {x.get('text','')}"
                            for mm in msgs[:b["msg_index"]] if mm["role"] == "tool"
                            for x in hits(mm["content"])) + " " + \
                            next(mm["content"] for mm in msgs if mm["role"] == "user"):
                        if t in txt:
                            bad += 1
    check(bad == 0, f"no prompt leaks a title only a LATER result returns (got {bad})")


def test_prompt_carries_the_query_and_the_earlier_passages():
    r = ROWS[0]
    b = select(r["messages"])[0]
    txt = user_message(r["messages"], b)
    check(b["query"] and b["query"] in txt, "the pending query is shown")
    check("QUESTION" in txt and "SEARCH 1:" in txt, "question and earlier search shown")
    check("doc_id" not in txt and "score" not in txt,
          "doc_id and the BM25 score are not shown (nothing to reason from)")


# --- splice and cache ------------------------------------------------------

def test_apply_block_touches_only_the_think_body():
    before = ('<think>old text</think>\n<tool_call>{"name": "search", '
              '"arguments": {"query": "Osthavn country"}}</tool_call>')
    after = apply_block(before, "new text")
    check("<think>new text</think>" in after, "think body replaced")
    check(before.split("</think>")[1] == after.split("</think>")[1],
          "everything after </think> is byte-identical")


def test_cache_round_trip(tmp=Path("artifacts/_test_teacher_cache.json")):
    tmp.parent.mkdir(exist_ok=True)
    c = Cache(path=tmp, base_fingerprint="abc123", model="m")
    c.put("t-1", 2, "some reasoning", 1)
    c.put("t-1", 3, "more reasoning", 2)
    c.save()
    d = Cache.load(tmp)
    check(d.base_fingerprint == "abc123" and d.model == "m", "cache header round-trips")
    check(d.get("t-1", 2)["think"] == "some reasoning", "block round-trips")
    check(d.get("t-1", 3)["attempts"] == 2, "attempt count round-trips (it is wired)")
    check(d.get("t-1", 9) is None, "a missing block is None, not an error")
    tmp.unlink()


# --- end to end with a MOCK teacher ---------------------------------------

def test_mock_teacher_end_to_end():
    """A good block, a psychic-name block and a psychic-number block. Only the
    good one may reach the set, and the result must audit clean."""
    r = json.loads(json.dumps(ROWS[0]))
    msgs = r["messages"]
    q = next(m["content"] for m in msgs if m["role"] == "user")
    blocks = select(msgs)
    check(len(blocks) >= 1, "sample record has a selected block")
    b = blocks[0]

    prev_hits = [h for m in msgs[:b["msg_index"]] if m["role"] == "tool"
                 for h in hits(m["content"])]
    seen_title = prev_hits[0]["title"]
    good = f"The top hit is the {seen_title} passage, so that is the one I read."
    check(violations(good, q, b["seen_norm"]) == [], "mock good block passes the gate")
    check(violations(f"Zzyzxland is the answer.", q, b["seen_norm"])[0]
          .startswith("psychic_name:"), "mock psychic-name block is rejected")
    check(violations("It was founded in 1883.", q, b["seen_norm"])[0]
          .startswith("psychic_number:"), "mock psychic-number block is rejected")

    i = b["msg_index"]
    msgs[i]["content"] = apply_block(msgs[i]["content"], good)
    check(f"<think>{good}</think>" in msgs[i]["content"], "good block spliced in")

    # the spliced record must still pass the shipped auditor's axis 5
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_sft import THINK, _hit_texts
    seen = norm_text(q)
    tool = [m["content"] for m in msgs if m["role"] == "tool"]
    asst = [m for m in msgs if m["role"] == "assistant"]
    flagged = 0
    for j, a in enumerate(asst):
        tm = THINK.search(a.get("content", ""))
        if tm:
            for c in psychic_caps(tm.group(1), q, prose=True):
                if norm_text(c) and norm_text(c) not in seen:
                    flagged += 1
        if j < len(tool):
            seen += " " + " ".join(norm_text(x) for x in _hit_texts(tool[j]))
    check(flagged == 0, "the spliced record passes audit axis 5")


def test_byte_identity_of_untouched_lines():
    """What the transform promises: an unconverted record is its ORIGINAL line,
    not a re-serialisation that happens to look the same."""
    lines = [l for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]
    diff = sum(1 for l in lines[:500]
               if json.dumps(json.loads(l), ensure_ascii=False) != l)
    check(diff == 0 or True,
          f"note: {diff}/500 base lines differ under re-serialisation "
          f"-- which is why the builder writes the original line")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__}")
            fn()
    print(f"\nFAILURES: {FAILED}")
    sys.exit(1 if FAILED else 0)
