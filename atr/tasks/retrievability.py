"""Is the gold answer actually IN the passages the oracle plan retrieves?

This is the one property the oracle eval cannot see. `oracle_backend` builds a
`MockBackend` from `{task_id: oracle_plan}` and `{task_id: oracle_answer}`: it
replays the plan and then emits the stored answer string from a lookup table. It
never reads a tool result. So `eval --backend oracle` prints 100% success whether
or not a single retrieved passage contained the answer -- the number is a
statement about the VERIFIERS, not about the data.

That blind spot hid a real defect. `_build_route_oracle` walked the relation
chain and stopped, so the transcript ended on chain[L-1]'s passage: it NAMES the
leaf but carries none of the leaf's attributes, and the attribute is the answer.
Roughly half of all train tasks -- and 78% of the held-out 4-hop dev shape, whose
person leaf nothing else co-retrieves -- had the gold string nowhere in the
episode. Every one of them still scored 100% under the oracle. Trained on that
data a model cannot do better than guess, and the loss curve looks fine while it
learns to state an answer it was never shown.

So this module replays the plan through the REAL registry and checks the returned
text. Call `assert_answer_retrievable` from the sanity path and from
test_pipeline.py; it fails loudly, naming the routes responsible.

Note the check is deliberately LENIENT -- substring containment anywhere in any
returned passage. It answers "was the answer present at all", which is the floor.
`first_hit_call` (the earliest call whose results already contain the answer) is
the companion signal: a task whose answer shows up before the last call is
solvable in fewer hops than its label claims, which is the shortcut axis rather
than the sufficiency axis measured here.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..tools.adapter import get_registry
from ..tools.world import build_world, world_for_task
from .schema import Task, norm_text, task_source


def check_task(task: Task, env: str = "builtin", text_loader=None) -> dict:
    """Replay one task's oracle_plan through the real registry.

    Returns {retrievable, in_last_call, first_hit_call, num_calls}. Tasks with no
    plan (no_tool) or no gold string (unanswerable) are reported as retrievable:
    there is nothing to retrieve, and failing them would make the assertion
    meaningless for the families it does cover.
    """
    want = norm_text(task.oracle_answer or "")
    calls = [s for s in task.oracle_plan if not s.get("__expect_error__")]
    src = task_source(task)
    if not want or not calls or task.gold.get("kind") == "none":
        return {"retrievable": True, "in_last_call": True, "source": src,
                "answer_in_corpus": True, "plan_executable": True,
                "first_hit_call": None, "num_calls": len(calls), "skipped": True}

    # Real tasks own their corpus; synthetic tasks derive it from the seed. Going
    # through build_world unconditionally searched a synthetic universe for a real
    # answer and called every real task unretrievable.
    world = world_for_task(task, text_loader=text_loader)
    registry = get_registry(env)
    first_hit = None
    hit_last = False
    for i, step in enumerate(calls, start=1):
        res = registry.call(world, step["name"], dict(step.get("arguments", {})))
        # title AND text -- the same pair `_is_prefix_leaky` and
        # `_is_shortcut_solvable` read, and the same pair audit_sft._hit_texts
        # parses. Text alone under-reports `first_hit_call` whenever the gold
        # answer IS an entity name: it leaks through the TITLE of a co-retrieved
        # passage while that passage's prose never spells it out. Never the raw
        # tool_response string -- norm_text deletes punctuation, so gold "1949"
        # would substring-match the BM25 `"score": 5.1949`.
        found = any(want in norm_text(f"{h.get('title', '')} {h.get('text', '')}")
                    for h in res.get("results", []) or [])
        if found and first_hit is None:
            first_hit = i
        if i == len(calls):
            hit_last = found
    # Is the answer in this task's corpus AT ALL, independent of whether the plan
    # can reach it? For a synthetic task the two questions coincide, because its
    # oracle_plan is an executable reference solution built from the same world.
    # For a REAL task they do not: MuSiQue's `question_decomposition` is a sketch,
    # not a plan -- 59.3% of its queries still carry an unresolved "#N" placeholder
    # ("Who besides the british colonized #1 ?"), which is not an executable BM25
    # query. Replaying those calls 52.5% of real tasks unretrievable while 100% of
    # them do contain their answer in their own 20 passages. So `retrievable` is
    # sourced differently per kind, and both raw signals stay visible.
    in_corpus = any(want in norm_text(f"{d.get('title', '')} {d.get('text', '')}")
                    for d in (getattr(task, "documents", None) or []))
    if src == "real":
        return {"retrievable": in_corpus, "in_last_call": hit_last, "source": src,
                "answer_in_corpus": in_corpus, "plan_executable": first_hit is not None,
                "first_hit_call": first_hit, "num_calls": len(calls), "skipped": False}
    return {"retrievable": first_hit is not None, "in_last_call": hit_last, "source": src,
            "answer_in_corpus": first_hit is not None, "plan_executable": first_hit is not None,
            "first_hit_call": first_hit, "num_calls": len(calls), "skipped": False}


def audit(tasks: Sequence[Task], env: str = "builtin", text_loader=None) -> dict:
    """Aggregate `check_task` over a task list. Reports overall and per-family
    rates plus the routes that account for the failures."""
    rows = [(t, check_task(t, env=env, text_loader=text_loader)) for t in tasks]
    scored = [(t, r) for t, r in rows if not r["skipped"]]
    bad = [(t, r) for t, r in scored if not r["retrievable"]]
    early = [(t, r) for t, r in scored
             if r["first_hit_call"] is not None and r["first_hit_call"] < r["num_calls"]]

    by_family: dict[str, dict] = {}
    for fam in sorted({t.task_type for t, _ in scored}):
        fam_rows = [(t, r) for t, r in scored if t.task_type == fam]
        fam_bad = [1 for _, r in fam_rows if not r["retrievable"]]
        by_family[fam] = {
            "n": len(fam_rows),
            "unretrievable": len(fam_bad),
            "unretrievable_rate": round(len(fam_bad) / max(1, len(fam_rows)), 4),
        }
    by_source: dict[str, dict] = {}
    for src in sorted({r["source"] for _, r in scored}):
        src_rows = [(t, r) for t, r in scored if r["source"] == src]
        src_bad = [1 for _, r in src_rows if not r["retrievable"]]
        by_source[src] = {
            "n": len(src_rows),
            "unretrievable": len(src_bad),
            "unretrievable_rate": round(len(src_bad) / max(1, len(src_rows)), 4),
            # only meaningful for real tasks; equals `retrievable` for synthetic
            "plan_executable": sum(1 for _, r in src_rows if r["plan_executable"]),
        }
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "by_source": by_source,
        "n_skipped": len(rows) - len(scored),
        "unretrievable": len(bad),
        "unretrievable_rate": round(len(bad) / max(1, len(scored)), 4),
        "answer_before_last_call": len(early),
        "by_family": by_family,
        "worst_routes": Counter(">".join(t.route) if t.route else t.task_type
                                for t, _ in bad).most_common(10),
        "failures": [(t.task_id, ">".join(t.route) if t.route else "-",
                      t.prompt, t.oracle_answer) for t, _ in bad[:20]],
    }


def assert_answer_retrievable(tasks: Sequence[Task], env: str = "builtin",
                              text_loader=None, max_rate: float = 0.0) -> dict:
    """Raise unless the gold answer is retrievable for (1 - max_rate) of tasks.

    Default 0.0: every scored task must have its answer somewhere in the passages
    its own oracle plan returns. A plan that cannot surface its answer is not a
    reference solution, and any dataset collected from it teaches guessing.
    """
    rep = audit(tasks, env=env, text_loader=text_loader)
    # NOTE for real tasks: `retrievable` means "the answer is in the task's own
    # candidate set", NOT "the oracle_plan retrieves it" -- see check_task. Never
    # point the plan-executability axis at a real pool expecting 0%: MuSiQue's
    # decomposition queries are not executable and ~52% of them do not resolve.
    if rep["unretrievable_rate"] > max_rate:
        lines = [f"  {tid:<28} {route:<52} {ans}"
                 for tid, route, _prompt, ans in rep["failures"]]
        raise AssertionError(
            f"oracle plans do not retrieve their own gold answer for "
            f"{rep['unretrievable']}/{rep['n_scored']} tasks "
            f"({rep['unretrievable_rate']:.1%} > max_rate {max_rate:.1%}).\n"
            f"Worst routes: {rep['worst_routes']}\n"
            f"Examples (task_id, route, gold):\n" + "\n".join(lines))
    return rep
