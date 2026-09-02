"""
Executable verification of a whole trajectory.

The headline number is `success`. Everything else exists so that when success
moves you know *which* skill moved -- and, more usefully, when success does not
move you know which skill to spend the next data batch on.

`success` deliberately is not just "final answer correct":
  * on a no-tool task, reaching the right answer by calling tools anyway is a
    failure of the decision we are training (set strict_necessity=False to relax);
  * on any task, firing a side-effect tool nobody asked for is a failure even if
    the text answer is right;
"""
from __future__ import annotations

from ..agent.loop import Trajectory
from .schema import ScoreCard, Task, answer_f1, match_answer

# The retired send_message tool was the only side-effect tool; with the 8->5
# tool reduction there are no write tools, so nothing to police here. The
# side-effect code paths stay (they are inert) in case a write tool returns.
SIDE_EFFECT_TOOLS: set[str] = set()


def _empty_retrieval(res) -> bool:
    """True when a search result is an empty retrieval (no hits)."""
    if not isinstance(res, dict):
        return False
    return res.get("num_results") == 0 or not res.get("results")


def _norm_val(v):
    """Canonical form for argument comparison: numeric strings coerce, strings
    strip+lower, dicts recurse key-sorted, lists become order-free. Kills the
    false negatives where a semantically identical call is written differently."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s = v.strip()
        try:
            return round(float(s), 6)
        except ValueError:
            return s.lower()
    if isinstance(v, dict):
        return {k: _norm_val(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return sorted((_norm_val(x) for x in v), key=repr)
    if v is None:
        return ""
    return str(v).strip().lower()


def _key_args_match(oracle_args: dict, got_args: dict) -> bool:
    """Compare only the arguments the oracle actually pinned; extra args are fine."""
    for k, v in oracle_args.items():
        if k not in got_args:
            return False
        if _norm_val(got_args[k]) != _norm_val(v):
            return False
    return True


def score(task: Task, traj: Trajectory, strict_necessity: bool = True,
          strict_match: bool = False) -> ScoreCard:
    called = [c["name"] for c in traj.call_log]
    ok_calls = {c["name"] for c in traj.call_log if c.get("ok")}
    errors = [c for c in traj.call_log if not c.get("ok")]

    # F5 (FIX-2): a turn whose generation was cut off mid-stream leaves an
    # unterminated marker behind. Those episodes carry reward noise.
    truncated = any("unterminated_" in e for s in traj.steps for e in s.parse_errors)

    sc = ScoreCard(
        task_id=task.task_id, task_type=task.task_type, difficulty=task.difficulty,
        num_steps=len(traj.steps), num_calls=len(traj.call_log),
        num_tool_errors=len(errors), oracle_calls=len(task.oracle_plan),
        stop_reason=traj.stop_reason)
    sc.detail["truncated"] = bool(truncated)

    # ---- format ---------------------------------------------------------
    sc.format_strict = bool(traj.steps) and all(s.strict_format for s in traj.steps)
    sc.format_loose = all(
        ("malformed_tool_call" not in s.parse_errors and "empty_turn" not in s.parse_errors)
        for s in traj.steps) and bool(traj.steps)

    # ---- necessity ------------------------------------------------------
    used_any = len(called) > 0
    if task.task_type == "compute":
        # F1 (FIX-2): calculator is allowed but not required on compute tasks --
        # neither answering directly nor calling the tool is an error, so the
        # necessity check must be neutral instead of penalising one of them.
        sc.necessity_ok = True
    else:
        sc.necessity_ok = used_any == task.needs_tool

    # ---- selection ------------------------------------------------------
    missing = [t for t in task.required_tools if t not in called]
    missing_groups = [g for g in task.required_any if not any(t in called for t in g)]
    if missing_groups:
        missing = missing + ['|'.join(missing_groups[0])]
    forbidden_used = [t for t in task.forbidden_tools if t in called]
    unrequested_side_effect = [t for t in SIDE_EFFECT_TOOLS
                               if t in called and t not in task.required_tools]
    sc.selection_ok = not missing and not forbidden_used

    # ---- arguments ------------------------------------------------------
    # loose: every required tool produced at least one successful (non-error) call
    loose_args = all(t in ok_calls for t in task.required_tools) and \
        all(any(t in ok_calls for t in g) for g in task.required_any)
    # strict: at least one call matched the oracle's pinned arguments
    strict_hits, strict_total = 0, 0
    for step in task.oracle_plan:
        strict_total += 1
        for c in traj.call_log:
            if c["name"] == step["name"] and _key_args_match(step.get("arguments", {}), c.get("args") or {}):
                strict_hits += 1
                break
    sc.args_ok = loose_args
    sc.detail["args_strict_frac"] = round(strict_hits / strict_total, 3) if strict_total else None
    sc.detail["arg_coercions"] = sum(1 for c in traj.call_log if c.get("coerced"))

    # ---- recovery -------------------------------------------------------
    if errors:
        # recovered if a later call to the same tool succeeded, or the task
        # still ended with a correct answer
        recovered = False
        for e in errors:
            idx = traj.call_log.index(e)
            if any(c.get("ok") and c["name"] == e["name"] for c in traj.call_log[idx + 1:]):
                recovered = True
                break
        sc.recovery_ok = recovered
    # Retrieval world: an empty retrieval (num_results == 0) is the analogue of
    # a hard tool error -- the model must rephrase and try again. Detect it as a
    # recoverable event: an empty result followed by a non-empty search (or a
    # correct final answer).
    if sc.recovery_ok is None:
        empty_idx = [i for i, c in enumerate(traj.call_log)
                     if c.get("ok") and _empty_retrieval(c.get("result"))]
        if empty_idx:
            sc.recovery_ok = any(
                idx < j and not _empty_retrieval(traj.call_log[j].get("result"))
                for idx in empty_idx for j in range(idx + 1, len(traj.call_log)))
    sc.detail["error_kinds"] = sorted({(c.get("result") or {}).get("error", "?") for c in errors})

    # ---- final answer ---------------------------------------------------
    correct, reason = match_answer(task.gold, traj.final_answer, strict=strict_match)
    if strict_match and traj.steps and "untagged_final_answer" in traj.steps[-1].parse_errors:
        # strict mode also refuses answers that were never wrapped in the tag
        correct, reason = False, "untagged_answer_strict"
    sc.final_correct = correct
    # The official judge scores token-level F1 against the gold string, so that
    # is what the reward optimises; final_correct stays boolean for diagnostics.
    # On an unanswerable task there is no gold string to overlap with -- F1
    # against an abstention marker measures phrasing, not correctness -- so the
    # abstention decision itself is the score.
    if task.gold.get("kind") == "none":
        sc.final_f1 = 1.0 if correct else 0.0
    else:
        sc.final_f1 = answer_f1(task.gold, traj.final_answer)
    sc.detail["answer_reason"] = reason
    sc.detail["final_answer"] = (traj.final_answer or "")[:400]

    # ---- side effects ---------------------------------------------------
    if task.expect_side_effect:
        want = task.expect_side_effect
        hit = None
        for m in traj.sent_messages:
            if want.get("to") and m["to"].lower() != want["to"].lower():
                continue
            blob = f"{m['subject']} {m['body']}".lower()
            if all(str(s).lower() in blob for s in want.get("must_contain", [])):
                hit = m
                break
        sc.side_effect_ok = hit is not None
        sc.detail["sent_messages"] = traj.sent_messages
    elif unrequested_side_effect:
        sc.side_effect_ok = False
        sc.detail["unrequested_side_effect"] = unrequested_side_effect

    # ---- headline -------------------------------------------------------
    sc.success = (
        sc.final_correct
        and not forbidden_used
        and sc.side_effect_ok is not False
        and (sc.necessity_ok or not strict_necessity or task.needs_tool)
    )
    if task.expect_side_effect:
        sc.success = sc.success and bool(sc.side_effect_ok)

    # ---- one-line failure mode -----------------------------------------
    if sc.success:
        sc.failure_mode = ""
    elif not sc.format_loose and traj.final_answer is None:
        sc.failure_mode = "format"
    elif traj.stop_reason == "max_steps":
        sc.failure_mode = "ran_out_of_steps"
    elif traj.stop_reason == "empty_turn":
        sc.failure_mode = "empty_turn"
    elif not sc.necessity_ok and not task.needs_tool:
        sc.failure_mode = "unnecessary_tool_use"
    elif not sc.necessity_ok and task.needs_tool:
        sc.failure_mode = "answered_without_tools"
    elif forbidden_used:
        sc.failure_mode = "forbidden_tool"
    elif task.expect_side_effect and not sc.side_effect_ok:
        sc.failure_mode = "missing_or_wrong_side_effect"
    elif sc.side_effect_ok is False:
        sc.failure_mode = "unrequested_side_effect"
    elif missing:
        sc.failure_mode = f"missing_tool:{missing[0]}"
    elif not loose_args:
        sc.failure_mode = "bad_arguments"
    elif sc.recovery_ok is False:
        sc.failure_mode = "no_recovery"
    elif reason == "no_answer":
        sc.failure_mode = "no_final_answer"
    else:
        sc.failure_mode = f"wrong_answer:{reason}"
    return sc
