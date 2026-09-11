"""
Synthesised per-step reasoning for oracle trajectories.

WHY THIS EXISTS. A failed 4-hop judge trajectory: "Where does the second largest
city in the state where Yuma's Library District is located hold NASCAR races?"
The model fired "Yuma Arizona Library District", "Yuma County Arizona", "Yuma
County Arizona largest city" -- three rephrasings orbiting the prompt's own
entity, all returning the identical three documents. Tucson was in the top-3 of
call 1 and was never used. Every step carried `"thinking": ""`.

That is not a retrieval failure, it is a *carry* failure: nothing in the turn
transports an entity out of one tool result and into the next query. And it is
exactly what the data teaches. Oracle trajectories are bare `<tool_call>` blocks
-- the reference plan is correct BECAUSE the generator already knew the chain,
so the plan never has to show the step where the chain is discovered. The model
imitates the surface of that: emit a query, receive passages, emit another query
that could have been written before the first one returned.

So we write the missing step down. We know the chain at generation time, so
before each call we emit a <think> block naming what the previous result
revealed and what the next hop needs. This is a BOOTSTRAP, not a diet -- the
warning that used to sit on build_sft._rationale still holds: templated
rationales teach the shape of "think then act" and nothing about adapting when a
result surprises you. GRPO is where the adapting gets learned; this is what gives
it a policy that carries entities at all.

TWO INVARIANTS, both load-bearing:

1. A <think> block may name ONLY what the episode has already seen -- the prompt,
   and the passages returned by strictly earlier calls. Naming the next entity
   before its hop retrieves it is the psychic-query defect wearing a different
   hat, and it is worse here than in a query: a query at least gets scored by
   retrieval, whereas invented deliberation is pure teacher-forcing of a
   hallucination. `_reveals()` is the accessor that keeps this honest -- the
   block before step k may reference `chain[k]` (put on screen by step k-1) and
   never `chain[k+1]`. tests/audit_sft.py re-checks it on the BUILT set against
   the tool responses actually present, which is the check that counts.

2. Phrasing varies across a few templates, chosen by a hash of (task_id, step),
   never by a live RNG. Verbatim-identical reasoning across thousands of records
   is memorised as a fixed string and carries no signal; a live RNG would make
   the build unreproducible and, if drawn from any shared stream, would re-point
   every seed after it (the same argument as generator._variant_rng).

Both task sources are handled here, and the only difference between them is where
the "what the last result revealed" string comes from:

  synthetic -- `Task.chain` (entity names) + `Task.route` (relation keys), which
               the generator recorded precisely because they cannot be recovered
               later (see Task.chain's docstring).
  real      -- `Task.chain` as written by atr/data/real_chains.py: the name each
               MuSiQue `#N` placeholder was resolved to, which that module's own
               gate guarantees OCCURS in the passages hop N returned.
"""
from __future__ import annotations

import hashlib

from ..tasks.generator import _REL
from ..tasks.schema import Task, task_source

# Readable renderings of the terminal leaf keyword. The keyword is what the
# oracle's final search pairs with the leaf name (_LEAF_ATTR[...][2]); this is how
# a human would say it in a sentence. Falls back to the keyword itself, so adding
# a leaf attribute cannot crash a build.
_ATTR_PHRASE = {
    "official language": "official language",
    "population": "population",
    "founded": "year of foundation",
    "field": "field of activity",
    "born": "year of birth",
}

# --- templates -------------------------------------------------------------
# Three per position. More would be cheap but pointless: the variation exists to
# stop the string being memorised verbatim, not to model natural language.
#
# No template may put a BARE ENTITY NAME ({head}/{prev}/{leaf}/{ans}) where a
# sentence starts -- including after a colon, which psychic_caps(prose=True)
# treats as a sentence boundary. That reader ignores sentence-initial capitals
# because in running prose the first word is capitalised whatever it is, so a
# one-word invented entity parked there is invisible to the auditor. "Starting
# point: {head}." did exactly that; it never produced a defect, because a
# synthetic head entity is always named by the prompt, but the invariant is what
# the auditor's blind spot is traded against and it has to actually hold.
# test_real_chains.test_no_template_opens_a_sentence_with_a_bare_entity pins it.
# {goal} is exempt: it expands to the hop's own query, which real_chains gates
# against everything the episode has seen.
_SYN_FIRST = [
    "The question names {head}. To get anywhere I first need {goal}.",
    "My starting point is {head}. Step one is to find {goal}.",
    "The only entity the question gives me is {head}, so I need {goal} first.",
]
_SYN_HOP = [
    "The result names {prev}. Now I need {goal}.",
    "That gives me {prev}. The next hop is {goal}.",
    "Now I have {prev}, and from there I need {goal}.",
]
_SYN_TERMINAL = [
    "The chain ends at {leaf}. The question asks for its {attr}, and that is in "
    "the {leaf} passage rather than the one I just read.",
    "So the entity the question is really about is {leaf}. I still need the {attr}, "
    "which the previous result does not carry.",
    "That resolves to {leaf}. One more search to read the {attr} off the {leaf} passage.",
]
_REAL_FIRST = [
    "This needs {n} steps. First: {goal}",
    "I break the question down. Step 1: {goal}",
    "Before I can answer I have to settle: {goal}",
]
_REAL_HOP = [
    "The result names {prev}. Next: {goal}",
    "That gives me {prev}. Now: {goal}",
    "Now I have {prev}. The next step: {goal}",
]
_REAL_HOP_NOPREV = [
    "Next: {goal}",
    "Now I need: {goal}",
    "The next step is: {goal}",
]
_ANSWER = [
    "The passage gives {ans}. That is what the question asked for.",
    "That passage states {ans}, which is the answer.",
    "So the answer is {ans}.",
]


def _pick(templates: list[str], task_id: str, step: int) -> str:
    """Deterministic template choice. Hash, not RNG: a build must be byte-identical
    across runs."""
    h = hashlib.sha1(f"{task_id}|{step}".encode()).digest()
    return templates[h[0] % len(templates)]


def _reveals(task: Task, step: int) -> str:
    """The entity name the block before step `step` (0-indexed) may legitimately
    refer to: the one the PREVIOUS call's result put on screen. Empty if none."""
    chain = list(task.chain or [])
    return chain[step] if 0 <= step < len(chain) else ""


def _leaf_keyword(task: Task) -> str:
    """The terminal read's keyword, recovered from the oracle plan rather than
    re-derived. The final query is `f"{leaf_name} {leaf_kw}"` and the leaf name is
    chain[-1], so stripping that prefix gives the keyword back exactly."""
    if not task.oracle_plan:
        return ""
    q = (task.oracle_plan[-1].get("arguments", {}) or {}).get("query", "")
    leaf = (list(task.chain) or [""])[-1]
    if leaf and q.lower().startswith(leaf.lower()):
        q = q[len(leaf):]
    return q.strip()


def _query_of(task: Task, step: int) -> str:
    if 0 <= step < len(task.oracle_plan):
        return (task.oracle_plan[step].get("arguments", {}) or {}).get("query", "")
    return ""


def _as_subquestion(q: str) -> str:
    """A real hop's goal phrase. MuSiQue's decomposition sub-questions already read
    as questions once resolved, so they are used verbatim apart from making sure
    they end in punctuation -- the <think> block is prose and a dangling fragment
    reads as a truncation."""
    q = " ".join((q or "").split())
    if q and q[-1] not in ".?!":
        q += "?"
    return q


def think_for_call(task: Task, step: int) -> str:
    """The <think> text preceding call `step` (0-indexed), or "" when nothing
    honest can be said about it."""
    if task_source(task) == "real":
        return _think_real(task, step)
    return _think_synthetic(task, step)


def _think_synthetic(task: Task, step: int) -> str:
    route, chain = list(task.route or []), list(task.chain or [])
    # An L-hop route plans L+1 searches; step L is the terminal read.
    if not route or len(chain) != len(route) + 1 or len(task.oracle_plan) != len(route) + 1:
        return ""
    if step == len(route):
        kw = _leaf_keyword(task)
        return _pick(_SYN_TERMINAL, task.task_id, step).format(
            leaf=chain[-1], attr=_ATTR_PHRASE.get(kw, kw or "value"))
    if step < 0 or step >= len(route):
        return ""
    src = _reveals(task, step)          # prompt (step 0) or the previous result
    rel = _REL.get(route[step])
    if not src or rel is None:
        return ""
    goal = rel[3].format(src)           # "the capital of Arizona"
    if step == 0:
        return _pick(_SYN_FIRST, task.task_id, step).format(head=src, goal=goal)
    return _pick(_SYN_HOP, task.task_id, step).format(prev=src, goal=goal)


def _think_real(task: Task, step: int) -> str:
    if step < 0 or step >= len(task.oracle_plan):
        return ""
    goal = _as_subquestion(_query_of(task, step))
    if not goal:
        return ""
    if step == 0:
        return _pick(_REAL_FIRST, task.task_id, step).format(
            n=len(task.oracle_plan), goal=goal)
    prev = _reveals(task, step)
    if prev:
        return _pick(_REAL_HOP, task.task_id, step).format(prev=prev, goal=goal)
    return _pick(_REAL_HOP_NOPREV, task.task_id, step).format(goal=goal)


def think_for_answer(task: Task, answer: str | None) -> str:
    """The <think> text on the final, answering turn. It may name the answer: the
    terminal read has already returned the passage that carries it."""
    ans = " ".join((answer or "").split())
    if not ans:
        return ""
    return _pick(_ANSWER, task.task_id, len(task.oracle_plan)).format(ans=ans)
