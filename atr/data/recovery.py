"""
STeCa-style RECOVERY trajectories: a step that goes wrong, on purpose, and is
then repaired.

WHY THIS EXISTS. `ScoreCard.recovery_ok` and `reward.w_recovery` have been in
this repo since the first commit with ZERO data behind them -- `eval --backend
oracle` prints "recovered after error  --" because an oracle replay never makes
a bad call, so the axis has never been measured and the reward term has never
fired. The direct evidence that it matters: of 18 failed 4-hop judge tasks, 4
hit `max_steps` carrying an EMPTY answer. Those are not wrong answers, they are
loops -- the policy re-queries, gets the same three documents back, and re-queries
again until the step budget runs out. Nothing in the pipeline produces a training
signal for the one move that breaks a loop: notice that a retrieval came back
useless, and change approach rather than rephrase around the same entity.

CONSTRUCTION. No teacher model is needed, because the generator already knows the
correct query. Take a chain that has passed all four sufficiency/leak gates, pick
one hop, replace its query with a plausible BAD one, EXECUTE that against the same
BM25 tool the agent uses, emit a <think> block that acknowledges the miss and
names a different approach, then continue with the correct query to the gold
answer. The failure is therefore genuine: the empty or irrelevant tool_response in
the record is the one the real tool actually returned for that query.

THE MISS HAS TO BE REAL, and in this world that is a stronger constraint than it
sounds. `search` boosts a title match 1.6x, so ANY query containing the source
entity's name retrieves the source entity's own passage -- which is precisely the
passage that names the next hop. A "wrong keyword on the right entity" therefore
retrieves exactly what the hop needed and misses nothing; splicing a recovery onto
it would teach the model to re-query when it ALREADY HAD the answer, which is the
looping behaviour this file exists to remove. So every candidate is executed and
kept only if:

  * no returned passage contains the gold answer (otherwise the record also trips
    audit axis 4: the answer becomes retrievable before the terminal read), and
  * no returned passage names the entity the next hop is written from (otherwise
    no ground was actually lost).

`_FLAVOURS` proposes; the tool disposes. The realised flavour mix is an OUTPUT of
the build, reported by `scripts/12_build_recovery.py`, not an input you can set.

THE RECOVERY <think> OBEYS THE SAME INVARIANT AS EVERY OTHER BLOCK: it may name
only what the episode has already seen. It is emitted AFTER the bad result, so
what it may name is the prompt, the passages the earlier correct hops returned,
and the bad hop's own (useless) result -- never the entity the repaired query is
about to reveal. Note that `tests/audit_sft.py` builds its "seen" text from
tool_responses only, so material the ASSISTANT invented (an absent entity name in
the bad query itself) is NOT seen: the templates below therefore never quote the
bad query back. Phrasing comes from sha1(task_id|step), never a live RNG, so a
rebuild is byte-identical -- the same rule as `reasoning._pick` and
`generator._variant_rng`. And as in reasoning.py, no template may open a sentence
(colons included) with a bare entity name, because `psychic_caps(prose=True)`
ignores sentence-initial capitals.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from ..tasks.generator import (_gold_as_answer, _LEAF_QUERY_VARIANTS, _REL,
                               _REL_QUERY_VARIANTS)
from ..tasks.schema import Task, norm_text
from ..tools.adapter import get_registry
from ..tools.world import (ORG_KIND, ORG_WORDS, PERSON_FIRST, PERSON_LAST,
                           World, build_world)
from .reasoning import _ATTR_PHRASE, _leaf_keyword

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)


# --- recovery <think> templates --------------------------------------------
# Two families, because the model has to read two different signals. An EMPTY
# result is unambiguous (nothing matched, so the query was wrong); a non-empty but
# irrelevant result is the harder and more common case -- three plausible-looking
# passages, none of them the one you need -- and it is the one the looping judge
# trajectories actually faced.
_EMPTY = [
    "That search came back with nothing at all, so those terms match no passage. "
    "Rather than rephrase around them I should go back to {prev} and ask for {goal}.",
    "Nothing matched that query. The terms were the problem, not the corpus, so the "
    "move is to anchor on {prev} again and search for {goal}.",
    "No passages came back for that. Let me change approach and use {prev} as the "
    "anchor, searching directly for {goal}.",
]
_IRRELEVANT = [
    "None of those passages is about {goal}; that query pulled the wrong entity "
    "entirely. Re-anchoring on {prev} and asking again.",
    "Those are the wrong passages -- none of them gives me {goal}. The query drifted "
    "off the chain, so I should anchor it back on {prev}.",
    "That returned passages I cannot use: none of them carries {goal}. Let me drop "
    "that phrasing and search {prev} for it directly.",
]

# Generic stand-ins for the over-conversational flavour, which deliberately drops
# the proper noun -- that is what makes it miss, and it is also the one thing that
# keeps it from being a psychic query.
_GENERIC = {"person": "this person", "organisation": "this company",
            "city": "this city", "country": "this country",
            "work": "this work", "feature": "this place"}


@dataclass
class RecoveryConfig:
    """`rate_by_hop` is the share of each hop family converted to a recovery
    trajectory. Weighted toward 3 and 4 hops because that is where the looping was
    measured: 4 of 18 failed 4-hop judge tasks hit max_steps with an empty answer,
    while 2-hop is already at 80.0% judge F1 with little left to repair."""
    rate_by_hop: dict[int, float] = field(
        default_factory=lambda: {2: 0.05, 3: 0.12, 4: 0.15})
    top_k: int = 3
    max_thinking_chars: int = 400


def _pick(templates: list[str], task_id: str, step: int) -> str:
    h = hashlib.sha1(f"recovery|{task_id}|{step}".encode()).digest()
    return templates[h[0] % len(templates)]


def _order(items: list, task_id: str, tag: str) -> list:
    """A deterministic permutation. Used for candidate ORDER (which hop, which
    flavour) so the realised mix varies across records without an RNG."""
    return sorted(items, key=lambda x: hashlib.sha1(
        f"{tag}|{task_id}|{x}".encode()).hexdigest())


def _hits(tool_content: str) -> list[dict]:
    try:
        return json.loads(tool_content).get("results", []) or []
    except (ValueError, AttributeError):
        return []


def _blob(hits: list[dict]) -> list[str]:
    return [norm_text(f"{h.get('title', '')} {h.get('text', '')}") for h in hits]


def _seen_entities(world: World, tool_contents: list[str]) -> list[dict]:
    """Entities whose passage was actually returned to the model, in order. The
    hit TITLE is the entity name in this world, so this is exact rather than a
    string search over prose."""
    out, seen = [], set()
    for c in tool_contents:
        for h in _hits(c):
            e = world.by_name(h.get("title", ""))
            if e and e["id"] not in seen:
                seen.add(e["id"])
                out.append(e)
    return out


# --- bad-query flavours ----------------------------------------------------
# Each returns candidate query strings, most plausible first. They are PROPOSALS:
# `_execute_miss` rejects any that does not actually lose ground.

def _flavour_wrong_entity(task, world, n, src, kw, seen) -> list[str]:
    """The most realistic failure by far, and the one the judge trajectories show:
    the model reads a passage, picks the WRONG capitalised name out of it, and
    queries that. Everything in the query was on screen, so it is a genuine
    mis-step rather than an invented one."""
    chain = list(task.chain or [])
    forbid = {norm_text(c) for c in chain[n:]}
    same_kind = [e for e in seen if e["kind"] == src["kind"]
                 and norm_text(e["name"]) not in forbid]
    other = [e for e in seen if e["kind"] != src["kind"]
             and norm_text(e["name"]) not in forbid]
    return [f"{e['name']} {kw}" for e in same_kind + other]


def _flavour_absent_entity(task, world, n, src, kw, seen) -> list[str]:
    """A name of the right SHAPE that this world does not contain. Restricted to
    person and organisation sources: those are the two kinds with a real name pool
    (1920 and 448 combinations against 10 and 8 actually minted), so an absent name
    is plausible. Countries and cities are a fixed set of six, and inventing one
    would not look like anything a model would write."""
    if src["kind"] == "person":
        pool = [f"{f} {l}" for f in PERSON_FIRST[:12] for l in PERSON_LAST[:12]]
    elif src["kind"] == "organisation":
        pool = [f"{w} {k}" for w in ORG_WORDS[:12] for k in ORG_KIND[:12]]
    else:
        return []
    absent = [nm for nm in pool if world.by_name(nm) is None]
    return [f"{nm} {kw}" for nm in _order(absent, task.task_id, f"absent{n}")[:8]]


def _flavour_conversational(task, world, n, src, kw, seen) -> list[str]:
    """Over-conversational, under-specified phrasing: a whole polite sentence that
    drops the proper noun for a generic stand-in. BM25 then scores it on function
    words, which is exactly why it misses -- and dropping the name is also what
    keeps this flavour from smuggling in an entity the episode never saw."""
    generic = _GENERIC.get(src["kind"], "this entity")
    route = list(task.route or [])
    if n < len(route):
        goal = _REL[route[n]][3].format(generic)
    else:
        goal = f"the {_ATTR_PHRASE.get(kw, kw)} of {generic}"
    return [f"Could you please tell me what {goal} is?",
            f"I would like to know {goal}, if you can tell me that.",
            f"Can you find out {goal} for me please?"]


def _flavour_wrong_keyword(task, world, n, src, kw, seen) -> list[str]:
    """The right entity with a keyword that does not apply to its kind. Kept in the
    table and almost never kept in the DATA: a query carrying the source name pulls
    the source passage on the 1.6x title boost whatever the keyword says, and that
    passage names the next hop -- so `_execute_miss` rejects it. That is a fact
    about this corpus worth leaving visible rather than a flavour worth deleting."""
    bad_kws = [v[0] for k, v in _REL_QUERY_VARIANTS.items()
               if _REL[k][0] != src["kind"] and _REL[k][4] != kw]
    return [f"{src['name']} {q}" for q in _order(bad_kws, task.task_id, f"kw{n}")]


_FLAVOURS = {
    "wrong_entity": _flavour_wrong_entity,
    "absent_entity": _flavour_absent_entity,
    "conversational": _flavour_conversational,
    "wrong_keyword": _flavour_wrong_keyword,
}


def _execute_miss(world: World, query: str, forbid_gold: str, forbid_next: str,
                  used: set[str], top_k: int) -> dict | None:
    """Run `query` for real and return its payload only if it GENUINELY missed.

    Two rejections, and both matter. A result carrying the gold answer would make
    the answer retrievable before the terminal read (audit axis 4) on a set whose
    whole point is that it is not. A result naming the entity the next hop is
    written from lost no ground at all -- splicing a recovery onto it teaches
    re-querying when the needed passage was already in hand, i.e. the loop."""
    if norm_text(query) in used:
        return None
    reg = get_registry("builtin")
    res = reg.call(world, "search", {"query": query, "top_k": top_k})
    if res.get("error"):
        return None
    texts = _blob(res.get("results", []) or [])
    if forbid_gold and any(forbid_gold in t for t in texts):
        return None
    if forbid_next and any(forbid_next in t for t in texts):
        return None
    return res


def _canonical_turn(thinking: str, query: str) -> str:
    """The same canonical assistant turn build_sft writes: optional <think>, then
    exactly one <tool_call>. Written here rather than imported so the recovery
    splice cannot drift from the exported form it is splicing into -- both are
    pinned by tests/test_recovery.py."""
    call = json.dumps({"name": "search", "arguments": {"query": query}},
                      ensure_ascii=False, separators=(", ", ": "))
    head = f"<think>{thinking}</think>\n" if thinking else ""
    return f"{head}<tool_call>{call}</tool_call>"


def build_recovery(task: Task, messages: list[dict], cfg: RecoveryConfig | None = None,
                   world: World | None = None) -> dict | None:
    """Splice one executed bad hop plus its repair into an exported record.

    Returns {"messages", "hop", "flavour", "num_results", "query"} or None when no
    hop of this task admits a genuine miss. It operates on the EXPORTED record
    rather than re-running the episode, so every untouched record in the set stays
    byte-identical and the recovery slice is the only difference between the two
    arms of the ablation.
    """
    cfg = cfg or RecoveryConfig()
    chain, route = list(task.chain or []), list(task.route or [])
    plan = list(task.oracle_plan or [])
    if not chain or len(plan) != len(route) + 1:
        return None
    world = world if world is not None else build_world(task.seed)

    calls = [i for i, m in enumerate(messages)
             if m["role"] == "assistant" and CALL.search(m["content"])]
    if len(calls) != len(plan):
        return None
    tool_msgs = [m["content"] for m in messages if m["role"] == "tool"]
    gold = norm_text(_gold_as_answer(task.gold))
    used = {norm_text((s.get("arguments") or {}).get("query", "")) for s in plan}

    # Hop 0 is excluded on purpose: a bad FIRST query is the psychic-first-query
    # defect (audit axis 1), and it is also the one hop with no earlier result to
    # recover onto. Every later hop is a candidate, the terminal read included.
    for n in _order(list(range(1, len(plan))), task.task_id, "hop"):
        src = world.by_name(chain[n])
        if src is None:
            continue
        if n < len(route):
            kw = _REL_QUERY_VARIANTS[route[n]][0]
            goal = _REL[route[n]][3].format(chain[n])
            nxt = norm_text(chain[n + 1])
        else:
            leaf_kw = _leaf_keyword(task)
            kw = _LEAF_QUERY_VARIANTS.get(leaf_kw, [leaf_kw])[0]
            goal = f"the {_ATTR_PHRASE.get(leaf_kw, leaf_kw)} of {chain[n]}"
            nxt = ""
        seen = _seen_entities(world, tool_msgs[:n])
        good = CALL.search(messages[calls[n]]["content"]).group(1)
        keep_think = THINK.search(messages[calls[n]]["content"])
        for fname in _order(list(_FLAVOURS), task.task_id, f"flav{n}"):
            for q in _FLAVOURS[fname](task, world, n, src, kw, seen):
                res = _execute_miss(world, q, gold, nxt, used, cfg.top_k)
                if res is None:
                    continue
                empty = res.get("num_results", 0) == 0
                think = _pick(_EMPTY if empty else _IRRELEVANT, task.task_id, n).format(
                    prev=chain[n], goal=goal)
                out = list(messages)
                out[calls[n]] = {"role": "assistant", "content": _canonical_turn(
                    keep_think.group(1) if keep_think else "", q)}
                out.insert(calls[n] + 1, {
                    "role": "tool", "name": "search",
                    "content": json.dumps(res, ensure_ascii=False, default=str)})
                out.insert(calls[n] + 2, {
                    "role": "assistant",
                    "content": f"<think>{think[:cfg.max_thinking_chars]}</think>\n"
                               f"<tool_call>{good}</tool_call>"})
                return {"messages": out, "hop": n, "flavour": fname,
                        "num_results": res.get("num_results", 0), "query": q}
    return None
