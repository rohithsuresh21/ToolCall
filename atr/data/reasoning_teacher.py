"""Teacher-written reasoning for the blocks where the template has nothing to say.

WHY ONLY SOME BLOCKS. atr/data/reasoning.py writes every <think> from the known
chain, so every block is correct and none of them REASONS: "The result names
Vestorland. Now I need the official language of Vestorland" asserts the carry
without ever saying why that name and not one of the six others on screen. A
larger model writing the block can weigh the candidates -- but only where there
are candidates to weigh, and that is a minority of blocks, measured rather than
assumed.

THE SELECTOR, and why the obvious one is useless. Counting capitalised names on
screen that the prompt does not contain gives a median of SEVEN per hop and >=5 on
98.2% of them: by that reading essentially every block is ambiguous and there is
nothing to select. The discriminator that actually separates is the DISTRACTOR,
not the candidate count. `search` boosts a title match 1.6x, so the passage that
reveals the next entity is rank 1 in 100.0% of hop blocks (99.8% terminal) -- the
carry is mechanically "read the top hit", which is exactly why the template is
adequate most of the time. What makes it inadequate is a SIBLING OF THE SAME KIND
sitting alongside it: `Quasar Maritime` next to `Vanguard Maritime`, where naming
the right one is a judgement and the template just asserts it. Measured on
sft_r0_max: 48.2% of hop blocks and 31.8% of terminal blocks, 25.2% of all blocks.

`_kind` reads the doc_id prefix rather than re-deriving the entity kind, because
the prefix IS the kind by construction in world.py (c/city/f/o/p/w) and the
tool_response already carries it. Note `city` must be tested before `c`.

THE INVARIANT IS ENFORCED BY THE PROMPT, NOT BY THE TEACHER'S GOODWILL. A block
may name only what the episode has already seen, so the teacher is shown the
episode PREFIX ONLY -- the question plus the passages strictly earlier calls
returned -- and never the chain, the gold answer or any later hop. It cannot name
chain[k+1] because it has not been told chain[k+1]. Handing it the whole record in
one call would be ~3x cheaper in input tokens and would invite precisely the
defect the invariant forbids; `reasoning.py`'s own docstring is the argument for
paying the 3x.

The gate is then belt-and-braces on top: `teacher_caps` (strict, no
sentence-initial exemption) plus `unseen_numbers`, both in atr/tasks/schema.py and
both scoring 0 on all 43,512 shipped blocks, so a non-zero count here is the
teacher and never the harness. A block that fails twice keeps its template; that
fallback is counted and printed, because an arm that silently degraded to its base
is the failure mode `12_build_recovery.py` was bitten by.

DETERMINISM MOVES INTO THE CACHE. reasoning.py picks phrasing with
sha1(task_id|step) precisely so a rebuild is byte-identical, and no LLM can hold
that contract. So the mint is offline and the CACHE is the reproducibility, keyed
"task_id|step" exactly as atr/data/naturalize.py keys passages "seed:doc_id":
minted once, committed, and loaded by the build. The same staleness hazard comes
with it -- a cache minted against one base loads silently against another, since
the keys still resolve -- which is why `cache_is_for()` records the base's own
fingerprint and the builder refuses a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..tasks.schema import norm_text, teacher_caps, unseen_numbers

THINK = re.compile(r"<think>(.*?)</think>", re.S)
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

# doc_id prefix -> entity kind. Longest first: "city1" must not read as "c".
_KIND_PREFIXES = ("city", "c", "f", "o", "p", "w")

MAX_WORDS = 70


@dataclass
class TeacherConfig:
    """Everything the arm is allowed to vary. Defaults are the (b) pilot."""
    model: str = "claude-opus-5"
    effort: str = "low"
    max_tokens: int = 2000
    max_attempts: int = 2          # one mint + one repair, then keep the template
    min_distractors: int = 1       # same-kind siblings needed to select a block


def _kind(doc_id: str) -> str:
    d = (doc_id or "").strip()
    for p in _KIND_PREFIXES:
        if d.startswith(p):
            return p
    return "?"


def hits(tool_content: str) -> list[dict]:
    """The parsed passages of a tool_response, or [] if it is not one.

    Reads the structured payload for the same reason tests/audit_sft.py does:
    the raw string also carries doc_ids and the BM25 score float, and norm_text
    deletes punctuation, so substring work on it false-positives."""
    try:
        payload = json.loads(tool_content)
        out = payload["results"]
    except (ValueError, KeyError, TypeError):
        return []
    return [h for h in out if isinstance(h, dict)]


def distractors(tool_content: str) -> int:
    """Same-kind siblings alongside the rank-1 passage of this result.

    Rank 1 is the passage that reveals the next entity in 100.0% of hop blocks,
    so its kind is the kind the next hop is choosing within."""
    hs = hits(tool_content)
    if len(hs) < 2:
        return 0
    k0 = _kind(hs[0].get("doc_id"))
    return sum(1 for h in hs[1:] if _kind(h.get("doc_id")) == k0)


def walk(messages: list[dict]):
    """Yield one entry per assistant turn that carries a <think>.

    Each is (msg_index, step, think, query, prev_tool, seen_norm) where `step` is
    the 0-based index of the call this block precedes (None on the answering
    turn), `prev_tool` is the tool_response immediately before it, and `seen_norm`
    is norm_text of the question plus every tool_response STRICTLY EARLIER --
    the same accumulator tests/audit_sft.py builds for axis 5, so a block this
    module accepts is a block the auditor accepts."""
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    seen = norm_text(question)
    prev_tool = None
    step = 0
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            # Folded in AFTER the assistant turn it answers, so a block never
            # counts its own call's result as seen.
            seen += " " + " ".join(
                norm_text(f"{h.get('title', '')} {h.get('text', '')}")
                for h in hits(m["content"]))
            prev_tool = m["content"]
            continue
        if m["role"] != "assistant":
            continue
        tm = THINK.search(m.get("content", ""))
        if tm is None:
            continue
        cm = CALL.search(m.get("content", ""))
        query = ""
        if cm is not None:
            try:
                query = json.loads(cm.group(1)).get("arguments", {}).get("query", "")
            except ValueError:
                query = ""
        yield (i, step if cm is not None else None, tm.group(1), query, prev_tool, seen)
        if cm is not None:
            step += 1


def select(messages: list[dict], cfg: TeacherConfig = TeacherConfig()) -> list[dict]:
    """The blocks a teacher should rewrite, in message order.

    Three exclusions, each for its own reason. Step 0 has no earlier result, so
    there is no carry to weigh and the head entity is in the question -- that is
    also audit axis 1's territory and not somewhere to put generated text. The
    ANSWERING block names the answer, which the terminal read just returned, so
    there is nothing to deliberate. Everything else is selected only on the
    distractor count."""
    out = []
    for i, step, think, query, prev_tool, seen in walk(messages):
        if step is None or step == 0 or prev_tool is None:
            continue
        if distractors(prev_tool) < cfg.min_distractors:
            continue
        out.append({"msg_index": i, "step": step, "template": think,
                    "query": query, "seen_norm": seen})
    return out


# --- the request -----------------------------------------------------------

SYSTEM = """\
You are writing the private reasoning of a research agent that answers multi-hop \
questions by issuing BM25 `search` calls over an encyclopedia.

You will be shown a question and the passages the agent's EARLIER searches \
returned. You will then be shown the query the agent is about to issue. Write the \
short reasoning that precedes that query.

Good reasoning here does one thing the agent cannot already do: it says WHY the \
entity being carried forward is the right one, when the search results put more \
than one candidate of the same kind on the table. Name the competing candidate \
and say what rules it out.

HARD CONSTRAINT. You may name ONLY things that appear in the question or in the \
passages shown to you. You have not been shown what the next search returns, and \
you must not guess it. Never invent a name, a date, a number, a place or a fact \
that is not on the page in front of you -- not as a guess, not as an example, not \
hedged. If you are unsure whether something was shown to you, leave it out.

Write 1-3 sentences, at most %d words, first person, plain prose. No markdown, no \
quotes, no tags, no preamble. Output only the reasoning itself.""" % MAX_WORDS


def render_prefix(messages: list[dict], upto_msg_index: int) -> str:
    """The episode as the teacher sees it: question, then each earlier search and
    the passages it returned. Prose rather than raw JSON payloads -- the doc_id
    and score carry no information the teacher should reason from, and they are
    exactly the fields that cause spurious substring matches elsewhere."""
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    parts = [f"QUESTION\n{question.strip()}\n"]
    n = 0
    pending = None
    for m in messages[:upto_msg_index]:
        if m["role"] == "assistant":
            cm = CALL.search(m.get("content", ""))
            if cm is not None:
                try:
                    pending = json.loads(cm.group(1)).get("arguments", {}).get("query", "")
                except ValueError:
                    pending = ""
        elif m["role"] == "tool":
            n += 1
            parts.append(f"SEARCH {n}: {pending or ''}")
            for h in hits(m["content"]):
                parts.append(f"  - {h.get('title', '')}: {h.get('text', '')}")
            parts.append("")
            pending = None
    return "\n".join(parts)


def user_message(messages: list[dict], block: dict) -> str:
    return (render_prefix(messages, block["msg_index"])
            + f"\nThe agent is about to issue SEARCH {block['step'] + 1}: "
              f"{block['query']}\n\nWrite the reasoning that precedes it.")


def custom_id(task_id: str, step: int) -> str:
    """Batch custom_id. The cache key is `task_id|step`; `|` is not allowed in a
    custom_id, so the id carries a sha1 and the mapping is kept alongside."""
    return "b" + hashlib.sha1(f"{task_id}|{step}".encode()).hexdigest()[:30]


def cache_key(task_id: str, step: int) -> str:
    return f"{task_id}|{step}"


# --- the gate --------------------------------------------------------------

def violations(text: str, question: str, seen_norm: str) -> list[str]:
    """Why this teacher block is not usable, or [] if it is.

    Ordered most-diagnostic first so a single label can be counted per block.
    `teacher_caps` reads at prose=False: a teacher opens a sentence with an
    invented name on its first try, and prose=True is structurally blind to that
    slot."""
    t = " ".join((text or "").split())
    if not t:
        return ["empty"]
    if len(t.split()) > MAX_WORDS + 10:
        return ["too_long"]
    if "<" in t or ">" in t:
        return ["markup"]
    bad = [c for c in teacher_caps(t, question)
           if norm_text(c) and norm_text(c) not in seen_norm]
    if bad:
        return ["psychic_name:" + ",".join(sorted(set(bad))[:4])]
    nums = unseen_numbers(t, seen_norm)
    if nums:
        return ["psychic_number:" + ",".join(sorted(set(nums))[:4])]
    return []


def repair_hint(reasons: list[str]) -> str:
    """The second-attempt instruction. Names the offending token so the retry is
    informed rather than a re-roll of the same dice."""
    r = reasons[0] if reasons else ""
    if r.startswith("psychic_name:"):
        return ("Your previous attempt named %s, which does not appear in the "
                "question or in any passage shown to you. Rewrite it using only "
                "names that do appear there." % r.split(":", 1)[1])
    if r.startswith("psychic_number:"):
        return ("Your previous attempt stated the number %s, which does not appear "
                "in the question or in any passage shown to you. Rewrite it "
                "without that number." % r.split(":", 1)[1])
    return {"empty": "Your previous attempt was empty. Write the reasoning.",
            "too_long": "Your previous attempt was too long. Use at most %d words."
                        % MAX_WORDS,
            "markup": "Your previous attempt contained tags. Write plain prose only.",
            }.get(r, "Rewrite it.")


# --- the cache -------------------------------------------------------------

@dataclass
class Cache:
    """Minted teacher blocks, keyed task_id|step. This file IS the determinism."""
    path: Path
    base_fingerprint: str = ""
    model: str = ""
    blocks: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Cache":
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(path=p, base_fingerprint=d.get("base_fingerprint", ""),
                   model=d.get("model", ""), blocks=d.get("blocks", {}))

    def save(self) -> None:
        self.path.write_text(json.dumps(
            {"base_fingerprint": self.base_fingerprint, "model": self.model,
             "blocks": dict(sorted(self.blocks.items()))},
            ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")

    def get(self, task_id: str, step: int):
        return self.blocks.get(cache_key(task_id, step))

    def put(self, task_id: str, step: int, think: str, attempts: int) -> None:
        self.blocks[cache_key(task_id, step)] = {"think": think, "attempts": attempts}


def rebuild_ok(task_id: str, difficulty: int, prompt: str) -> bool:
    """Whether the record's Task reconstructs from its own `task_id`.

    The same walk as `12_build_recovery._rebuild`, kept here because three
    scripts now need it and a fourth copy would be a fourth thing to keep in
    sync. `_hop_or_shorter` can degrade a longer request to a shorter chain and
    consumes from the generator's rng on the way, so the requested hop count is
    not always the recorded difficulty -- try every one that could have produced
    this record and accept the one whose prompt matches exactly.

    Imported lazily: this module is otherwise the stdlib-only decision half of
    the arm, and nothing that only gates text should pay for the world generator.
    Run it inside `legacy_vocab()` for any base built before commit 566f4d2."""
    import random

    from ..tasks.generator import _hop_or_shorter
    from ..tools.world import build_world

    seed = int(task_id.split("-")[1])
    w = build_world(seed)
    for h in range(difficulty, 5):
        t = _hop_or_shorter(random.Random(seed * 7919 + 13), w, seed, h)
        if t.task_id == task_id and t.difficulty == difficulty and t.prompt == prompt:
            return True
    return False


def fingerprint(path) -> str:
    """sha1 of the base set. A cache minted against a different base would load
    silently -- the keys are task ids and they still resolve -- and quietly paste
    another population's reasoning into this one."""
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def apply_block(content: str, think: str) -> str:
    """Swap the <think> body of one assistant turn, touching nothing else."""
    return THINK.sub(lambda _: f"<think>{think}</think>", content, count=1)
