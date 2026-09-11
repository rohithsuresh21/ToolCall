"""
Turn real MuSiQue train rows into EXECUTABLE, LEAK-FREE oracle chains.

The problem this solves. `data/musique_train_tasks.jsonl` carries an `oracle_plan`
copied straight from MuSiQue's `question_decomposition`, and that decomposition is
a sketch, not a plan: 59.3% of its sub-questions still contain an unresolved `#N`
placeholder ("Who besides the british colonized #1 ?"), where `#N` means "the
answer of hop N". `#1` is not a BM25 query. Replayed as-is, only 20.7% of the
3975 rows execute at all -- which is why CLAUDE.md records real sufficiency as a
CORPUS property (100% of rows contain their gold answer in their own 20 passages)
and explicitly not as a plan property.

Resolving those placeholders is what turns the corpus property into trajectories
we can actually train on. The resolution has to be done from what an EPISODE can
see, or the resulting plan is psychic and teaches exactly the defect we are trying
to remove. So `#N` is resolved to a capitalised span drawn from hop N's OWN top-k
hits -- passages the episode has already been shown -- and every candidate must
clear four checks before it is accepted:

  occurs      the substituted name appears verbatim in the passage this hop then
              retrieves. A substitution that merely happens to rank the right
              document is a lucky query; training on it teaches the lucky string
              rather than the reasoning step.
  entity-like a degenerate capitalised span ("The", "It", "However") is a sentence
              opener the regex picked up. BM25 ignores it, so it retrieves fine and
              has to be rejected HERE rather than by the retrieval gate.
  not the subject
              `#N` is the ANSWER of hop N, so it is new information. A candidate
              that just restates hop N's own query is the thing hop N was asking
              ABOUT, not what it found. This rejects "What year did WOSF voters
              once again vote for a Barack Obama?", where `#1` resolved to WOSF --
              a term hop 1 asked FROM. `ChainConfig.subject_rule` selects how hard
              to swing: "subset" (default, 2055 survivors before the other gates)
              or "overlap" (2034). The stricter rule costs only 21 records, which
              is worth knowing but is not why it is off by default -- MuSiQue nests
              place names, so "overlap" also rejects correct answers that reuse a
              word from the question ("Yuma" -> "Yuma County").
  unicode-folded
              the subject test compares tokens, and the tokeniser this repo's BM25
              uses (`[a-z0-9]+`) silently truncates accented names: "Beyonce"
              tokenises to {beyonce} but the acute-accented spelling tokenises to
              {beyonc}, so the two never compare equal and the subject test misses.
              Folding to NFKD-stripped ASCII first makes them compare. MEASURED: on
              this corpus it changes 0 of 7524 substitutions -- it is insurance
              against a name distribution, not a fix for one. Keep it anyway; the
              cost is one normalise call and the failure it prevents is silent.

Then the chain itself has to be worth training on:

  executable  every hop retrieves at least one NEW supporting passage, and the gold
              answer is in the FINAL hop's results.
  leak-free   and the gold answer is in NO earlier hop's results. This is the same
              axis as generator._is_prefix_leaky: BM25 returns whole passages, so
              the answer routinely turns up before the chain has been walked, the
              chain is truncatable, a model that stops early is still scored right,
              and that is what it learns.
  every query writable
              no hop's query may name a proper noun the episode has not been shown.
              This is axis 1 of tests/audit_sft.py stated as a build-time filter
              instead of a post-hoc complaint -- MuSiQue's own decomposition
              sometimes opens on an entity the question never names, and such a row
              is unlearnable for the same reason a psychic synthetic plan was. Both
              sides call `schema.psychic_caps`, ONE definition, so a set cannot be
              built against one rule and judged against another.

              It is TWO gates, not one, because they reject different populations.
              Hop 1 (`require_writable_first_query`) is checked against the question
              alone: it catches a decomposition whose entry point is an entity the
              asker never supplied. Hops 2+ (`require_writable_later_queries`) are
              checked against the question PLUS the prose of every strictly earlier
              hit: they catch a sub-question that assumes a name retrieval has not
              surfaced ("Where are the villages of Wengen and Zermatt located?",
              where the chain reveals Zermatt and nothing reveals Wengen). Only the
              first gate existed until 2026-09-11, and 3 of 325 records in the first
              combined smoke build tripped the auditor's <think> axis because of it
              -- the synthesised reasoning quotes the resolved sub-question, so a
              psychic query surfaces as psychic deliberation.

MEASURED YIELD at top_k=3, defaults: 1935 of 3975 rows (48.7%), split 1132 / 521 /
282 across 2/3/4 hops. Rejections, per axis: 1310 substitution-not-in-evidence, 451
no-new-evidence, 119 gold-leaks-early, 115 psychic-query-hop2+, 45 psychic-first-query.
(Before the hop-2+ gate: 2009 rows / 50.5%, split 1138 / 536 / 335. It costs 74 rows,
53 of them 4-hop -- the thinnest family paying the most, because a longer chain has
more hops that can assume a name.)

The 4-hop tail is thin by construction -- the gates that make a 4-hop chain
trainable are the same ones a 4-hop chain is most likely to fail -- and that
scarcity is a fact about MuSiQue, not a tuning knob.

That psychic count is 46 and not 404 because `psychic_caps` excludes capitalised
FUNCTION words. Synthetic oracle queries are keyword fragments ("Meridian City
capital") that never open on one; real sub-questions are whole sentences ("What
country was Signmark from?"). Counting their leading "What" as an invented entity
rejected 404 of 2055 otherwise-clean chains -- 20% of the yield -- for a word that
names nothing. Same trap `_oracle_entity` fell into by anchoring at `^`.

The BM25 index here is a faster re-implementation of `atr/tools/builtin.py::_search`
(per-doc token counts precomputed instead of re-tokenised inside the term loop),
because resolution tries up to 400 candidate substitutions per hop and the shipped
one re-tokenises every document for every query term. `verify_index()` asserts the
two rank identically against the real registry, and tests/test_real_chains.py runs
it -- without that, every number above is a statement about a re-implementation.
"""
from __future__ import annotations

import collections
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..tasks.schema import Task, norm_text, psychic_caps

_K1, _B = 1.5, 0.75
_TOK = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"#(\d+)")
# Entity names run 1-5 words; the optional lowercase joiners are what keeps
# "University of California" and "Ludwig van Beethoven" in one span.
_CAP = re.compile(
    r"([A-Z][A-Za-z'’-]*(?:\s+(?:of\s+|the\s+|de\s+|van\s+|von\s+|and\s+)?[A-Z][A-Za-z'’-]*){0,4})")

# Function words that make a capitalised span a sentence opener rather than a name.
_FUNC = {
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "for", "to", "by",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "his",
    "her", "their", "there", "then", "when", "what", "who", "which", "as",
    "is", "was", "were", "be", "been", "from", "with", "after", "before",
    "during", "however", "although", "while", "but", "also", "such", "other",
}

MAX_COMBOS = 400
MAX_CANDIDATES = 40


def fold(s: str) -> str:
    """NFKD-decompose and drop combining marks, so an accented name and its plain
    spelling produce the same tokens. See the module docstring: this exists so the
    subject-overlap test cannot be defeated by a diacritic."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


def content_tokens(text: str) -> set[str]:
    """Tokens that carry topic. Folded, function words dropped, 1-2 char tokens
    dropped -- those match everything and would reject every candidate."""
    return {t for t in tokens(fold(text)) if t not in _FUNC and len(t) > 2}


def is_entity_like(name: str) -> bool:
    return any(t not in _FUNC and len(t) > 2 for t in tokens(fold(name)))


class Index:
    """BM25 over one task's candidate set. Ranks identically to
    builtin._search -- asserted by verify_index(), not assumed."""

    def __init__(self, docs: Sequence[dict]):
        self.docs = list(docs)
        self.n = len(self.docs)
        self.tf: list[collections.Counter] = []
        self.lens: list[int] = []
        self.title_toks: list[set] = []
        postings: dict[str, set] = {}
        for i, d in enumerate(self.docs):
            toks = tokens(d["title"] + " " + d["text"])
            c = collections.Counter(toks)
            self.tf.append(c)
            self.lens.append(len(toks))
            self.title_toks.append(set(tokens(d["title"])))
            for t in c:
                postings.setdefault(t, set()).add(i)
        self.postings = postings
        self.avg = sum(self.lens) / max(self.n, 1)
        self.blob = [norm_text(d["title"] + " " + d["text"]) for d in self.docs]

    def search(self, query: str, top_k: int = 3) -> list[int]:
        q = tokens(query)
        if not q:
            return []
        scores: list[tuple[float, int]] = []
        for i in range(self.n):
            sc = 0.0
            for t in q:
                tf = self.tf[i].get(t, 0)
                if tf <= 0:
                    continue
                df = len(self.postings.get(t, ()))
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                tfw = tf * (_K1 + 1) / (
                    tf + _K1 * (1 - _B + _B * self.lens[i] / max(self.avg, 1e-9)))
                sc += idf * tfw
                # builtin._search scales the ACCUMULATED score, not this term's
                # contribution, so the title boost compounds and is sensitive to
                # query token order. Reproduced exactly -- that is the behaviour,
                # whatever the comment there says it intends.
                if t in self.title_toks[i]:
                    sc *= 1.6
            if sc > 0:
                scores.append((sc, i))
        scores.sort(key=lambda x: (-x[0], self.docs[x[1]]["doc_id"]))
        return [i for _, i in scores[:max(1, min(int(top_k), 10))]]


def tidy_query(q: str) -> str:
    """Close up MuSiQue's tokeniser spacing ("Antarctica 's border", "colonized #1 ?").

    Cosmetic for retrieval and NOT cosmetic for training: the resolved query is the
    exact string the model is taught to emit, and " 's" is an artefact of MuSiQue's
    preprocessing rather than anything a person would type. The guard is the point --
    BM25 here tokenises on `[a-z0-9]+`, so this must not change a single token, and
    if it ever does the original is kept rather than silently re-ranking a hop."""
    out = re.sub(r"\s+([?.,!;:])", r"\1", re.sub(r"\s+(['’]s\b)", r"\1", q or ""))
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out if tokens(out) == tokens(q) else q


def _spans(text: str, cap: int = 24) -> list[str]:
    out, seen = [], set()
    for m in _CAP.finditer(text):
        s = m.group(1).strip(" .,-")
        if len(s) < 3 or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= cap:
            break
    return out


@dataclass
class ChainConfig:
    top_k: int = 3
    require_leak_free: bool = True
    require_writable_first_query: bool = True
    # Same rule as require_writable_first_query, applied to hops 2..L against the
    # prompt PLUS every strictly-earlier hop's retrieved text. The first-query gate
    # alone let 3 of 325 smoke records through: MuSiQue's own sub-question for a
    # later hop can open on a proper noun ("Wengen", "All Saints Church") that the
    # question never names and no earlier passage returned, so the resolved query
    # is writable only by someone who already read the answer key. Kept as its own
    # flag with its own rejection key because it rejects a DIFFERENT population
    # from hop 1's: hop 1 fails on the decomposition's entry point, hops 2+ fail on
    # retrieval not having surfaced a name the sub-question assumes.
    require_writable_later_queries: bool = True
    require_full_coverage: bool = False   # every supporting passage retrieved
    # How hard to reject a substitution that restates hop N's own subject.
    #   "subset"  -- reject only when EVERY token of the candidate is already in
    #                hop N's query. Measured yield 2051/3975.
    #   "overlap" -- reject on ANY shared content token. Sounds stricter and is,
    #                but MuSiQue nests place names ("Yuma" -> "Yuma County",
    #                "Arizona" -> "Arizona State University"), so it throws away
    #                correct answers that legitimately reuse a word from the
    #                question. Measured cost: see tests/test_real_chains.py.
    subject_rule: str = "subset"


@dataclass
class ChainResult:
    task_id: str
    hops: int
    ok: bool = False
    reason: str = ""
    queries: list[str] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    first_gold_hop: int | None = None


def _resolve(row: dict, cfg: ChainConfig) -> ChainResult:
    docs = row["documents"]
    idx = Index(docs)
    by_id = {d["doc_id"]: i for i, d in enumerate(docs)}
    unused = {by_id[s] for s in row["route"] if s in by_id}
    gold = norm_text(row["gold"]["value"])
    plan = [(s["arguments"] or {}).get("query", "") for s in row["oracle_plan"]]
    res = ChainResult(task_id=row["task_id"], hops=len(plan))
    if not plan:
        res.reason = "empty_plan"
        return res

    env: dict[int, str] = {}
    hop_hits: list[list[int]] = []
    # What the EPISODE has seen before the current hop: the question, then the
    # prose of every strictly earlier hit. psychic_caps lowercases and substring-
    # matches, and idx.blob is norm_text of `title + " " + text` -- the same two
    # pieces tests/audit_sft.py concatenates for its <think> axis, so the gate and
    # the auditor read the same context rather than two dialects of it.
    seen_ctx = row["prompt"]
    # chain[k] = the name call k-1 revealed AND call k substitutes. Only filled
    # when call k references the IMMEDIATELY preceding hop, so "the result names X"
    # is literally true rather than approximately true; see reasoning._reveals.
    chain = [""] * len(plan)

    for k, raw in enumerate(plan):
        refs = sorted({int(n) for n in _PLACEHOLDER.findall(raw)})
        needed = [n for n in refs if n not in env]
        cands: dict[int, list[str]] = {}
        for n in needed:
            src = hop_hits[n - 1] if 0 < n <= len(hop_hits) else []
            own_q = plan[n - 1] if 0 < n <= len(plan) else ""
            own_all = set(tokens(fold(own_q)))
            own_content = content_tokens(own_q)
            pool = [docs[i]["title"] for i in src]
            for i in src:
                pool += _spans(docs[i]["text"])
            seen, ded = set(), []
            for p in pool:
                if p.lower() in seen:
                    continue
                seen.add(p.lower())
                if not is_entity_like(p):
                    continue
                # `#N` is what hop N FOUND, so a candidate that just restates hop
                # N's own query is the subject it asked about, not the answer.
                if cfg.subject_rule == "overlap":
                    if content_tokens(p) & own_content:
                        continue
                elif set(tokens(fold(p))).issubset(own_all):
                    continue
                ded.append(p)
            cands[n] = ded[:MAX_CANDIDATES]
        if any(not v for v in cands.values()):
            res.reason = f"unresolvable_placeholder_hop{k + 1}"
            return res

        combos: list[dict[int, str]] = [{}]
        for n in needed:
            combos = [{**c, n: v} for c in combos for v in cands[n]]
            if len(combos) > MAX_COMBOS:
                combos = combos[:MAX_COMBOS]
        best = None
        for c in combos:
            e = dict(env)
            e.update(c)
            qq = _PLACEHOLDER.sub(lambda m: e.get(int(m.group(1)), m.group(0)), raw)
            hh = idx.search(qq, cfg.top_k)
            new = [h for h in hh if h in unused]
            rank = hh.index(new[0]) if new else 99
            # the substituted name must OCCUR in the evidence this hop pulls
            occurs = bool(new) and all(
                norm_text(v) and norm_text(v) in idx.blob[new[0]] for v in c.values())
            score = (0 if occurs else 1, rank, -len(new))
            if best is None or score < best[0]:
                best = (score, c, qq, hh)
            if score[:2] == (0, 0):
                break
        score, chosen, query, hits = best
        if chosen and score[0] != 0:
            res.reason = f"substitution_not_in_evidence_hop{k + 1}"
            return res
        env.update(chosen)
        if k > 0 and k in chosen:
            chain[k] = chosen[k]
        elif k > 0 and k in env and k in refs:
            chain[k] = env[k]

        if "#" in query:
            res.reason = f"unresolved_hop{k + 1}"
            return res
        new = [h for h in hits if h in unused]
        if not new:
            res.reason = f"no_new_evidence_hop{k + 1}"
            return res
        tq = tidy_query(query)
        # Hop 1 is gated once at the end, against the prompt alone; see
        # require_writable_first_query. A row that fails both is counted here,
        # because this is the check that fires first.
        if cfg.require_writable_later_queries and k > 0 and psychic_caps(tq, seen_ctx):
            res.reason = f"psychic_query_hop{k + 1}"
            return res
        for h in new:
            unused.discard(h)
        res.queries.append(tq)
        hop_hits.append(hits)
        seen_ctx += " " + " ".join(idx.blob[h] for h in hits)

    if cfg.require_full_coverage and unused:
        res.reason = "support_not_covered"
        return res

    for i, hits in enumerate(hop_hits, start=1):
        if gold and any(gold in idx.blob[h] for h in hits):
            res.first_gold_hop = i
            break
    if res.first_gold_hop is None:
        res.reason = "gold_never_retrieved"
        return res
    if res.first_gold_hop != len(plan):
        res.reason = ("gold_leaks_at_hop%d" % res.first_gold_hop
                      if cfg.require_leak_free else "")
        if cfg.require_leak_free:
            return res

    if cfg.require_writable_first_query and psychic_caps(res.queries[0], row["prompt"]):
        res.reason = "psychic_first_query"
        return res

    res.chain = chain
    res.ok = True
    return res


def build_chain_tasks(rows: Sequence[dict], cfg: ChainConfig | None = None
                      ) -> tuple[list[dict], dict]:
    """rows -> (task dicts with a RESOLVED oracle_plan and a `chain`, stats).

    Rejection counts are kept per reason rather than as one total, for the same
    reason generator._SHORTCUT_STATS is: a single number cannot tell an
    unresolvable placeholder from a leaky chain, and the two call for opposite
    responses."""
    cfg = cfg or ChainConfig()
    kept: list[dict] = []
    stats: collections.Counter = collections.Counter()
    for row in rows:
        r = _resolve(row, cfg)
        stats["seen"] += 1
        if not r.ok:
            stats["reject:" + re.sub(r"\d+$", "N", r.reason)] += 1
            continue
        out = dict(row)
        out["oracle_plan"] = [{"name": "search", "arguments": {"query": q}}
                              for q in r.queries]
        out["chain"] = r.chain
        out["notes"] = (row.get("notes", "") +
                        "; resolved chain, leak-free, gold first retrieved at hop "
                        f"{r.first_gold_hop}/{r.hops}")
        kept.append(out)
        stats["kept"] += 1
        stats[f"kept:{r.hops}hop"] += 1
        stats[f"substitutions:{sum(1 for c in r.chain if c)}"] += 1
    return kept, dict(sorted(stats.items()))


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def write_tasks(tasks: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for t in sorted(tasks, key=lambda t: t["task_id"]):
            f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_chain_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(d) for d in load_rows(path)]


def verify_index(rows: Sequence[dict], n: int = 60, seed: int = 7) -> tuple[int, int]:
    """Assert the fast Index ranks identically to the shipped BM25. Returns
    (matching, checked)."""
    import random
    from types import SimpleNamespace

    from ..tools.adapter import get_registry
    from ..tools.world import world_for_task

    reg = get_registry("builtin")
    rng = random.Random(seed)
    good = checked = 0
    for row in rng.sample(list(rows), min(n, len(rows))):
        w = world_for_task(SimpleNamespace(documents=row["documents"], seed=row["seed"]))
        idx = Index(row["documents"])
        for s in row["oracle_plan"]:
            q = (s["arguments"] or {}).get("query", "")
            if not tokens(q):
                continue
            want = [r["doc_id"] for r in reg.call(w, "search", {"query": q, "top_k": 3})
                    .get("results", [])]
            mine = [row["documents"][i]["doc_id"] for i in idx.search(q, 3)]
            checked += 1
            good += int(want == mine)
    return good, checked
