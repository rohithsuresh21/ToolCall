"""
Programmatic task generator for single-tool BM25 multi-hop retrieval.

The benchmark is MuSiQue-style: a multi-hop question over Wikipedia-style
candidate passages, answered by composing facts retrieved across passages with
one BM25 `search` tool. Every task is emitted together with (a) a gold answer
computed from the same World the tool reads, and (b) an `oracle_plan`: the
reference sequence of `search` calls. Nothing is hand-labelled and nothing is
frozen, so you can mint 50k training tasks and a held-out dev set from disjoint
seed ranges without any leakage.

The oracle_plan earns its keep three times over: it drives MockBackend for CPU
testing, it gives the verifier a strict argument target, and it is the fallback
teacher when you want SFT data without paying for a large model.

Difficulty = the number of *dependent* calls -- calls whose arguments cannot be
written down until an earlier call has returned (the number of hops / searches).
That is the axis small models actually fall off, so it is the axis the
curriculum sorts on.
"""
from __future__ import annotations

import random
from typing import Callable

from ..tools.world import World, build_world
from .schema import Task

# The only tool in the benchmark contract.
ALL_TOOLS = ["search"]

# ---------------------------------------------------------------------------
# no-tool bank: answerable from the model's own knowledge, and *not* present in
# the world, so a tool call cannot help. This is the abstention signal. It is
# domain-independent general knowledge (not logistics).
# ---------------------------------------------------------------------------
STATIC_NO_TOOL: list[tuple[str, dict]] = [
    ("What is the chemical symbol for gold?", {"kind": "text", "value": "Au"}),
    ("Which planet in our solar system is closest to the Sun?", {"kind": "text", "value": "Mercury"}),
    ("In what year did Apollo 11 land humans on the Moon?", {"kind": "numeric", "value": 1969, "tol": 0}),
    ("How many sides does a regular hexagon have?", {"kind": "numeric", "value": 6, "tol": 0}),
    ("What is the capital city of Japan?", {"kind": "text", "value": "Tokyo"}),
    ("Who wrote the play 'Romeo and Juliet'?", {"kind": "text", "value": "Shakespeare"}),
    ("What is the freezing point of water in degrees Celsius?", {"kind": "numeric", "value": 0, "tol": 0}),
    ("Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
     {"kind": "any_of", "value": ["carbon dioxide", "co2"]}),
    ("Name the largest ocean on Earth.", {"kind": "text", "value": "Pacific"}),
]

_CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Austria": "Vienna", "Sweden": "Stockholm", "Norway": "Oslo", "Denmark": "Copenhagen",
    "Finland": "Helsinki", "Poland": "Warsaw", "Greece": "Athens", "Ireland": "Dublin",
    "Japan": "Tokyo", "China": "Beijing", "South Korea": "Seoul", "India": "New Delhi",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
    "Australia": "Canberra", "New Zealand": "Wellington", "Egypt": "Cairo",
    "Kenya": "Nairobi", "Nigeria": "Abuja", "South Africa": "Pretoria",
    "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Chile": "Santiago",
    "Peru": "Lima", "Colombia": "Bogota", "Mexico": "Mexico City", "Canada": "Ottawa",
    "Turkey": "Ankara",
}
_ELEMENTS = {
    "hydrogen": "H", "helium": "He", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "sodium": "Na", "magnesium": "Mg", "aluminium": "Al", "silicon": "Si",
    "phosphorus": "P", "sulfur": "S", "chlorine": "Cl", "potassium": "K",
    "calcium": "Ca", "iron": "Fe", "copper": "Cu", "zinc": "Zn", "silver": "Ag",
    "tin": "Sn", "iodine": "I", "platinum": "Pt", "mercury": "Hg", "lead": "Pb",
}
_PLURALS = [("child", "children"), ("person", "people"), ("mouse", "mice"),
            ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"),
            ("man", "men"), ("woman", "women"), ("crisis", "crises"),
            ("phenomenon", "phenomena"), ("cactus", "cacti"), ("datum", "data")]
_PAST = [("go", "went"), ("eat", "ate"), ("see", "saw"), ("take", "took"),
         ("write", "wrote"), ("drive", "drove"), ("fly", "flew"), ("swim", "swam"),
         ("begin", "began"), ("drink", "drank"), ("sing", "sang"), ("give", "gave")]
_LETTER_COUNT = [("elephant", 8), ("knowledge", 9), ("rhythm", 6), ("library", 7),
                 ("machine", 7), ("network", 7), ("quality", 7), ("surface", 7),
                 ("system", 6), ("weather", 7), ("journey", 7), ("kingdom", 7),
                 ("language", 8), ("mountain", 8), ("notebook", 8), ("pattern", 7)]
_WORDS_FOR_SORT = ["lantern", "bridge", "copper", "meadow", "signal", "harbor",
                   "quartz", "velvet", "anchor", "pilot", "cobalt", "orchid",
                   "timber", "marble", "falcon", "ginger", "pepper", "willow"]


def _build_no_tool_bank() -> list[tuple[str, dict]]:
    bank = list(STATIC_NO_TOOL)
    for country, cap in sorted(_CAPITALS.items()):
        bank.append((f"What is the capital city of {country}?",
                     {"kind": "text", "value": cap}))
        bank.append((f"Which city is the capital of {country}?",
                     {"kind": "text", "value": cap}))
    for elem, sym in sorted(_ELEMENTS.items()):
        bank.append((f"What is the chemical symbol for {elem}?",
                     {"kind": "text", "value": sym}))
    for sing, plur in _PLURALS:
        bank.append((f"What is the plural of the noun '{sing}'?",
                     {"kind": "text", "value": plur}))
    for pres, past in _PAST:
        bank.append((f"What is the past tense of the verb 'to {pres}'?",
                     {"kind": "text", "value": past}))
    for word, n in _LETTER_COUNT:
        bank.append((f"How many letters are in the word '{word}'?",
                     {"kind": "numeric", "value": n, "tol": 0}))
    for i in range(60):
        a, b = 12 + i, 3 + (i % 17)
        op = ["+", "-", "x"][i % 3]
        if op == "+":
            q, v = f"What is {a} plus {b}?", a + b
        elif op == "-":
            q, v = f"What is {a} minus {b}?", a - b
        else:
            q, v = f"What is {a} times {b}?", a * b
        bank.append((q, {"kind": "numeric", "value": v, "tol": 0}))
    for i in range(40):
        n = 5 + i
        bank.append((f"In binary, what is the decimal number {n}?",
                     {"kind": "text", "value": bin(n)[2:]}))
    for i in range(60):
        ws = [_WORDS_FOR_SORT[(i * 7 + j * 3) % len(_WORDS_FOR_SORT)] for j in range(3)]
        if len(set(ws)) < 3:
            ws = sorted(set(ws)) + [_WORDS_FOR_SORT[i % len(_WORDS_FOR_SORT)]]
            ws = ws[:3]
        bank.append((f"Sort these words alphabetically and give them comma separated: "
                     f"{', '.join(ws)}.", {"kind": "all_of", "value": sorted(ws)}))
    return bank


NO_TOOL_BANK: list[tuple[str, dict]] = _build_no_tool_bank()


def _gold_as_answer(gold: dict) -> str:
    v = gold["value"]
    if gold["kind"] == "all_of":
        return ", ".join(str(x) for x in v)
    if gold["kind"] == "any_of":
        return str(v[0])
    if gold["kind"] == "none":
        return "That information is not available in the provided passages."
    return str(v)


def _tid(kind: str, seed: int) -> str:
    return f"{kind}-{seed:07d}"


def _chain_len(task: Task) -> int:
    """Tier = number of real tool calls in the reference plan."""
    return sum(1 for step in task.oracle_plan if not step.get("__expect_error__"))


def _pick(rng: random.Random, variants: list[str]) -> str:
    return rng.choice(variants)


def gen_no_tool(rng: random.Random, w: World, seed: int) -> Task:
    q, gold = rng.choice(NO_TOOL_BANK)
    return Task(
        task_id=_tid("notool", seed), seed=seed, prompt=q, task_type="no_tool", difficulty=0,
        gold=gold, forbidden_tools=list(ALL_TOOLS), oracle_plan=[],
        oracle_answer=_gold_as_answer(gold),
        notes="answerable without tools; any tool call is a necessity failure")


# ---------------------------------------------------------------------------
# Multi-hop retrieval chains.
#
# The world is an encyclopedic entity graph. A "route" is a list of relation
# steps; each step (from_kind, fact_key, to_kind, phrase, query_kw) follows a
# fact that names the NEXT entity. The leaf's.
# terminal attribute is read from its passage. A route of length L needs L
# searches (hop 1 finds the start passage, hop i finds the i-th linked entity,
# the last hop's passage supplies the answer).
#
# Question shape (MuSiQue-style nesting):
#   chain E1 -r1-> E2 -r2-> ... -eL (leaf), answer attr A(leaf)
#   phrase(E1) = "E1"
#   phrase(Ek) = r_{k-1}.phrase(phrase(E_{k-1}))
#   question = "What is the {A_word} of {phrase(leaf)}?"
# ---------------------------------------------------------------------------

# relation steps: (from_kind, fact_key, to_kind, phrase_template, query_keyword)
_REL = {
    "feature_country":    ("feature", "located_in", "country",
                           "the country where {0} is located", "country"),
    "country_city":       ("country", "capital", "city",
                           "the capital of {0}", "capital"),
    "city_country":       ("city", "country", "country",
                           "the country that contains {0}", "country"),
    "person_city":        ("person", "born_in", "city",
                           "the city where {0} was born", "born"),
    "person_org":         ("person", "employed_by", "organisation",
                           "the company that employs {0}", "company"),
    "org_city":           ("organisation", "headquartered_in", "city",
                           "the city where {0} is headquartered", "headquarters"),
    "work_person":        ("work", "author", "person",
                           "the person who created {0}", "author"),
}

# terminal attributes by leaf kind: fact_key -> (attr_question_word, answer_kind)
_LEAF_ATTR = {
    "country":    {"official_language": ("official language ", "text"),
                   "population": ("population ", "text")},
    "city":       {"population": ("population ", "text"),
                   "founded_in": ("year of foundation ", "numeric")},
    "organisation": {"founded_in": ("year of foundation ", "numeric"),
                     "field": ("field of activity ", "text")},
    "person":     {"year_of_birth": ("year of birth ", "numeric")},
}

# route templates by hop count: each is a list of _REL keys. Any leaf kind that
# the route reaches is valid as long as _LEAF_ATTR covers it, so the builder
# picks a terminal attribute present on the leaf.
_ROUTES = {
    2: [
        ["feature_country", "country_city"],          # leaf city
        ["person_org", "org_city"],                   # leaf city
        ["org_city", "city_country"],                 # leaf country
        ["work_person", "person_city"],               # leaf city
    ],
    3: [
        ["feature_country", "country_city", "city_country"],   # leaf country
        ["person_org", "org_city", "city_country"],            # leaf country
        ["work_person", "person_org", "org_city"],             # leaf city
        ["feature_country", "country_city", "city_country"],   # leaf country (var B)
    ],
    4: [
        ["feature_country", "country_city", "city_country", "country_city"],   # leaf city
        ["person_org", "org_city", "city_country", "country_city"],            # leaf city
        ["work_person", "person_org", "org_city", "city_country"],             # leaf country
    ],
}


def _ent_kind(w: World, name: str) -> str | None:
    e = w.by_name(name)
    return e["kind"] if e else None


def _follow(w: World, ent: dict, step_key: str) -> dict | None:
    """Return the linked entity reached by step_key, or None."""
    from_k, fact_key, to_k, _, _ = _REL[step_key]
    if ent["kind"] != from_k:
        return None
    val = ent["attrs"].get(fact_key)
    if not val:
        return None
    target = w.by_name(val)
    if target is None or target["kind"] != to_k:
        return None
    return target


def _resolve_chain(w: World, rng: random.Random, steps: list[str]) -> list[dict] | None:
    """From a random eligible start entity, follow the route. Returns [E1..leaf]
    or None if the anchor cannot start the route (try again)."""
    from_k = _REL[steps[0]][0]
    cands = [e for e in w.entities if e["kind"] == from_k]
    rng.shuffle(cands)
    for start in cands:
        chain = [start]
        ok = True
        for step_key in steps:
            nxt = _follow(w, chain[-1], step_key)
            if nxt is None:
                ok = False
                break
            chain.append(nxt)
        if ok:
            return chain
    return None


def _leaf_attr_choice(w: World, rng: random.Random, leaf: dict) -> tuple[str, dict]:
    """Pick a terminal attribute present on the leaf passage."""
    opts = _LEAF_ATTR.get(leaf["kind"], {})
    avail = [k for k in opts if leaf["attrs"].get(k) is not None]
    key = rng.choice(avail)
    (word, kind) = opts[key]
    val = leaf["attrs"][key]
    if kind == "text" and key == "population":
        gold = {"kind": "text", "value": f"{int(val):,}"}
        answer = f"{int(val):,}"
    else:
        gold = {"kind": kind, "value": val, "tol": 0}
        answer = str(val)
    return word, gold, answer


def _phrase_chain(steps: list[str], chain: list[dict]) -> str:
    """Build the nested natural-language phrase for the leaf entity."""
    phrase = chain[0]["name"]
    for k in range(1, len(chain)):
        from_k, _, _, tmpl, _ = _REL[steps[k - 1]]
        phrase = tmpl.format(phrase)
    return phrase


def _build_route_oracle(steps: list[str], chain: list[dict]) -> list[dict]:
    """One search per hop. Query = next entity name + this hop's relation kw."""
    plan = []
    for k in range(1, len(chain)):
        _, _, _, _, kw = _REL[steps[k - 1]]
        plan.append({"name": "search", "arguments": {"query": f"{chain[k]['name']} {kw}"}})
    return plan


def gen_musique(w: World, rng: random.Random, seed: int, hops: int, task_type: str) -> Task | None:
    routes = list(_ROUTES[hops])   # copy: shuffle mutates in place
    rng.shuffle(routes)
    for steps in routes:
        chain = _resolve_chain(w, rng, steps)
        if chain is None:
            continue
        leaf = chain[-1]
        attr_word, gold, answer = _leaf_attr_choice(w, rng, leaf)
        phrase = _phrase_chain(steps, chain)
        prompt = f"What is the {attr_word}of {phrase}?"
        plan = _build_route_oracle(steps, chain)
        return Task(
            task_id=_tid(task_type, seed), seed=seed, prompt=prompt,
            task_type=task_type, difficulty=hops, gold=gold,
            required_tools=list(ALL_TOOLS), required_any=[["search"]],
            forbidden_tools=[],
            oracle_plan=plan,
            oracle_answer=answer,
            notes=f"{hops}-hop retrieval chain; {len(plan)} dependent searches")
    return None


def gen_2hop(rng, w, seed):
    return gen_musique(w, rng, seed, 2, "musique_2hop") or gen_no_tool(rng, w, seed)


def gen_3hop(rng, w, seed):
    return gen_musique(w, rng, seed, 3, "musique_3hop") or gen_no_tool(rng, w, seed)


def gen_4hop(rng, w, seed):
    return gen_musique(w, rng, seed, 4, "musique_4hop") or gen_no_tool(rng, w, seed)


# ---------------------------------------------------------------------------
# unanswerable: the information genuinely is not in the corpus; must abstain.
# ---------------------------------------------------------------------------
FAKE_PRODUCTS = ["Zephyr Vale", "The Obsidian Cipher", "Karnovsky Aeronautics"]


def gen_unanswerable(rng: random.Random, w: World, seed: int) -> Task:
    fake = rng.choice(FAKE_PRODUCTS)
    style = rng.choice(["founded", "population", "location", "work"])
    if style == "founded":
        q = (f"In what year was {fake} founded? Answer only from the passages provided.")
        query = f"{fake} founded"
    elif style == "population":
        q = (f"What is the population of {fake}? Answer only from the passages provided.")
        query = f"{fake} population"
    elif style == "location":
        q = (f"Where is {fake} located? Answer only from the passages provided.")
        query = f"{fake} location"
    else:
        q = (f"Who wrote {fake}? Answer only from the passages provided.")
        query = f"{fake} author"
    return Task(
        task_id=_tid("unans", seed), seed=seed, prompt=q, task_type="unanswerable", difficulty=2,
        gold={"kind": "none", "value": None},
        required_any=[["search"]], forbidden_tools=[],
        oracle_plan=[{"name": "search", "arguments": {"query": query}}],
        oracle_answer="That information is not available in the provided passages.",
        notes="must look, then abstain; an invented answer is the failure mode")


GENERATORS: dict[str, Callable] = {
    "musique_2hop": gen_2hop,
    "musique_3hop": gen_3hop,
    "musique_4hop": gen_4hop,
    "unanswerable": gen_unanswerable,
    "no_tool": gen_no_tool,
}

# Default mix. Multi-hop work is where score is won; negatives are cheap
# insurance against the two default failure modes (call a tool for everything;
# invent an answer rather than abstain).
DEFAULT_MIX: dict[str, float] = {
    "musique_2hop": 0.22,
    "musique_3hop": 0.24,
    "musique_4hop": 0.18,
    "unanswerable": 0.11,
    "no_tool": 0.25,
}

# Tool-chain-length curriculum tier: number of `search` calls the reference
# plan needs (0 for no_tool).
TASK_TIERS: dict[str, int] = {
    "musique_2hop": 2,
    "musique_3hop": 3,
    "musique_4hop": 4,
    "unanswerable": 1,
    "no_tool": 0,
}


def generate(n: int, seed_start: int = 0, mix: dict[str, float] | None = None,
             rng_seed: int = 0) -> list[Task]:
    """
    Mint `n` tasks from seeds [seed_start, seed_start + n).

    Use DISJOINT seed ranges for train and eval. The world is a pure function of
    the seed, so overlapping ranges leak gold answers directly.
    """
    mix = mix or DEFAULT_MIX
    kinds = list(mix)
    weights = [mix[k] for k in kinds]
    picker = random.Random(rng_seed)
    tasks: list[Task] = []
    for i in range(n):
        seed = seed_start + i
        kind = picker.choices(kinds, weights=weights, k=1)[0]
        w = build_world(seed)
        rng = random.Random(seed * 7919 + 13)
        task = GENERATORS[kind](rng, w, seed)
        task.tier = max(TASK_TIERS[kind], _chain_len(task))
        tasks.append(task)
    return tasks


def dev_set(n_per_type: int = 6, seed_start: int = 900_000) -> list[Task]:
    """A balanced, reproducible development set: equal weight per family."""
    tasks: list[Task] = []
    seed = seed_start
    for kind, fn in GENERATORS.items():
        made = 0
        while made < n_per_type:
            w = build_world(seed)
            t = fn(random.Random(seed * 7919 + 13), w, seed)
            if t.task_type != kind:
                # fallback substitution (gen_*_hop can degrade to no_tool); keep going
                tasks.append(t)
                made += 1
                seed += 1
                continue
            t.tier = max(TASK_TIERS.get(kind, 0), _chain_len(t))
            tasks.append(t)
            made += 1
            seed += 1
    return tasks
