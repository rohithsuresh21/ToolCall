"""
Deterministic, seeded encyclopedic world state for single-tool BM25 retrieval.

The benchmark contract (judge, tools-benchmark = MuSiQue-style multi-hop RC over
a provided 20-passage candidate set) collapses the tool set to ONE action:
BM25 retrieval over the candidate passages. There is no calculator, no database,
no web_search -- the whole task is: retrieve the right passage hop by hop and
compose the answer across passages.

Every episode gets its OWN World built from a seed, so tools are executable and
stateful without leaking between tasks. The task generator reads the same World
to compute gold answers, which is what makes the verifiers *executable* rather
than string-matched against a frozen answer key.

The world is an encyclopedic entity universe: persons, places, organisations and
works, each with a Wikipedia-style passage that states its own attributes AND
cross-references (links) other entities. Multi-hop questions chain through those
links, so answering requires a sequence of BM25 searches whose queries only
become writable after the previous passage has been read -- the dependency
structure that forces (and teaches) sequential retrieval.

Swap-out note: when the organizers' real tool environment arrives, this file and
`builtin.py` are the only two that get replaced. Everything downstream talks to
`ToolRegistry`, not to World.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- surface-form pools (span across the encyclopedic domains, not logistics) --
PERSON_FIRST = ["Ada", "Bo", "Chen", "Divya", "Eli", "Farid", "Gita", "Hana",
                "Ivan", "Jae", "Kira", "Luca", "Mei", "Noor", "Omar", "Priya",
                "Rhea", "Sam", "Tariq", "Uma", "Vik", "Wren", "Xin", "Yara"]
PERSON_LAST = ["Alvarez", "Bakshi", "Cortez", "Dumas", "Eriksen", "Fontaine",
               "Gupta", "Haddad", "Ibrahim", "Jensen", "Kowalski", "Lindqvist",
               "Moreau", "Nakamura", "Okafor", "Petrov", "Rossi", "Silva",
               "Tanaka", "Vasquez"]

NATIONS = [
    ("Khaldonia", "Meridian City", "eastern"), ("Vestorland", "Osthavn", "northern"),
    ("Amaranthe", "Belvora", "southern"), ("Caldury", "Dunskeep", "western"),
    ("Orinella", "Lacerta", "coastal"), ("Thallia", "Nyrmont", "inland"),
]
GEO_FEATURES = ["Aurora Peaks", "Sable River", "Emerald Gorge", "Cobalt Bay",
                "Silver Highlands", "Amber Delta", "Mistveil Range", "Stonetide Coast"]
GEO_KIND = ["mountain range", "river", "canyon", "bay", "highland", "delta",
            "mountain range", "coastline"]

ORG_WORDS = ["Vertex", "Nimbus", "Quartz", "Ember", "Delta", "Onyx", "Prism",
             "Cobalt", "Solstice", "Zenith", "Argon", "Boreas", "Comet", "Drift"]
ORG_KIND = ["Aeronautics", "Dynamics", "Biotech", "Energy", "Mining", "Logistics",
            "Pharma", "Telecom"]
FIELD = ["aerospace engineering", "biotechnology", "renewable energy", "mineral extraction",
         "telecommunications", "logistics"]

WORK_WORDS = ["The Hollow Star", "Echoes of Winter", "The Silent Harbor",
              "Cinder and Ash", "A Song for the Depthless", "Glass Rivers"]
WORK_KIND = ["novel", "film", "symphony", "poetry collection", "biography"]

# attribute answer pools for the leaf of a chain
POP_RANGE = (240_000, 9_800_000)
YEAR_RANGE = (1910, 2019)
ALT_M = (120, 5400)


@dataclass
class World:
    """One episode's universe. Deterministic in `seed`."""
    seed: int
    entities: list[dict] = field(default_factory=list)   # {id, name, kind, attrs}
    documents: list[dict] = field(default_factory=list)  # {doc_id, title, text, facts, links}
    id_index: dict = field(default_factory=dict)         # name_lower -> entity
    now: datetime = field(default_factory=lambda: datetime(2026, 3, 17, 9, 30, tzinfo=timezone.utc))
    # --- observability: every tool call the agent made, recorded by the registry ---
    call_log: list[dict] = field(default_factory=list)
    sent_messages: list[dict] = field(default_factory=list)

    def by_name(self, name: str) -> dict | None:
        return self.id_index.get(name.lower().strip())

    @property
    def tables(self) -> dict[str, list[dict]]:
        return {"documents": self.documents}


def _uniq_names(rng: random.Random, n: int) -> list[str]:
    seen = set()
    out: list[str] = []
    while len(out) < n:
        cand = f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def build_world(seed: int, text_loader=None) -> World:
    """Build the deterministic seeded world.

    `text_loader` is an OPT-IN hook (Part A naturalization): a callable
    (seed:int, doc_id:str) -> str|None that supplies cached, naturally-varied
    passage prose. When it returns text for a doc, that text replaces the templated
    _passage() rendering; the entity `attrs` dict (and thus gold answers and every
    verifier) is never touched. Default None keeps the fixed templated prose."""
    rng = random.Random(seed)
    w = World(seed=seed)

    # ---- geographic layer --------------------------------------------------
    nats = [n for n in NATIONS]
    countries = []
    for i, (cname, ccap, zone) in enumerate(nats):
        pop = rng.randrange(*POP_RANGE, 1000)
        countries.append({
            "id": f"c{i}", "name": cname, "kind": "country",
            "attrs": {"capital": ccap, "region": zone, "population": pop,
                      "official_language": rng.choice(["Caldish", "Vestish", "Andorish", "Nyrmic"])},
        })
    feats = []
    for i, (fname, fkind) in enumerate(zip(rng.sample(GEO_FEATURES, 5), rng.sample(GEO_KIND, 5))):
        host = rng.choice(countries)
        feats.append({
            "id": f"f{i}", "name": fname, "kind": "feature",
            "attrs": {"kind": fkind, "located_in": host["name"],
                      "length_km": rng.randrange(40, 3200), "notable_for": rng.choice([c["name"] for c in countries])},
        })
    cities = []
    for i, c in enumerate(countries):
        cities.append({
            "id": f"city{i}", "name": c["attrs"]["capital"], "kind": "city",
            "attrs": {"country": c["name"], "population": c["attrs"]["population"],
                      "founded_in": rng.randrange(*YEAR_RANGE)},
        })

    # ---- organistions ------------------------------------------------------
    n_org = 8
    org_names = set()
    orgs = []
    while len(orgs) < n_org:
        nm = f"{rng.choice(ORG_WORDS)} {rng.choice(ORG_KIND)}"
        if nm in org_names:
            continue
        org_names.add(nm)
        hq = rng.choice(cities)
        orgs.append({
            "id": f"o{len(orgs)}", "name": nm, "kind": "organisation",
            "attrs": {"field": rng.choice(FIELD).strip(), "headquartered_in": hq["name"],
                      "founded_in": rng.randrange(*YEAR_RANGE),
                      "founder": None},
        })

    # ---- people -------------------------------------------------------------
    people_names = _uniq_names(rng, 10)
    people = []
    for i, nm in enumerate(people_names):
        home = rng.choice(cities)
        people.append({
            "id": f"p{i}", "name": nm, "kind": "person",
            "attrs": {"born_in": home["name"], "year_of_birth": rng.randrange(1850, 2001),
                      "profession": rng.choice(["engineer", "author", "architect", "physicist", "conductor", "miner"]),
                      "employed_by": None},
        })

    # ---- works ---------------------------------------------------------------
    works = []
    wnames = rng.sample(WORK_WORDS, 3)
    for i, nm in enumerate(wnames):
        works.append({
            "id": f"w{i}", "name": nm, "kind": "work",
            "attrs": {"kind": WORK_KIND[i % len(WORK_KIND)],
                      "published_in": rng.randrange(*YEAR_RANGE),
                      "author": None},
        })

    # wire cross-references so the passages form a navigable graph
    for i, pe in enumerate(people):
        pe["attrs"]["employed_by"] = rng.choice(orgs)["name"]
    for i, wk in enumerate(works):
        wk["attrs"]["author"] = rng.choice(people)["name"]

    entities = countries + feats + cities + orgs + people + works
    w.entities = entities
    for e in entities:
        w.id_index[e["name"].lower()] = e

    # ---- build document passages from entities ------------------------------
    if text_loader is not None:
        w.documents = []
        for e in entities:
            d = _passage(e, w)
            nat = text_loader(seed, d["doc_id"])
            if nat:
                d["text"] = nat
                d["naturalized"] = True
            w.documents.append(d)
    else:
        w.documents = [_passage(e, w) for e in entities]
    return w


def _passage(e: dict, w: World) -> dict:
    """Render one Wikipedia-style passage (title + text) from an entity."""
    a = e["attrs"]
    kind = e["kind"]
    if kind == "country":
        text = (f"{e['name']} is a {a['region']} country. Its capital city is "
                f"{a['capital']}. It has an estimated population of {a['population']:,}. "
                f"The official language is {a['official_language']}.")
    elif kind == "feature":
        text = (f"{e['name']} is a {a['kind']} located in {a['located_in']}. It extends "
                f"approximately {a['length_km']} km and is notable for {a['notable_for']}.")
    elif kind == "city":
        text = (f"{e['name']} is the capital city of {a['country']}. It has a population of "
                f"{a['population']:,} and was founded in {a['founded_in']}.")
    elif kind == "organisation":
        text = (f"{e['name']} is an organisation active in the field of {a['field']}. It is "
                f"headquartered in {a['headquartered_in']} and was founded in {a['founded_in']}. "
                f"{'Its founder is ' + a['founder'] + '.' if a.get('founder') else ''}")
    elif kind == "person":
        emp = a.get("employed_by") or "an unnamed employer"
        text = (f"{e['name']} is a {a['profession']} born in {a['born_in']} in {a['year_of_birth']}. "
                f"They are best known for their work at {emp}.")
    else:  # work
        text = (f"{e['name']} is a {a['kind']} published in {a['published_in']}. "
                f"It was created by {a['author']}.")
    return {
        "doc_id": e["id"],
        "title": e["name"],
        "text": text,
        "facts": dict(a),
        "links": [  # names of entities this passage points to (the hop graph)
            a.get("located_in"), a.get("capital"), a.get("country"),
            a.get("headquartered_in"), a.get("founder"), a.get("born_in"),
            a.get("employed_by"), a.get("author"), a.get("notable_for"),
        ],
    }
