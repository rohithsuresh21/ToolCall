"""End-to-end liveness check for Part A: load the cache into build_world and
independently verify fact-preservation + attrs/gold isolation."""
import json

from atr.tools.world import build_world
from atr.data.naturalize import load_naturalized_loader, _facts_present

CACHE = "artifacts/naturalized_passages.json"

entries = json.load(open(CACHE, encoding="utf-8"))
print("cache entries:", len(entries))

loader = load_naturalized_loader(CACHE)
assert loader is not None, "loader should load from a valid cache"

checked = 0
changed = 0
losses = 0
for seed in (0, 1):
    base = build_world(seed)                       # templated (facts + original text)
    nat = build_world(seed, text_loader=loader)    # naturalized
    for d_t, d_n in zip(base.documents, nat.documents):
        checked += 1
        assert d_n.get("naturalized") is True, f"{d_n['doc_id']} not marked naturalized"
        if d_n["text"] != d_t["text"]:
            changed += 1
    # independent fact-preservation: every surfaced fact survives in naturalized text
    for d in nat.documents:
        src = next(x for x in base.documents if x["doc_id"] == d["doc_id"])
        if not _facts_present(src, d["text"]):
            losses += 1
            print("FACT LOSS", d["doc_id"])
    # attrs / gold isolation: entities identical with and without the loader
    assert base.entities == nat.entities, f"seed {seed}: attrs changed by naturalization"

assert losses == 0, "fact-preservation violated"
print(f"checked {checked} docs; prose changed: {changed}; fact-loss: {losses}")
print("attrs/gold isolation OK")
print("ALL ALIVE: live naturalization verified end-to-end (loader -> build_world -> facts)")
