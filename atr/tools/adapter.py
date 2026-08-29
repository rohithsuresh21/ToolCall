"""
The one file to edit when the organizers' real tool environment arrives.

`get_registry(name)` is what every other module calls. Add a branch here that
builds a ToolRegistry whose ToolSpec.fn forwards to the real environment, and
nothing else in this repo changes -- the agent loop, verifiers, data pipeline,
SFT and GRPO all address tools through the registry.

Three things the adapter must preserve, because metrics depend on them:
  1. Model-caused failures come back as dict payloads with an "error" key,
     never as raised exceptions. Recovery is a trained skill; a crash is not.
  2. The registry appends to `world.call_log`. If the real environment has its
     own session object, wrap it in a shim that exposes `.call_log` (a list),
     `.sent_messages` (a list of side-effect records) and whatever the verifiers
     need to read. Use `tools.sandbox.SessionShim`.
  3. Every forwarded call goes through `tools.sandbox.timed(...)` so a hung
     executor becomes a readable timeout payload instead of a frozen batch
     (F9, FIX-2).

Skeleton for the official branch:

    def _official(world, name, args, session):
        fn = ORGANIZER_EXECUTORS[name]                  # their dispatch table
        result = timed(fn, world.shim, **args, timeout_s=30.0, tool_name=name)
        ok = "error" not in result
        world.shim.record(name, args, result, ok)
        return result

    # build ToolSpecs over _official with each tool's JSON schema, and give every
    # Task's World a `.shim = SessionShim(their_session)` before episodes run.
"""
from __future__ import annotations

from .builtin import build_registry as _build_builtin
from .registry import ToolRegistry
from .sandbox import SessionShim, timed  # noqa: F401  (re-exported for adapters)

_REGISTRIES: dict[str, ToolRegistry] = {}


def get_registry(name: str = "builtin") -> ToolRegistry:
    if name in _REGISTRIES:
        return _REGISTRIES[name]
    if name == "builtin":
        reg = _build_builtin()
    elif name == "official":
        raise NotImplementedError(
            "Plug the organizers' tools in here. Build a ToolRegistry of ToolSpecs "
            "whose fn forwards through tools.sandbox.timed() to their executor and "
            "returns a jsonable dict; log calls via SessionShim.record(); return "
            "error dicts rather than raising."
        )
    else:
        raise KeyError(f"unknown tool environment {name!r}")
    _REGISTRIES[name] = reg
    return reg
