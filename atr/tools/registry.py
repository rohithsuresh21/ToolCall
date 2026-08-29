"""
Tool registry: the single seam between "the environment" and "everything else".

`ToolSpec` carries an OpenAI/Qwen-style JSON schema (what the model sees) plus a
python callable (what actually runs). The registry validates arguments before
dispatch and records every call on the World, which is what the verifiers and
the per-objective eval metrics read back.

ADAPTER POINT: to plug in the organizers' fixed tools, write a module that
returns a ToolRegistry whose ToolSpec.fn shells out to their environment. Nothing
else in the codebase changes.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """Raised inside a tool to signal a *recoverable* failure the model should read and react to."""

    def __init__(self, message: str, kind: str = "tool_error", hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.hint = hint

    def to_payload(self) -> dict:
        p = {"error": self.kind, "message": self.message}
        if self.hint:
            p["hint"] = self.hint
        return p


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict            # JSON Schema, object type
    fn: Callable[..., Any]      # fn(world, **kwargs) -> jsonable
    # side_effect tools mutate the world; verifiers check for them, and we can
    # penalise calling them speculatively during RL.
    side_effect: bool = False

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----------------------------------------------------------------------------
# lightweight JSON-schema validation (no jsonschema dependency; keeps the
# training image small and the error messages model-readable)
# ----------------------------------------------------------------------------
_TYPES = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict,
}


def validate_args(schema: dict, args: dict) -> tuple[dict, str | None]:
    """Return (coerced_args, error_message_or_None)."""
    if not isinstance(args, dict):
        return {}, "arguments must be a JSON object"
    props: dict = schema.get("properties", {})
    required: list = schema.get("required", [])
    out: dict = {}

    unknown = [k for k in args if k not in props]
    if unknown:
        return {}, (f"unknown argument(s) {unknown}. valid arguments: {sorted(props)}")

    for key in required:
        if key not in args or args[key] is None:
            return {}, f"missing required argument '{key}'"

    for key, val in args.items():
        spec = props[key]
        want = spec.get("type")
        if want and want in _TYPES:
            py = _TYPES[want]
            if want == "integer" and isinstance(val, float) and val.is_integer():
                val = int(val)
            # a very common small-model failure: numbers emitted as strings.
            # We coerce rather than reject, but the coercion is *recorded* so
            # eval can report "args_loose_ok" vs "args_strict_ok" separately.
            elif want in ("integer", "number") and isinstance(val, str):
                try:
                    val = int(val) if want == "integer" else float(val)
                except ValueError:
                    return {}, f"argument '{key}' must be a {want}, got {val!r}"
            elif want == "boolean" and isinstance(val, str) and val.lower() in ("true", "false"):
                val = val.lower() == "true"
            # bool is a subclass of int in python; don't let True satisfy "integer"
            if want in ("integer", "number") and isinstance(val, bool):
                return {}, f"argument '{key}' must be a {want}, got boolean"
            if not isinstance(val, py):
                return {}, f"argument '{key}' must be a {want}, got {type(val).__name__}"
        if "enum" in spec and val not in spec["enum"]:
            return {}, f"argument '{key}' must be one of {spec['enum']}, got {val!r}"
        out[key] = val
    return out, None


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self.specs)

    def schemas(self) -> list[dict]:
        return [self.specs[n].openai_schema() for n in self.names()]

    def call(self, world, name: str, args: dict) -> dict:
        """
        Dispatch one tool call. NEVER raises for model-caused problems: bad tool
        names and bad arguments come back as readable error payloads, because
        recovering from them is one of the skills we are training.
        """
        t0 = time.time()
        record = {"name": name, "args": args, "ok": False, "coerced": False}

        if name not in self.specs:
            close = [n for n in self.names() if n.startswith(name[:4])] or self.names()
            payload = {"error": "unknown_tool",
                       "message": f"no tool named '{name}'",
                       "hint": f"available tools: {close}"}
            record.update(result=payload, latency_ms=0.0)
            world.call_log.append(record)
            return payload

        spec = self.specs[name]
        coerced, err = validate_args(spec.parameters, args if isinstance(args, dict) else {})
        if err:
            payload = {"error": "invalid_arguments", "message": err,
                       "hint": f"schema for {name}: {json.dumps(spec.parameters)}"}
            record.update(result=payload, latency_ms=0.0)
            world.call_log.append(record)
            return payload
        record["coerced"] = coerced != args

        try:
            result = spec.fn(world, **coerced)
            payload = result if isinstance(result, dict) else {"result": result}
            record["ok"] = True
        except ToolError as e:
            payload = e.to_payload()
        except TypeError as e:
            payload = {"error": "invalid_arguments", "message": str(e)}
        except Exception as e:  # environment bug -> still readable, still recorded
            payload = {"error": "internal_error", "message": f"{type(e).__name__}: {e}"}

        record.update(result=payload, latency_ms=round((time.time() - t0) * 1000, 2))
        world.call_log.append(record)
        return payload
