"""
Turn parser: model text -> (thinking, tool calls, final answer, parse diagnostics).

Why this is more than a regex: format failures are one of the largest single
buckets of lost score for a 1.7B model, and they are also the *cheapest* to fix.
So the parser (a) accepts the four call formats small Qwen models actually emit,
(b) repairs the specific JSON malformations they actually produce, and (c)
records exactly which repair fired.

That last point matters more than it looks. `strict_ok` (parsed with zero
repairs) is the number that tells you whether SFT has taught the format; the
repairs are a safety net for eval, not a substitute for training. Watch the gap
between strict_ok and loose_ok shrink across checkpoints -- when it stops
shrinking, format is no longer your bottleneck and you should spend the next
data batch on reasoning instead.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
OPEN_THINK_RE = re.compile(r"<think>(.*)$", re.S)
TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
# unterminated final tool_call -- extremely common when max_tokens truncates
TOOLCALL_OPEN_RE = re.compile(r"<tool_call>\s*(\{.*)$", re.S)
FN_TAG_RE = re.compile(r"<function\s*=\s*([A-Za-z_][\w]*)\s*>\s*(\{.*?\})\s*</function>", re.S)
FENCE_RE = re.compile(r"```(?:json|tool_code|python)?\s*(.*?)```", re.S)
FINAL_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.S)
FINAL_OPEN_RE = re.compile(r"<final_answer>\s*(.*)$", re.S)
PYCALL_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\((.*)\)\s*$", re.S | re.I)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""
    repairs: list[str] = field(default_factory=list)

    @property
    def strict(self) -> bool:
        return not self.repairs


@dataclass
class ParsedTurn:
    raw: str
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    errors: list[str] = field(default_factory=list)
    # visible text with think blocks and call blocks removed
    content: str = ""

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def strict_format(self) -> bool:
        return not self.errors and all(c.strict for c in self.tool_calls)


# ---------------------------------------------------------------------------
# lenient JSON
# ---------------------------------------------------------------------------
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][\w\-]*)(\s*:)")


def loads_lenient(text: str) -> tuple[Any, list[str]]:
    """Parse JSON, repairing the malformations small models actually produce.

    Returns (obj_or_None, repairs_applied). Order matters: each repair is tried
    only after the cheaper ones fail, so `repairs` stays an honest difficulty
    signal.
    """
    repairs: list[str] = []
    s = text.strip()
    if not s:
        return None, ["empty"]

    try:
        return json.loads(s), repairs
    except Exception:
        pass

    # 1. fenced or prefixed by prose -> take the outermost balanced object
    cand = _extract_balanced(s)
    if cand is not None and cand != s:
        s, _ = cand, repairs.append("extracted_object")
        try:
            return json.loads(s), repairs
        except Exception:
            pass

    # 2. trailing commas
    s2 = _TRAILING_COMMA.sub(r"\1", s)
    if s2 != s:
        s, _ = s2, repairs.append("trailing_comma")
        try:
            return json.loads(s), repairs
        except Exception:
            pass

    # 3. python literal (single quotes, True/False/None, tuples)
    try:
        obj = ast.literal_eval(s)
        repairs.append("python_literal")
        return obj, repairs
    except Exception:
        pass

    # 4. unquoted keys
    s3 = _UNQUOTED_KEY.sub(r'\1"\2"\3', s)
    if s3 != s:
        s, _ = s3, repairs.append("unquoted_key")
        try:
            return json.loads(s), repairs
        except Exception:
            pass

    # 5. truncated mid-object -> close open brackets and retry
    closed = _close_brackets(s)
    if closed != s:
        repairs.append("closed_truncated")
        try:
            return json.loads(_TRAILING_COMMA.sub(r"\1", closed)), repairs
        except Exception:
            pass

    return None, repairs + ["unparseable"]


def _extract_balanced(s: str) -> str | None:
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _close_brackets(s: str) -> str:
    stack, in_str, esc = [], False, False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]" and stack:
            stack.pop()
    out = s
    if in_str:
        out += '"'
    for c in reversed(stack):
        out += "}" if c == "{" else "]"
    return out


# ---------------------------------------------------------------------------
def _normalise_call(obj: Any, raw: str, repairs: list[str]) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        inner = name
        name = inner.get("name")
        obj = {**obj, **inner}
    if not isinstance(name, str) or not name:
        return None
    if any(k in obj for k in ("tool", "tool_name", "function")) and "name" not in obj:
        repairs = repairs + ["alias_name_key"]
    args = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input"))))
    if args is None:
        args, repairs = {}, repairs + ["missing_arguments_key"]
    if isinstance(args, str):
        parsed, r2 = loads_lenient(args)
        if isinstance(parsed, dict):
            args, repairs = parsed, repairs + ["stringified_arguments"] + r2
        else:
            return None
    if not isinstance(args, dict):
        return None
    if "arguments" not in obj and "parameters" in obj:
        repairs = repairs + ["alias_arguments_key"]
    return ToolCall(name=name.strip(), arguments=args, raw=raw, repairs=repairs)


def _parse_pycall(text: str) -> ToolCall | None:
    m = PYCALL_RE.match(text.strip())
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    try:
        node = ast.parse(f"_f({argstr})", mode="eval").body
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        if node.args or not args:
            return None
    except Exception:
        return None
    return ToolCall(name=name, arguments=args, raw=text, repairs=["python_call_syntax"])


def parse_turn(text: str, allow_fallbacks: bool = True) -> ParsedTurn:
    """
    `allow_fallbacks=False` restricts parsing to the canonical
    <tool_call>{json}</tool_call> form. Use it to measure true format accuracy;
    use the default (True) when running the agent, so a recoverable format slip
    does not zero out an otherwise-correct trajectory.
    """
    out = ParsedTurn(raw=text)
    body = text

    m = THINK_RE.search(body)
    if m:
        out.thinking = m.group(1).strip()
        body = THINK_RE.sub("", body)
    else:
        m = OPEN_THINK_RE.search(body)
        if m:  # generation stopped inside the reasoning block
            out.thinking = m.group(1).strip()
            body = OPEN_THINK_RE.sub("", body)
            out.errors.append("unterminated_think")

    # --- canonical form ---------------------------------------------------
    blocks = TOOLCALL_RE.findall(body)
    consumed = TOOLCALL_RE.sub("", body)
    if not blocks:
        m = TOOLCALL_OPEN_RE.search(body)
        if m:
            blocks = [m.group(1)]
            consumed = TOOLCALL_OPEN_RE.sub("", body)
            out.errors.append("unterminated_tool_call")

    for b in blocks:
        obj, reps = loads_lenient(b)
        call = _normalise_call(obj, b, reps)
        if call is None:
            out.errors.append("malformed_tool_call")
        else:
            out.tool_calls.append(call)

    # --- fallbacks --------------------------------------------------------
    if not out.tool_calls and allow_fallbacks:
        for name, argblob in FN_TAG_RE.findall(consumed):
            obj, reps = loads_lenient(argblob)
            if isinstance(obj, dict):
                out.tool_calls.append(ToolCall(name, obj, argblob, reps + ["function_tag_syntax"]))
        if not out.tool_calls:
            consumed_nofence = consumed
            for fenced in FENCE_RE.findall(consumed):
                obj, reps = loads_lenient(fenced)
                call = _normalise_call(obj, fenced, reps + ["markdown_fence"])
                if call is None:
                    call = _parse_pycall(fenced)
                if call is not None:
                    out.tool_calls.append(call)
                    consumed_nofence = consumed_nofence.replace(f"```{fenced}```", "")
            consumed = consumed_nofence if out.tool_calls else consumed
        if not out.tool_calls and "{" in consumed:
            cand = _extract_balanced(consumed)
            if cand:
                obj, reps = loads_lenient(cand)
                call = _normalise_call(obj, cand, reps + ["bare_json"])
                if call is not None:
                    out.tool_calls.append(call)
                    consumed = consumed.replace(cand, "")

    # --- final answer -----------------------------------------------------
    fm = FINAL_RE.search(consumed)
    if fm:
        out.final_answer = fm.group(1).strip()
        consumed = FINAL_RE.sub("", consumed)
    else:
        fm = FINAL_OPEN_RE.search(consumed)
        if fm and fm.group(1).strip():
            out.final_answer = fm.group(1).strip()
            consumed = FINAL_OPEN_RE.sub("", consumed)
            out.errors.append("unterminated_final_answer")

    out.content = re.sub(r"\n{3,}", "\n\n", consumed).strip()
    # A turn with no tool call and no tagged answer still counts as an answer if
    # it said something -- we do not want to punish a correct plain reply.
    if out.final_answer is None and not out.tool_calls and out.content:
        out.final_answer = out.content
        out.errors.append("untagged_final_answer")
    if out.final_answer is None and not out.tool_calls:
        out.errors.append("empty_turn")
    return out
