"""
System prompt construction.

Two variants, and the choice is a real experiment knob:

* "native"  -- tools go through the tokenizer's chat template `tools=` argument.
               Qwen3 renders its own tool block. Least drift from pretraining,
               fewest tokens, and the safest default.
* "explicit"-- we render the tool block ourselves into the system message.
               Use this when the serving stack cannot pass `tools=`, or when you
               want to A/B a hand-written policy section.

The POLICY text below is deliberately short. Every token here is paid on every
training example and every eval turn, and long behavioural preambles are exactly
what small models learn to ignore. Behaviour should come from the SFT data; this
is a reminder, not a program.
"""
from __future__ import annotations

import json

POLICY = """You are a task-solving assistant with access to tools.

How to work:
- Think briefly about what the task needs, then act.
- If you need information you do not already have, call a tool. Emit the call as:
<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>
- Call one tool at a time. Read its result before deciding the next step.
- If a tool returns an error or an empty result, do not repeat the same call. Change the arguments, or use a different tool.
- If the task needs no external information, answer directly without calling any tool.
- Only call a tool with a side effect when the user explicitly asked for that action.

When you have the answer, give it as:
<final_answer>your answer</final_answer>
Keep the final answer short and direct: the value or fact asked for, not a recap of your steps."""


def render_tools(schemas: list[dict]) -> str:
    lines = ["# Tools", "", "You may call these functions:", "", "<tools>"]
    for s in schemas:
        lines.append(json.dumps(s["function"], ensure_ascii=False))
    lines += ["</tools>"]
    return "\n".join(lines)


def system_message(registry, mode: str = "native", extra: str | None = None) -> str:
    parts = [POLICY]
    if mode == "explicit":
        parts.append(render_tools(registry.schemas()))
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def build_messages(task_prompt: str, registry, mode: str = "native",
                   extra_system: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system_message(registry, mode, extra_system)},
        {"role": "user", "content": task_prompt},
    ]
