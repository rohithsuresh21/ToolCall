"""
One renderer, used by training AND by inference.

This exists to kill an entire class of silent accuracy loss. If you tokenise
training data with `tokenizer.apply_chat_template` and then serve with a stack
that renders tools even slightly differently, the model sees a prompt it was
never trained on and you lose several points for no visible reason. Qwen3's own
template also *strips <think> blocks from previous assistant turns*, which
quietly desynchronises multi-turn tool-use training from multi-turn inference.

So we render ChatML ourselves, identically in both places, and we make the
choice explicit rather than inherited from whatever transformers version is
installed.

Conventions (chosen to sit close to Qwen3's own tool format):
  * tool schemas live in the system turn inside <tools>...</tools>
  * a tool result is a user turn wrapping <tool_response>...</tool_response>
  * consecutive tool results merge into one user turn
"""
from __future__ import annotations

import json
from typing import Sequence

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_system(content: str, tools: Sequence[dict] | None) -> str:
    if not tools:
        return content
    lines = [content, "", "# Tools", "",
             "You may call one or more functions. Their signatures are:", "", "<tools>"]
    for t in tools:
        fn = t.get("function", t)
        lines.append(json.dumps(fn, ensure_ascii=False))
    lines.append("</tools>")
    return "\n".join(lines)


def render(messages: Sequence[dict], tools: Sequence[dict] | None = None,
           add_generation_prompt: bool = True) -> str:
    """messages: [{role: system|user|assistant|tool, content: str, name?: str}]"""
    out: list[str] = []
    i = 0
    msgs = list(messages)
    while i < len(msgs):
        m = msgs[i]
        role = m["role"]
        if role == "system":
            out.append(f"{IM_START}system\n{render_system(m['content'], tools)}{IM_END}\n")
            i += 1
        elif role == "tool":
            blocks = []
            while i < len(msgs) and msgs[i]["role"] == "tool":
                blocks.append(f"<tool_response>\n{msgs[i]['content']}\n</tool_response>")
                i += 1
            out.append(f"{IM_START}user\n" + "\n".join(blocks) + f"{IM_END}\n")
        else:
            out.append(f"{IM_START}{role}\n{m['content']}{IM_END}\n")
            i += 1
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


def assistant_spans(messages: Sequence[dict], tools: Sequence[dict] | None,
                    tokenize) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Tokenise the full conversation and return (input_ids, [(start, end), ...])
    where each span covers exactly one assistant turn's *content plus its
    <|im_end|>* -- i.e. precisely the tokens the model must learn to emit.

    Everything outside these spans (system prompt, user turn, tool results, the
    `<|im_start|>assistant` header) is masked out of the loss. Training on tool
    results in particular is actively harmful: it teaches the model to
    hallucinate tool output instead of waiting for it.
    """
    spans: list[tuple[int, int]] = []
    full_ids = tokenize(render(messages, tools, add_generation_prompt=False))
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = render(messages[:idx], tools, add_generation_prompt=True)
        upto = render(messages[:idx + 1], tools, add_generation_prompt=False)
        s = len(tokenize(prefix))
        e = len(tokenize(upto))
        if e > s:
            spans.append((s, min(e, len(full_ids))))
    return full_ids, spans
