import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from atr.agent.parser import parse_turn

CASES = [
    # (label, text, expect_name, expect_args_subset, expect_strict)
    ("canonical", '<tool_call>\n{"name": "web_search", "arguments": {"query": "berlin"}}\n</tool_call>',
     "web_search", {"query": "berlin"}, True),
    ("with_think", '<think>I should search.</think>\n<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
     "web_search", {"query": "x"}, True),
    ("trailing_comma", '<tool_call>{"name":"calculator","arguments":{"expression":"1+1",},}</tool_call>',
     "calculator", {"expression": "1+1"}, False),
    ("single_quotes", "<tool_call>{'name': 'calculator', 'arguments': {'expression': '2*3'}}</tool_call>",
     "calculator", {"expression": "2*3"}, False),
    ("unquoted_keys", '<tool_call>{name: "fetch_page", arguments: {doc_id: "policy/returns"}}</tool_call>',
     "fetch_page", {"doc_id": "policy/returns"}, False),
    ("truncated", '<tool_call>{"name": "db_query", "arguments": {"table": "orders"',
     "db_query", {"table": "orders"}, False),
    ("parameters_alias", '<tool_call>{"name":"db_query","parameters":{"table":"employees"}}</tool_call>',
     "db_query", {"table": "employees"}, False),
    ("stringified_args", '<tool_call>{"name":"calculator","arguments":"{\\"expression\\": \\"7*8\\"}"}</tool_call>',
     "calculator", {"expression": "7*8"}, False),
    ("markdown_fence", '```json\n{"name": "web_search", "arguments": {"query": "rma"}}\n```',
     "web_search", {"query": "rma"}, False),
    ("function_tag", '<function=calculator>{"expression": "3+4"}</function>',
     "calculator", {"expression": "3+4"}, False),
    ("bare_json", 'Let me look it up. {"name": "fetch_page", "arguments": {"doc_id": "policy/discount"}}',
     "fetch_page", {"doc_id": "policy/discount"}, False),
    ("python_call", '```\ncalculator(expression="12*3")\n```', "calculator", {"expression": "12*3"}, False),
    ("nested_function", '<tool_call>{"type":"function","function":{"name":"web_search","arguments":{"query":"q"}}}</tool_call>',
     "web_search", {"query": "q"}, True),
]

fails = 0
for label, text, name, args, strict in CASES:
    p = parse_turn(text)
    ok = len(p.tool_calls) == 1 and p.tool_calls[0].name == name
    ok = ok and all(p.tool_calls[0].arguments.get(k) == v for k, v in args.items())
    ok = ok and (p.strict_format == strict)
    print(f"{'PASS' if ok else 'FAIL'}  {label:<18} calls={[(c.name, c.repairs) for c in p.tool_calls]} errs={p.errors}")
    fails += not ok

# --- answer / no-tool cases ---
ANS = [
    ("tagged", "<final_answer>42</final_answer>", "42", 0),
    ("untagged", "The capital of France is Paris.", "The capital of France is Paris.", 0),
    ("think_then_answer", "<think>easy</think>\n<final_answer>Paris</final_answer>", "Paris", 0),
    ("parallel_calls", '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>'
                       '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>', None, 2),
    ("answer_after_text", "Based on the docs.\n<final_answer>WH-BER-38</final_answer>", "WH-BER-38", 0),
]
for label, text, ans, ncalls in ANS:
    p = parse_turn(text)
    ok = (p.final_answer == ans) and len(p.tool_calls) == ncalls
    print(f"{'PASS' if ok else 'FAIL'}  {label:<18} answer={p.final_answer!r} ncalls={len(p.tool_calls)}")
    fails += not ok

# strict-mode gate must reject fallback formats
p = parse_turn('```json\n{"name":"web_search","arguments":{"query":"x"}}\n```', allow_fallbacks=False)
ok = len(p.tool_calls) == 0
print(f"{'PASS' if ok else 'FAIL'}  strict_mode_rejects_fence")
fails += not ok

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
