"""Token-F1 answer scoring — the official judge metric.

`answer_f1` is what the GRPO reward optimises, so a bug here is not a reporting
bug: it silently retargets training. These checks pin the SQuAD normaliser
(articles, punctuation, case), multiset counting, and the Task.gold dict forms.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from atr.agent.loop import Step, Trajectory
from atr.tasks.schema import Task, _norm_tokens, answer_f1
from atr.tasks.verifiers import score

FAILS = []


def check(cond, msg):
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def close(got, want, tol=1e-3):
    return abs(got - want) < tol


# ---------------------------------------------------------------- normaliser
check(_norm_tokens("The Richland County!") == ["richland", "county"],
      f"normaliser drops articles, punctuation and case (got {_norm_tokens('The Richland County!')})")
check(_norm_tokens("3,456,000") == ["3456000"],
      "punctuation is deleted, not split on: '3,456,000' is one token")
check(_norm_tokens("  a   an   the  ") == [], "an all-article string normalises to nothing")
check(_norm_tokens("") == [] and _norm_tokens(None) == [], "empty and None normalise to []")
# "a"/"an"/"the" are removed only as whole words -- a substring rule would
# shred 'language' into 'l ngu ge' and quietly change every score.
check(_norm_tokens("The official language") == ["official", "language"],
      "articles are removed on word boundaries only")

# ---------------------------------------------------------------- exact / partial
check(close(answer_f1("Richland County", "Richland County"), 1.0),
      "exact match scores 1.0")
check(close(answer_f1("Richland County", "Richland"), 0.667),
      f"half the gold span scores 0.667 (got {answer_f1('Richland County', 'Richland'):.4f})")
check(close(answer_f1("Portuguese", "The official language is Portuguese"), 0.4),
      f"padded but correct answer scores 0.4 (got {answer_f1('Portuguese', 'The official language is Portuguese'):.4f})")
check(close(answer_f1("1848", ""), 0.0), "empty prediction scores 0.0")
check(close(answer_f1("1848", None), 0.0), "None prediction scores 0.0")
check(close(answer_f1("1848", "   "), 0.0), "whitespace-only prediction scores 0.0")
check(close(answer_f1("1848", "1923"), 0.0), "no token overlap scores 0.0")
check(close(answer_f1("the White House", "White House."), 1.0),
      "articles and trailing punctuation do not cost anything")
check(close(answer_f1("3,456,000", "3456000"), 1.0),
      "comma-grouped and bare numbers score identically")

# ---------------------------------------------------------------- multiset behaviour
# gold has 'new'/'york' TWICE. Multiset intersection caps the overlap at what the
# prediction actually supplies (2 tokens) -> recall 2/4, F1 0.667. A set-based
# intersection would see both distinct types present in both and report 1.0.
f1_rep = answer_f1("New York New York", "New York")
check(close(f1_rep, 0.667),
      f"multiset: repeated gold tokens are not satisfied by one copy (got {f1_rep:.4f}, set-intersection would give 1.0)")
# Mirror case: the prediction repeats a token the gold has once. Precision must
# be charged for the duplicate (2 pred tokens, 1 match) -> 0.5, not 1.0.
f1_rep2 = answer_f1("Tokyo", "Tokyo Tokyo")
check(close(f1_rep2, 0.667),
      f"multiset: a duplicated prediction token is only credited once (got {f1_rep2:.4f})")
# 3 gold tokens, 3 pred tokens, overlap capped at 1 -> p=r=1/3. Set intersection
# would score p=1/1, r=1/3 -> 0.5, i.e. it would reward the padding.
check(close(answer_f1("cat sat mat", "cat cat cat"), 0.333),
      f"multiset: padding with repeats cannot farm precision (got {answer_f1('cat sat mat', 'cat cat cat'):.4f}, set-intersection would give 0.5)")

# ---------------------------------------------------------------- Task.gold dicts
check(close(answer_f1({"kind": "text", "value": "Tokyo"}, "Tokyo"), 1.0),
      "gold dict, kind=text")
check(close(answer_f1({"kind": "numeric", "value": 1969, "tol": 0}, "1969"), 1.0),
      "gold dict, kind=numeric coerces to its surface form")
check(close(answer_f1({"kind": "any_of", "value": ["carbon dioxide", "co2"]}, "CO2"), 1.0),
      "gold dict, kind=any_of scores against its best member")
check(close(answer_f1({"kind": "all_of", "value": ["alpha", "beta"]}, "alpha and beta"), 0.8),
      f"gold dict, kind=all_of scores against the concatenation (got {answer_f1({'kind': 'all_of', 'value': ['alpha', 'beta']}, 'alpha and beta'):.4f})")
check(close(answer_f1({"kind": "none", "value": None}, "anything"), 0.0),
      "gold dict, kind=none has no string to overlap (verifier scores abstention itself)")
check(close(answer_f1(None, "anything"), 0.0), "None gold scores 0.0 rather than raising")

# ---------------------------------------------------------------- bounds
for g, p in [("Richland County", "Richland"), ("Portuguese", "the language is Portuguese"),
             ("a b c", "c b a"), ("1848", "1848 1848")]:
    v = answer_f1(g, p)
    if not (0.0 <= v <= 1.0):
        check(False, f"F1 out of [0,1] for ({g!r},{p!r}): {v}")
check(True, "every F1 stays within [0.0, 1.0]")

# ---------------------------------------------------------------- verifier wiring
def _traj(ans):
    t = Trajectory(task_id="t", prompt="p", stop_reason="answered")
    t.steps = [Step(index=0, assistant_text=f"<final_answer>{ans}</final_answer>",
                    thinking="", tool_calls=[], tool_results=[],
                    parse_errors=[], strict_format=True)]
    t.final_answer = ans
    return t


def _task(gold, ttype="no_tool"):
    return Task(task_id="t", seed=0, prompt="p", task_type=ttype, difficulty=0, gold=gold)


c = score(_task({"kind": "text", "value": "Tokyo"}), _traj("The capital city is Tokyo"))
check(c.final_correct and close(c.final_f1, 0.4),
      f"verifier fills final_f1 independently of final_correct (correct={c.final_correct}, f1={c.final_f1:.4f})")
check(close(score(_task({"kind": "text", "value": "Tokyo"}), _traj("Tokyo")).final_f1, 1.0),
      "verifier: a tight correct answer scores F1 1.0")
check(close(score(_task({"kind": "text", "value": "Tokyo"}), _traj("Paris")).final_f1, 0.0),
      "verifier: a wrong answer scores F1 0.0")

# unanswerable: F1 against an abstention marker would measure phrasing, so the
# abstention decision is the score.
unans = _task({"kind": "none", "value": None}, ttype="unanswerable")
ok = score(unans, _traj("That information is not available in the provided passages."))
bad = score(unans, _traj("The population is 3,456,000."))
check(ok.final_correct and close(ok.final_f1, 1.0),
      f"unanswerable: correct abstention scores F1 1.0 (got {ok.final_f1})")
check(not bad.final_correct and close(bad.final_f1, 0.0),
      f"unanswerable: an invented answer scores F1 0.0 (got {bad.final_f1})")

# ---------------------------------------------------------------- reward wiring
from atr.train.reward import RewardConfig, compute_reward

cfg = RewardConfig()
check(not hasattr(cfg, "p_per_1k_chars"),
      "dead knob removed: p_per_1k_chars (F1 precision subsumes it)")
check(hasattr(cfg, "p_no_answer") and cfg.p_no_answer > 0,
      "p_no_answer retained: pure-F1 reward collapses into answer-avoidance without it")

t = _task({"kind": "text", "value": "Tokyo"})
r_tight, parts_tight = compute_reward(t, score(t, _traj("Tokyo")))
r_padded, parts_padded = compute_reward(t, score(t, _traj("The capital city is Tokyo")))
check(parts_tight["final_correct"] > parts_padded["final_correct"] > 0,
      f"reward: F1 gives a padded-but-correct answer strictly less credit "
      f"({parts_tight['final_correct']:.4f} > {parts_padded['final_correct']:.4f} > 0)")
check("verbosity" not in parts_tight and "verbosity" not in parts_padded,
      "reward: no separate verbosity term is emitted")
check(close(parts_tight["final_correct"], cfg.w_final_correct),
      "reward: a perfect F1 earns the full w_final_correct")

# clip_high must not swallow w_recovery on an otherwise-perfect episode
max_positive = (cfg.w_success + cfg.w_final_correct + cfg.w_selection + cfg.w_args
                + cfg.w_args_strict + cfg.w_format_strict + cfg.w_recovery)
check(cfg.clip_high >= max_positive,
      f"clip_high ({cfg.clip_high}) does not clip a perfect episode ({max_positive:.2f}), "
      f"so w_recovery stays visible inside a group")

# ---------------------------------------------------------------- F6 checkpoint selection
# The shipped checkpoint must be chosen on the metric we train on. The official
# eval is 2/3/4-hop retrieval scored by token-F1 against a gold string -- no
# no_tool family, and the judge never sees tool calls -- so dev_f1 is the whole
# selection rule and there is no second axis worth trading it against.
from atr.tasks.generator import dev_set
from atr.train.grpo import GRPOConfig, GRPOTrainer

accept = GRPOTrainer.canary_accept


def row(f1, succ=0.9, rm=0.5, step=1):
    return {"dev_f1": f1, "dev_success": succ, "reward_mean": rm, "step": step}


check(accept(row(0.10), None), "F6: the first canary always wins (no incumbent)")
check(accept(row(0.90), row(0.80)), "F6: higher dev_f1 wins")
check(not accept(row(0.70), row(0.80)), "F6: lower dev_f1 loses")
check(not accept(row(0.80), row(0.80)), "F6: an exact tie keeps the incumbent (earlier checkpoint)")

# dev_success is a diagnostic. It must have NO influence on selection in either
# direction -- these pairs differ only in dev_success and must rank the same.
check(accept(row(0.90, succ=0.10), row(0.80, succ=0.99)),
      "F6: a collapsed dev_success does not block a higher dev_f1")
check(not accept(row(0.70, succ=0.99), row(0.80, succ=0.10)),
      "F6: a perfect dev_success does not rescue a lower dev_f1")
check(accept(row(0.90, succ=0.1), row(0.80)) == accept(row(0.90, succ=0.9), row(0.80)),
      "F6: dev_success is inert -- identical F1 pairs rank identically at any success")
check(accept(row(0.90, rm=0.0), row(0.80, rm=9.9)), "F6: reward_mean is inert too")

# The freeze scenario that motivated dropping the guard: a run whose F1 climbs
# while success drifts must keep promoting, and must end on the LAST step.
best = None
for step, (f1, succ) in enumerate([(0.60, 0.93), (0.66, 0.87), (0.71, 0.87),
                                   (0.75, 0.80), (0.79, 0.87)], 1):
    cand = row(f1, succ, step=step * 50)
    if accept(cand, best):
        best = cand
check(best["step"] == 250 and abs(best["dev_f1"] - 0.79) < 1e-9,
      f"F6: monotone F1 gains all promote -- ships step {best['step']} at F1 {best['dev_f1']} "
      f"(the old dev_success veto froze this run at step 50, F1 0.60)")

# Removed machinery must stay removed (same contract as test_fix2.py).
check(not hasattr(GRPOTrainer, "canary_tol_warning"), "dead code removed: canary_tol_warning")
check(not hasattr(GRPOConfig(), "canary_success_tol"), "dead knob removed: canary_success_tol")
check(not hasattr(GRPOTrainer, "best_dev_success"),
      "dead state removed: best_dev_success high-water mark")
import inspect

sig = inspect.signature(GRPOTrainer.canary_accept)
check(list(sig.parameters) == ["cand", "best"],
      f"F6: canary_accept takes exactly (cand, best) (got {list(sig.parameters)})")

# dev_success stays in the canary log as a diagnostic.
canary_src = inspect.getsource(GRPOTrainer.run_dev_canary)
check(chr(34) + "dev_f1" + chr(34) in canary_src,
      "F6: run_dev_canary reports dev_f1 (the selection metric)")
check(all(chr(34) + k + chr(34) in canary_src for k in ("dev_success", "dev_necessity")),
      "F6: run_dev_canary still logs dev_success/dev_necessity as diagnostics")

# The canary now draws the SAME families the judge scores (2/3/4-hop only), so
# dev_f1 from the in-run canary is the judge's metric on the judge's population,
# not just an internal ranking proxy. Pinned in both directions.
from collections import Counter

fams = Counter(t.task_type for t in dev_set(n_per_type=GRPOConfig().eval_per_type))
hop = sum(v for k, v in fams.items() if k.startswith("musique"))
check(hop == sum(fams.values()) > 0,
      f"F6: the canary mix IS the judge's mix -- all {hop}/{sum(fams.values())} tasks are hop "
      f"families, so dev_f1 measures what is scored")
check(len(set(fams.values())) == 1,
      f"F6: the canary is balanced across the hop families ({dict(fams)})")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
