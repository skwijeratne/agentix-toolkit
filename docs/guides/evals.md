# Measuring quality (evals)

How do you know your agent is *good* — and that a change to the prompt or model
didn't quietly make it worse? You **evaluate** it: run it against a set of example
questions with known-good answers, and score how it does. This is often called
"evals," and it's how you catch quality regressions before your users do.

## A quick eval

Write some cases (an input and what you expect), run the agent over them, and get a
report:

```python
from agentix import Case, evaluate, contains

cases = [
    Case("What is 2+2?", expected="4"),
    Case("Capital of France?", expected="Paris"),
]

report = await evaluate(cases, agent, scorer=contains())
print(f"{report.passed}/{report.total} passed ({report.pass_rate:.0%})")
report.assert_pass_rate(0.9)     # raises an error if fewer than 90% pass
```

That last line is the trick: drop it into your test suite, and a change that drops
quality below your bar **fails the build** — just like a normal failing test.

## Scoring

A **scorer** decides whether one answer is good enough. Pick one that fits:

| Scorer | Passes when… |
|---|---|
| `exact_match()` | the answer equals the expected text |
| `contains()` | the answer contains the expected text |
| `regex_match(...)` | the answer matches a pattern |
| `predicate(fn)` | your own function returns `True` |
| `llm_judge(...)` | another model judges it against a rubric |

→ Runnable example:
[`examples/17_eval.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/17_eval.py)

## Loading cases from a file

Keep your test cases in a data file instead of in code — `load_cases` reads
`.jsonl`, `.json`, or `.csv`:

```python
from agentix import load_cases

cases = load_cases("tests/cases.jsonl")
```

Each row needs at least an `input`; `expected`, `id`, and any extra columns are
picked up too.

## Double-checking answers

Two more tools for trustworthy results, covered in **[Reliability](reliability.md)**:
ask the model the same question several times and take the majority answer
(`SelfConsistencyModel`), or have a second model review the final answer before it
goes out (`JudgeGuard`).

→ Runnable example:
[`examples/18_verification.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/18_verification.py)
