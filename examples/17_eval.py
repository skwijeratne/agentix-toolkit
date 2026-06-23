"""17 — Evaluation harness (catch regressions in CI).

Define a dataset of golden cases, run your agent over them, score each, and get
a report you can `assert_pass_rate(...)` on — so a prompt or model change that
drops quality fails your build instead of shipping.

This demo uses a per-case MockModel (deterministic) so it runs with no API key.
In a real eval you'd pass your actual agent (e.g. backed by AnthropicModel).

Run:
    PYTHONPATH=src python examples/17_eval.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, Case, MockModel, ModelResponse, contains, evaluate

# A golden dataset: the question and what the answer must contain.
DATASET = [
    Case("What is 2 + 2?", expected="4", id="math"),
    Case("Capital of France?", expected="Paris", id="geo"),
    Case("Largest planet?", expected="Jupiter", id="astro"),
    Case("Author of Hamlet?", expected="Shakespeare", id="lit"),
]

# Simulated model answers (one is wrong, to show a sub-100% pass rate).
ANSWERS = {
    "What is 2 + 2?": "2 + 2 = 4.",
    "Capital of France?": "The capital of France is Paris.",
    "Largest planet?": "The largest planet is Saturn.",  # WRONG
    "Author of Hamlet?": "Hamlet was written by William Shakespeare.",
}


def agent_for(case: Case) -> Agent:
    return Agent(
        model=MockModel([ModelResponse(text=ANSWERS.get(case.input, ""))]),
        system_prompt="Answer the question concisely.",
    )


async def main() -> None:
    report = await evaluate(DATASET, agent_for, scorer=contains(), concurrency=4)
    print(report.summary())
    print(f"\npass_rate = {report.pass_rate:.0%}, format-success = {report.format_success_rate:.0%}")

    # In CI you'd gate on a threshold; this run is 75%, so 0.7 passes, 0.9 fails.
    try:
        report.assert_pass_rate(0.9)
    except AssertionError:
        print("\n(assert_pass_rate(0.9) failed — a regression would fail the build here)")


if __name__ == "__main__":
    asyncio.run(main())
