"""18 — Verify before trusting: self-consistency + LLM-as-judge.

Two more reliability techniques:

  * `SelfConsistencyModel(model, samples=N)` — wraps a model, samples it N times
    per turn, and returns the majority vote (damps non-determinism on hard
    steps). It's a model, so it drops into `Agent(model=...)`.
  * `JudgeGuard(model, rubric=...)` — an LLM reviews the final answer against a
    rubric and replaces it if it fails (safety / on-brand / format gate).

Dependency-free (everything uses simple stand-in models).

Run:
    PYTHONPATH=src python examples/18_verification.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, JudgeGuard, ModelResponse, Role, SelfConsistencyModel


class CyclingModel:
    """Returns answers from a fixed cycle — mostly '42', sometimes wrong."""

    def __init__(self, answers: list[str]):
        self._answers = answers
        self._i = 0

    async def __call__(self, messages, *, tools=()):
        ans = self._answers[self._i % len(self._answers)]
        self._i += 1
        return ModelResponse(text=ans)


class VerdictModel:
    """A stand-in judge that returns a fixed PASS/FAIL verdict."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    async def __call__(self, messages, *, tools=()):
        return ModelResponse(text=f"{self.verdict} — reviewed")


async def demo_self_consistency() -> None:
    print("== self-consistency (majority vote of 5 samples) ==")
    # The model is noisy: 42, 42, 41, 42, 42 -> majority is 42.
    noisy = CyclingModel(["42", "42", "41", "42", "42"])
    agent = Agent(model=SelfConsistencyModel(noisy, samples=5), system_prompt="sys")
    outcome = await agent.run("what is the answer?")
    print(f"  voted answer: {outcome.answer}  (noise filtered out)\n")


async def demo_judge() -> None:
    print("== JudgeGuard (replace an off-policy answer) ==")
    from agentix import MockModel

    # The agent produces an off-brand answer; the judge fails it.
    agent = Agent(
        model=MockModel([ModelResponse(text="ugh, figure it out yourself")]),
        system_prompt="sys",
        guards=[JudgeGuard(VerdictModel("FAIL"), rubric="must be professional and helpful")],
    )
    outcome = await agent.run("help me")
    print(f"  user sees: {outcome.answer}")


if __name__ == "__main__":
    asyncio.run(demo_self_consistency())
    asyncio.run(demo_judge())
