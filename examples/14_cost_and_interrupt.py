"""14 — Cost tracking, USD budgets, and interrupts.

Three operational features for running agents in production:

  * cost: provider adapters set `ModelResponse.cost_usd`; the loop sums it into
    `outcome.cost_usd`. (Here we set it on MockModel responses to demo it.)
  * USD budget: `AgentPolicy(max_budget_usd=...)` aborts once cost exceeds it.
  * interrupt: pass an `Interrupt` and call `trigger()` to stop a run at its
    next safe boundary (e.g. from a UI button or signal handler).

Also shows the pricing helper for real model ids. Dependency-free.

Run:
    PYTHONPATH=src python examples/14_cost_and_interrupt.py
"""

from __future__ import annotations

import asyncio

from agentix import (
    Agent,
    AgentPolicy,
    Done,
    Interrupt,
    MockModel,
    ModelResponse,
    ToolCall,
    cost_usd,
    tool,
)


@tool
def work() -> str:
    """Do a unit of work."""
    return "done"


def demo_pricing() -> None:
    print("== pricing helper (real model ids) ==")
    print("  opus-4-8, 1M in / 0.5M out:", f"${cost_usd('claude-opus-4-8', 1_000_000, 500_000):.2f}")
    print("  haiku-4-5, same usage:     ", f"${cost_usd('claude-haiku-4-5', 1_000_000, 500_000):.2f}\n")


def demo_cost_and_budget() -> None:
    print("== USD budget abort ==")
    # Each model turn 'costs' $0.04; budget is $0.05 -> trips on the 2nd turn.
    def model_fn(_messages: object) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall("work", {})], cost_usd=0.04)

    agent = Agent(
        model=MockModel(model_fn),
        system_prompt="sys",
        tools=[work],
        policy=AgentPolicy(max_budget_usd=0.05, max_steps=100),
    )
    outcome = agent.run_sync("keep working")
    print(f"  status={outcome.status} reason={outcome.reason} cost=${outcome.cost_usd:.2f}\n")


async def demo_interrupt() -> None:
    print("== interrupt ==")
    interrupt = Interrupt()
    calls = 0

    def model_fn(_messages: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls >= 3:           # in a real app, some other task/handler triggers this
            interrupt.trigger()
        return ModelResponse(tool_calls=[ToolCall("work", {})])

    agent = Agent(
        model=MockModel(model_fn),
        system_prompt="sys",
        tools=[work],
        policy=AgentPolicy(max_steps=100),
    )
    async for event in agent.stream("work forever", interrupt=interrupt):
        if isinstance(event, Done):
            o = event.outcome
            print(f"  status={o.status} reason={o.reason} after {o.steps} steps")


if __name__ == "__main__":
    demo_pricing()
    demo_cost_and_budget()
    asyncio.run(demo_interrupt())
