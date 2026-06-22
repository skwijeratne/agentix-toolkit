"""03 — An async tool and a dynamic model.

Two things this shows that the scripted examples don't:

  1. Tools can be `async` — the executor awaits them. Here `fetch_weather`
     pretends to do async I/O.
  2. MockModel can take a *callable* instead of a fixed list, so the "model"
     reacts to the conversation. Our fake model calls the tool the first time,
     then writes a final answer once it sees the tool's result in the history.
     This mirrors how a real model-driven loop actually flows.

Run:
    PYTHONPATH=src python examples/03_async_dynamic_loop.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from agentix import (
    Agent,
    LocalToolExecutor,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
)


async def fetch_weather(city: str) -> str:
    """An async tool: imagine this hits a weather API."""
    await asyncio.sleep(0.01)
    return f"{city}: 22C, sunny"


def fake_model(messages: Sequence[Message]) -> ModelResponse:
    """Decide what to do based on the conversation so far.

    If we haven't run a tool yet, request the weather tool. Otherwise, the most
    recent tool result is in the history — answer using it.
    """
    last = messages[-1]
    if last.role == Role.TOOL:
        return ModelResponse(text=f"Here's what I found -> {last.content}")
    return ModelResponse(tool_calls=[ToolCall("fetch_weather", {"city": "Lisbon"})])


async def main() -> None:
    agent = Agent(
        model=MockModel(fake_model),
        system_prompt="You are a weather assistant.",
        tool_executor=LocalToolExecutor({"fetch_weather": fetch_weather}),
    )

    # Note: we await run() directly here since we're already in async code.
    outcome = await agent.run("What's the weather in Lisbon?")

    print("answer:", outcome.answer)
    print("steps: ", outcome.steps)
    print("\ntranscript roles:", [m.role.value for m in outcome.transcript])


if __name__ == "__main__":
    asyncio.run(main())
