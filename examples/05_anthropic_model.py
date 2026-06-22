"""05 — A real Anthropic-backed agent.

Unlike the other examples (which use MockModel), this one talks to a live
Claude model through the Anthropic adapter, with one tool wired up.

Requirements:
  * pip install "agentix[anthropic]"
  * export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/05_anthropic_model.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, tool
from agentix.providers.anthropic import AnthropicModel


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. 'Paris'.
    """
    # A real tool would call a weather API; we fake it for the demo.
    return f"{city}: 21C, partly cloudy."


async def main() -> None:
    agent = Agent(
        model=AnthropicModel(model="claude-opus-4-8", max_tokens=1024),
        system_prompt=(
            "You are a concise assistant. Use the get_weather tool when the "
            "user asks about the weather, then answer in one sentence."
        ),
        # Pass the decorated function — the adapter gets the schema (translated
        # to Anthropic's input_schema) and the executor automatically.
        tools=[get_weather],
    )

    outcome = await agent.run("What's the weather in Paris right now?")

    print("status:", outcome.status)
    print("answer:", outcome.answer)
    print("steps: ", outcome.steps)
    print("tokens:", outcome.tokens_used)


if __name__ == "__main__":
    asyncio.run(main())
