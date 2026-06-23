"""21 — One agent, many providers.

`agentix` is provider-agnostic: the loop, tools, and guards are identical no
matter which model backs them. Swapping providers is a one-line change to the
`model=` argument. This example is the gallery of those one-liners, plus a
dependency-free demo proving the loop itself doesn't care which you pick.

Each adapter defers its SDK import to construction, so importing the classes is
always safe; you only need the matching extra to actually *run* one:

    pip install "agentix[openai]"     # OpenAIModel  (+ any OpenAI-compatible URL)
    pip install "agentix[gemini]"     # GeminiModel
    pip install "agentix[bedrock]"    # BedrockModel (AWS Converse API)
    pip install "agentix[ollama]"     # OllamaModel  (local models)
    pip install "agentix[litellm]"    # LiteLLMModel (100+ providers via one bridge)

Run:
    python examples/21_providers.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, MockModel, ModelResponse, ToolCall, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. 'Paris'.
    """
    return f"{city}: 21C, partly cloudy."


# ── How you'd construct each provider (copy the line you need) ───────────────
#
#   from agentix.providers.openai import OpenAIModel
#   model = OpenAIModel(model="gpt-4o")                 # reads OPENAI_API_KEY
#   model = OpenAIModel(base_url="http://localhost:11434/v1", api_key="ollama",
#                       model="llama3.1")               # any OpenAI-compatible URL
#
#   from agentix.providers.gemini import GeminiModel
#   model = GeminiModel(model="gemini-2.0-flash")       # reads GOOGLE_API_KEY
#
#   from agentix.providers.bedrock import BedrockModel
#   model = BedrockModel(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
#                                                        # uses the AWS cred chain
#   from agentix.providers.ollama import OllamaModel
#   model = OllamaModel(model="llama3.1")               # local; needs `ollama serve`
#
#   from agentix.providers.litellm import LiteLLMModel
#   model = LiteLLMModel(model="anthropic/claude-opus-4-8")  # provider-prefixed ids
#
# Then it's the same Agent for every one of them:
#
#   agent = Agent(model=model, system_prompt="...", tools=[get_weather])
#   outcome = await agent.run("What's the weather in Paris?")


async def main() -> None:
    # The dependency-free proof: a scripted MockModel drives the exact same loop
    # (tool call -> tool result -> final answer) that every real adapter drives.
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("get_weather", {"city": "Paris"})]),
            ModelResponse(text="It's 21C and partly cloudy in Paris."),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="You are a concise weather assistant.",
        tools=[get_weather],
    )
    outcome = await agent.run("What's the weather in Paris?")

    print("answer:", outcome.answer)
    print("(swap `model=` for any provider above — nothing else changes.)")


if __name__ == "__main__":
    asyncio.run(main())
