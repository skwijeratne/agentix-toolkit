"""Ollama adapter translation, via an injected fake `chat`.

Uses dict-shaped responses to exercise the native-Ollama path: tool-call
arguments arrive as JSON **objects** and usage uses `prompt_eval_count` /
`eval_count`. Local inference => cost is always 0.
"""

from __future__ import annotations

from typing import Any

from agentix import Agent, LocalToolExecutor
from agentix.providers.ollama import OllamaModel


def _resp(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "message": {"content": content, "tool_calls": tool_calls},
        "prompt_eval_count": 12,
        "eval_count": 8,
    }


class FakeOllama:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


async def test_tool_round_trip_object_args_and_zero_cost() -> None:
    call = {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}
    fake = FakeOllama(
        [
            _resp(tool_calls=[call]),
            _resp(content="Cold in Oslo."),
        ]
    )
    agent = Agent(
        model=OllamaModel(client=fake, model="llama3.1"),
        system_prompt="weather",
        tool_executor=LocalToolExecutor({"get_weather": lambda city: f"{city}: 2C"}),
        tool_schemas=[WEATHER_TOOL],
    )

    outcome = await agent.run("Weather in Oslo?")

    assert outcome.answer == "Cold in Oslo."
    assert outcome.steps == 2
    assert outcome.tokens_used == 40  # (12+8) per call
    assert outcome.cost_usd == 0.0  # local inference

    # Tools sent in OpenAI function-wrapper form; replayed tool call uses object args.
    assert fake.calls[0]["tools"][0]["type"] == "function"
    assistant = next(m for m in fake.calls[1]["messages"] if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"city": "Oslo"}
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == "Oslo: 2C"


async def test_plain_answer() -> None:
    fake = FakeOllama([_resp(content="hej")])
    agent = Agent(model=OllamaModel(client=fake), system_prompt="s")
    outcome = await agent.run("hi")
    assert outcome.answer == "hej"
