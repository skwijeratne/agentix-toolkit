"""Gemini adapter translation, via an injected fake `aio.models`.

The fake mimics `generate_content` returning `candidates[0].content.parts`
(each a text or function_call) plus `usage_metadata`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentix import Agent, LocalToolExecutor
from agentix.providers.gemini import GeminiModel


def _part_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _part_call(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args))


def _resp(parts: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=6),
    )


class FakeModels:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self.aio = SimpleNamespace(models=FakeModels(scripted))


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


async def test_tool_round_trip_translation() -> None:
    fake = FakeClient(
        [
            _resp([_part_call("get_weather", {"city": "Rome"})]),
            _resp([_part_text("Sunny in Rome.")]),
        ]
    )
    agent = Agent(
        model=GeminiModel(client=fake, model="gemini-2.0-flash"),
        system_prompt="weather",
        tool_executor=LocalToolExecutor({"get_weather": lambda city: f"{city}: 28C"}),
        tool_schemas=[WEATHER_TOOL],
    )

    outcome = await agent.run("Weather in Rome?")

    assert outcome.answer == "Sunny in Rome."
    assert outcome.steps == 2
    assert outcome.tokens_used == 30  # (9+6) per call
    assert outcome.cost_usd > 0  # gemini-2.0-flash is priced

    first, second = fake.aio.models.calls

    # System prompt rides on the config, not in contents.
    assert first["config"]["system_instruction"] == "weather"
    # Function declarations nested under a tools[0].function_declarations.
    decls = first["config"]["tools"][0]["function_declarations"]
    assert decls[0]["name"] == "get_weather"

    # Second call replays the model function_call + the user function_response.
    contents = second["contents"]
    model_turn = next(c for c in contents if c["role"] == "model")
    assert model_turn["parts"][0]["function_call"]["name"] == "get_weather"
    fr = next(
        p["function_response"]
        for c in contents
        if c["role"] == "user"
        for p in c["parts"]
        if "function_response" in p
    )
    assert fr["name"] == "get_weather"
    assert fr["response"] == {"result": "Rome: 28C"}


async def test_plain_answer() -> None:
    fake = FakeClient([_resp([_part_text("ciao")])])
    agent = Agent(model=GeminiModel(client=fake), system_prompt="s")
    outcome = await agent.run("hi")
    assert outcome.answer == "ciao"
