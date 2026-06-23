"""Bedrock adapter translation, via an injected fake `converse` (sync).

`boto3` is synchronous; the adapter dispatches to a thread, so the fake exposes
a plain `converse(**kwargs)`. Responses follow the Converse API dict shape.
"""

from __future__ import annotations

from typing import Any

from agentix import Agent, LocalToolExecutor, register_price
from agentix.providers.bedrock import BedrockModel


def _resp(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4},
    }


class FakeBedrock:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:  # sync, like boto3
        self.calls.append(kwargs)
        return self._scripted.pop(0)


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


async def test_tool_round_trip_translation() -> None:
    register_price(MODEL_ID, 3.0, 15.0)  # opt into cost tracking
    fake = FakeBedrock(
        [
            _resp([{"toolUse": {"toolUseId": "tu_1", "name": "get_weather",
                                "input": {"city": "Cairo"}}}]),
            _resp([{"text": "Hot in Cairo."}]),
        ]
    )
    agent = Agent(
        model=BedrockModel(client=fake, model=MODEL_ID),
        system_prompt="weather",
        tool_executor=LocalToolExecutor({"get_weather": lambda city: f"{city}: 35C"}),
        tool_schemas=[WEATHER_TOOL],
    )

    outcome = await agent.run("Weather in Cairo?")

    assert outcome.answer == "Hot in Cairo."
    assert outcome.steps == 2
    assert outcome.tokens_used == 30  # (11+4) per call
    assert outcome.cost_usd > 0

    first, second = fake.calls

    # System is a list of text blocks; tools nested under toolConfig/toolSpec.
    assert first["system"] == [{"text": "weather"}]
    spec = first["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["inputSchema"]["json"]["required"] == ["city"]
    assert first["inferenceConfig"]["maxTokens"] == 4096

    # Replay: assistant toolUse + user toolResult, matched by toolUseId.
    msgs = second["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    tool_use = next(b["toolUse"] for b in assistant["content"] if "toolUse" in b)
    assert tool_use["toolUseId"] == "tu_1"
    tool_result = next(
        b["toolResult"]
        for m in msgs
        if m["role"] == "user"
        for b in m["content"]
        if "toolResult" in b
    )
    assert tool_result["toolUseId"] == "tu_1"
    assert tool_result["content"] == [{"text": "Cairo: 35C"}]
    assert tool_result["status"] == "success"


async def test_plain_answer() -> None:
    fake = FakeBedrock([_resp([{"text": "salaam"}])])
    agent = Agent(model=BedrockModel(client=fake, model=MODEL_ID), system_prompt="s")
    outcome = await agent.run("hi")
    assert outcome.answer == "salaam"
