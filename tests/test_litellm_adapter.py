"""LiteLLM bridge translation, via an injected fake `acompletion`.

LiteLLM returns OpenAI-shaped responses, so the bridge reuses the OpenAI-compat
translation. The fake records request kwargs and pops scripted responses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentix import Agent, LocalToolExecutor
from agentix.providers.litellm import LiteLLMModel


def _resp(content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )


def _tc(id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeLiteLLM:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


ADD_TOOL = {
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
}


async def test_tool_round_trip_and_model_forwarding() -> None:
    fake = FakeLiteLLM(
        [
            _resp(tool_calls=[_tc("c1", "add", '{"a": 2, "b": 3}')]),
            _resp(content="The sum is 5."),
        ]
    )
    agent = Agent(
        model=LiteLLMModel(client=fake, model="anthropic/claude-opus-4-8"),
        system_prompt="math",
        tool_executor=LocalToolExecutor({"add": lambda a, b: str(a + b)}),
        tool_schemas=[ADD_TOOL],
    )

    outcome = await agent.run("2 + 3?")

    assert outcome.answer == "The sum is 5."
    assert outcome.steps == 2
    assert outcome.tokens_used == 20  # (7+3) per call

    # The provider-prefixed model id flows straight to acompletion.
    assert fake.calls[0]["model"] == "anthropic/claude-opus-4-8"
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == "5"


async def test_plain_answer() -> None:
    fake = FakeLiteLLM([_resp(content="hi")])
    agent = Agent(model=LiteLLMModel(client=fake, model="gpt-4o-mini"), system_prompt="s")
    outcome = await agent.run("hello")
    assert outcome.answer == "hi"
