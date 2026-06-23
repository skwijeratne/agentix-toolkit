"""OpenAI adapter translation, verified with an injected fake client.

No `openai` package or key needed: the fake mimics the Chat Completions surface
the adapter uses (`chat.completions.create` returning `choices[0].message`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentix import Agent, AnswerDelta, LocalToolExecutor
from agentix.providers.openai import OpenAIModel


def _msg(content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, type="function", function=SimpleNamespace(name=name, arguments=arguments)
    )


def _resp(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeCompletions:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(scripted))


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


async def test_tool_use_round_trip_translation() -> None:
    fake = FakeClient(
        [
            _resp(_msg(tool_calls=[_tc("call_1", "get_weather", '{"city": "Paris"}')])),
            _resp(_msg(content="It's 21C in Paris.")),
        ]
    )
    agent = Agent(
        model=OpenAIModel(client=fake, model="gpt-4o"),
        system_prompt="You are helpful.",
        tool_executor=LocalToolExecutor({"get_weather": lambda city: f"{city}: 21C"}),
        tool_schemas=[WEATHER_TOOL],
    )

    outcome = await agent.run("Weather in Paris?")

    assert outcome.status == "completed"
    assert outcome.answer == "It's 21C in Paris."
    assert outcome.steps == 2
    assert outcome.tokens_used == 30  # (10+5) per call
    assert outcome.cost_usd > 0  # gpt-4o is in the price table

    calls = fake.chat.completions.calls
    first, second = calls

    # System prompt is the first chat message.
    assert first["messages"][0] == {"role": "system", "content": "You are helpful."}

    # Tools translated to the function wrapper, parameters preserved.
    tool = first["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["required"] == ["city"]

    # Second call replays the assistant tool_calls + the tool result message.
    msgs = second["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"

    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "Paris: 21C"


async def test_plain_text_answer() -> None:
    fake = FakeClient([_resp(_msg(content="hello there"))])
    agent = Agent(model=OpenAIModel(client=fake), system_prompt="sys")
    outcome = await agent.run("hi")
    assert outcome.answer == "hello there"
    assert outcome.steps == 1


# ── streaming ─────────────────────────────────────────────────────────────


class _StreamChunks:
    """Async-iterable of streamed chat-completion chunks."""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for p in self._pieces:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=p, tool_calls=None))],
                    usage=None,
                )
            # final usage-only chunk (include_usage)
            yield SimpleNamespace(
                choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4)
            )

        return gen()


class FakeStreamingClient:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> _StreamChunks:
        self.calls.append(kwargs)
        return _StreamChunks(self._pieces)


async def test_stream_emits_answer_deltas() -> None:
    fake = FakeStreamingClient(["Hel", "lo ", "world"])
    agent = Agent(model=OpenAIModel(client=fake, model="gpt-4o"), system_prompt="sys")

    deltas = [e.text async for e in agent.stream("hi") if isinstance(e, AnswerDelta)]
    assert "".join(deltas) == "Hello world"
    # The streaming request opted into incremental output + usage.
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["stream_options"] == {"include_usage": True}
