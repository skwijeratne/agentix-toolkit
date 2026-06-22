from types import SimpleNamespace
from typing import Any

from agentix import (
    Agent,
    AnswerDelta,
    Done,
    MockModel,
    ModelResponse,
    PiiRedactionGuard,
    ToolCall,
    ToolFinished,
    ToolStarted,
    tool,
)
from agentix.providers.anthropic import AnthropicModel


async def _collect(agen):
    return [e async for e in agen]


async def test_stream_simple_answer_deltas_then_done() -> None:
    agent = Agent(model=MockModel([ModelResponse(text="hello there world")]), system_prompt="sys")
    events = await _collect(agent.stream("hi"))

    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    assert "".join(d.text for d in deltas) == "hello there world"
    assert isinstance(events[-1], Done)
    assert events[-1].outcome.answer == "hello there world"


async def test_stream_emits_tool_events() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
            ModelResponse(text="5"),
        ]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[add])
    events = await _collect(agent.stream("2+3?"))

    types = [type(e).__name__ for e in events]
    assert "ToolStarted" in types
    assert "ToolFinished" in types
    started = next(e for e in events if isinstance(e, ToolStarted))
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert started.call.name == "add"
    assert finished.result.content == "5"
    assert isinstance(events[-1], Done)
    assert events[-1].outcome.answer == "5"


async def test_stream_done_outcome_is_redacted_even_though_deltas_are_raw() -> None:
    # Documented behavior: deltas stream raw, but Done.outcome.answer goes
    # through on_answer guards.
    model = MockModel([ModelResponse(text="email jane@acme.com please")])
    agent = Agent(model=model, system_prompt="sys", guards=[PiiRedactionGuard()])
    events = await _collect(agent.stream("go"))

    done = events[-1]
    assert isinstance(done, Done)
    assert "jane@acme.com" not in (done.outcome.answer or "")
    assert "[REDACTED]" in (done.outcome.answer or "")


async def test_stream_falls_back_for_non_streaming_model() -> None:
    # A model object without a `stream` method still works via the fallback.
    class OneShot:
        async def __call__(self, messages, *, tools=()):
            return ModelResponse(text="fallback answer")

    agent = Agent(model=OneShot(), system_prompt="sys")
    events = await _collect(agent.stream("go"))
    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    assert "".join(d.text for d in deltas) == "fallback answer"
    assert isinstance(events[-1], Done)


# ── Anthropic streaming via a fake client ────────────────────────────────


class _FakeStreamCtx:
    def __init__(self, deltas: list[str], final: Any) -> None:
        self._deltas = deltas
        self._final = final

    async def __aenter__(self) -> "_FakeStreamCtx":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    async def text_stream(self):  # type: ignore[no-untyped-def]
        for d in self._deltas:
            yield d

    async def get_final_message(self) -> Any:
        return self._final


class _FakeMessages:
    def __init__(self, deltas: list[str], final: Any) -> None:
        self._deltas = deltas
        self._final = final

    def stream(self, **kwargs: Any) -> _FakeStreamCtx:
        return _FakeStreamCtx(self._deltas, self._final)


class _FakeClient:
    def __init__(self, deltas: list[str], final: Any) -> None:
        self.messages = _FakeMessages(deltas, final)


async def test_anthropic_adapter_streams_text_then_response() -> None:
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello world")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    model = AnthropicModel(client=_FakeClient(["hello", " world"], final))
    agent = Agent(model=model, system_prompt="sys")
    events = await _collect(agent.stream("hi"))

    deltas = [e for e in events if isinstance(e, AnswerDelta)]
    assert [d.text for d in deltas] == ["hello", " world"]
    assert isinstance(events[-1], Done)
    assert events[-1].outcome.answer == "hello world"
    assert events[-1].outcome.tokens_used == 5
