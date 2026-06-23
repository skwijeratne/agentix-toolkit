from typing import Any

from agentix import (
    Agent,
    AgentPolicy,
    MockModel,
    ModelResponse,
    ToolCall,
    TracingModel,
    secure_defaults,
    tool,
    trace_run,
    tracing_events,
)

# ── a minimal fake OpenTelemetry tracer/span (duck-typed) ────────────────


class FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True

    def __enter__(self) -> "FakeSpan":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.ended = True
        return False


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(self, name: str, **kwargs: Any) -> FakeSpan:
        span = FakeSpan(name)
        self.spans.append(span)
        return span

    def start_as_current_span(self, name: str, **kwargs: Any) -> FakeSpan:
        span = FakeSpan(name)
        self.spans.append(span)
        return span

    def named(self, name: str) -> list[FakeSpan]:
        return [s for s in self.spans if s.name == name]


# ── TracingModel ─────────────────────────────────────────────────────────


async def test_tracing_model_records_attributes() -> None:
    tracer = FakeTracer()
    model = TracingModel(
        MockModel([ModelResponse(text="hi", tokens_used=15, input_tokens=10,
                                 output_tokens=5, cost_usd=0.002)]),
        tracer=tracer,
    )
    resp = await model([])
    assert resp.text == "hi"
    span = tracer.named("agentix.model")[0]
    assert span.attributes["agentix.tokens_used"] == 15
    assert span.attributes["agentix.cost_usd"] == 0.002
    assert span.attributes["agentix.is_final"] is True
    assert span.ended


async def test_tracing_model_records_exception() -> None:
    class BoomModel:
        async def __call__(self, messages, *, tools=()):
            raise RuntimeError("boom")

    tracer = FakeTracer()
    try:
        await TracingModel(BoomModel(), tracer=tracer)([])
        raise AssertionError("expected the error to propagate")
    except RuntimeError:
        pass
    span = tracer.named("agentix.model")[0]
    assert any(isinstance(e, RuntimeError) for e in span.exceptions)


# ── tracing_events: tool spans ───────────────────────────────────────────


async def test_tracing_events_spans_tool_calls() -> None:
    tracer = FakeTracer()

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 1, "b": 2})]),
            ModelResponse(text="3"),
        ]
    )
    agent = Agent(
        model=model, system_prompt="sys", tools=[add], events=tracing_events(tracer=tracer)
    )
    await agent.run("add")

    tool_spans = tracer.named("agentix.tool.add")
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes["agentix.tool.ok"] is True
    assert span.ended


async def test_tracing_events_records_guard_decision() -> None:
    tracer = FakeTracer()

    @tool
    def wire(amount: int) -> str:
        """Wire."""
        return "sent"

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("wire", {"amount": 1})]),
            ModelResponse(text="ok"),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[wire],
        policy=AgentPolicy(prohibited={"wire"}),
        guards=secure_defaults(),
        events=tracing_events(tracer=tracer),
    )
    await agent.run("wire")
    span = tracer.named("agentix.tool.wire")[0]
    decisions = [e for e in span.events if e[0] == "guard_decision"]
    assert decisions and decisions[0][1]["decision"] == "deny"


# ── trace_run ────────────────────────────────────────────────────────────


async def test_trace_run_yields_a_span() -> None:
    tracer = FakeTracer()
    async with trace_run("agentix.run", tracer=tracer) as span:
        span.set_attribute("custom", 1)
    assert span.name == "agentix.run"
    assert span.attributes["custom"] == 1
    assert span.ended


async def test_end_to_end_span_tree() -> None:
    tracer = FakeTracer()

    @tool
    def lookup(q: str) -> str:
        """Lookup."""
        return "result"

    model = TracingModel(
        MockModel(
            [
                ModelResponse(tool_calls=[ToolCall("lookup", {"q": "x"})]),
                ModelResponse(text="done"),
            ]
        ),
        tracer=tracer,
    )
    agent = Agent(
        model=model, system_prompt="sys", tools=[lookup], events=tracing_events(tracer=tracer)
    )
    async with trace_run(tracer=tracer):
        outcome = await agent.run("go")

    assert outcome.answer == "done"
    names = [s.name for s in tracer.spans]
    assert "agentix.run" in names
    assert names.count("agentix.model") == 2  # tool turn + final answer
    assert "agentix.tool.lookup" in names
