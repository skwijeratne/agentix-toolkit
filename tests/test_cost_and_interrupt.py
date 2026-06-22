
from agentix import (
    Agent,
    AgentEvents,
    AgentPolicy,
    Done,
    Interrupt,
    MockModel,
    ModelResponse,
    ToolCall,
    cost_usd,
    register_price,
    tool,
)

# ── pricing ──────────────────────────────────────────────────────────────


def test_cost_usd_known_model() -> None:
    # opus-4-8: $5/MTok input, $25/MTok output.
    c = cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
    assert c == 30.0
    assert cost_usd("claude-opus-4-8", 0, 0) == 0.0


def test_cost_usd_unknown_model_is_zero() -> None:
    assert cost_usd("some-other-model", 1000, 1000) == 0.0


def test_register_price() -> None:
    register_price("my-model", 2.0, 8.0)
    assert cost_usd("my-model", 1_000_000, 1_000_000) == 10.0


# ── cost flows into the outcome ──────────────────────────────────────────


async def test_outcome_accumulates_cost() -> None:
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("noop", {}, id="c1")], cost_usd=0.02),
            ModelResponse(text="done", cost_usd=0.03),
        ]
    )

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent(model=model, system_prompt="sys", tools=[noop])
    outcome = await agent.run("go")
    assert outcome.status == "completed"
    assert abs(outcome.cost_usd - 0.05) < 1e-9


async def test_max_budget_usd_aborts() -> None:
    # Each call costs 0.04; budget 0.05 -> second call trips it.
    def always_tool(_messages: object) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall("noop", {})], cost_usd=0.04)

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent(
        model=MockModel(always_tool),
        system_prompt="sys",
        tools=[noop],
        policy=AgentPolicy(max_budget_usd=0.05, max_steps=50),
    )
    outcome = await agent.run("go")
    assert outcome.status == "aborted"
    assert outcome.reason == "budget_usd_exceeded"
    assert outcome.cost_usd >= 0.05


# ── interrupt ────────────────────────────────────────────────────────────


async def test_interrupt_stops_at_safe_boundary() -> None:
    interrupt = Interrupt()
    calls = 0

    def model_fn(_messages: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(tool_calls=[ToolCall("noop", {})])

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    # Trigger the interrupt after the 3rd model call via an event hook.
    def on_model(_msgs: object, _resp: object) -> None:
        if calls >= 3:
            interrupt.trigger()

    agent = Agent(
        model=MockModel(model_fn),
        system_prompt="sys",
        tools=[noop],
        policy=AgentPolicy(max_steps=100),
        events=AgentEvents(on_model=on_model),
    )
    outcome = await agent.run("go", interrupt=interrupt)
    assert outcome.status == "aborted"
    assert outcome.reason == "interrupted"
    # Stopped promptly at a boundary, nowhere near max_steps.
    assert outcome.steps < 10


async def test_interrupt_before_start_returns_immediately() -> None:
    interrupt = Interrupt()
    interrupt.trigger()
    agent = Agent(model=MockModel([ModelResponse(text="never")]), system_prompt="sys")
    outcome = await agent.run("go", interrupt=interrupt)
    assert outcome.status == "aborted"
    assert outcome.reason == "interrupted"
    assert outcome.steps == 0


async def test_interrupt_in_stream() -> None:
    interrupt = Interrupt()
    calls = 0

    def model_fn(_messages: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls >= 2:
            interrupt.trigger()
        return ModelResponse(tool_calls=[ToolCall("noop", {})])

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent(
        model=MockModel(model_fn),
        system_prompt="sys",
        tools=[noop],
        policy=AgentPolicy(max_steps=100),
    )
    events = [e async for e in agent.stream("go", interrupt=interrupt)]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.outcome.reason == "interrupted"
