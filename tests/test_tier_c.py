"""Tier C polish: eval loaders, cassettes, subagent cost roll-up, instrument()."""

from __future__ import annotations

from typing import Any

import pytest

from agentix import (
    Agent,
    AgentEvents,
    CassetteModel,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    TracingModel,
    instrument,
    load_cases,
    subagent_tool,
)

# ── eval dataset loaders ───────────────────────────────────────────────────


def test_load_jsonl_folds_extra_keys_into_metadata(tmp_path: Any) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text(
        '{"input": "2+2", "expected": "4"}\n'
        '{"input": "cap of France", "expected": "Paris", "id": "geo", "topic": "geography"}\n'
    )
    cases = load_cases(str(p))
    assert [c.input for c in cases] == ["2+2", "cap of France"]
    assert cases[0].expected == "4"
    assert cases[1].id == "geo"
    assert cases[1].metadata == {"topic": "geography"}


def test_load_json_array_and_csv(tmp_path: Any) -> None:
    j = tmp_path / "cases.json"
    j.write_text('[{"input": "a", "expected": "b"}]')
    assert load_cases(str(j))[0].expected == "b"

    c = tmp_path / "cases.csv"
    c.write_text("input,expected,topic\n2+2,4,math\n")
    row = load_cases(str(c))[0]
    assert row.input == "2+2" and row.expected == "4"
    assert row.metadata == {"topic": "math"}


def test_load_cases_errors(tmp_path: Any) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"expected": "4"}\n')  # missing 'input'
    with pytest.raises(ValueError, match="input"):
        load_cases(str(bad))

    txt = tmp_path / "x.txt"
    txt.write_text("")
    with pytest.raises(ValueError, match="unsupported"):
        load_cases(str(txt))


# ── record / replay cassettes ──────────────────────────────────────────────


async def test_cassette_record_then_replay(tmp_path: Any) -> None:
    path = str(tmp_path / "c.json")
    recorded = ModelResponse(
        text="recorded", tool_calls=[ToolCall("t", {"a": 1}, id="x")], cost_usd=0.01
    )
    real = MockModel([recorded])
    rec = CassetteModel(path, model=real, mode="record")
    await rec([Message(Role.USER, "hi")], tools=[{"name": "t"}])
    rec.save()

    replay = CassetteModel(path, mode="replay")
    r = await replay([Message(Role.USER, "hi")])
    assert r.text == "recorded"
    assert r.tool_calls[0].name == "t" and r.tool_calls[0].args == {"a": 1}
    assert r.cost_usd == 0.01


async def test_cassette_auto_mode_records_then_replays(tmp_path: Any) -> None:
    path = str(tmp_path / "a.json")
    first = CassetteModel(path, model=MockModel([ModelResponse(text="x")]), mode="auto")
    assert first.mode == "record"  # file missing
    await first([Message(Role.USER, "q")])
    first.save()

    second = CassetteModel(path, mode="auto")
    assert second.mode == "replay"  # file now exists
    assert (await second([])).text == "x"


async def test_cassette_exhausted_raises(tmp_path: Any) -> None:
    path = str(tmp_path / "e.json")
    rec = CassetteModel(path, model=MockModel([ModelResponse(text="x")]))
    await rec([])
    rec.save()
    replay = CassetteModel(path)
    await replay([])
    with pytest.raises(RuntimeError, match="exhausted"):
        await replay([])


def test_cassette_record_requires_model(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="model"):
        CassetteModel(str(tmp_path / "x.json"), mode="record")


# ── subagent cost roll-up ──────────────────────────────────────────────────


async def test_subagent_cost_rolls_up_into_parent() -> None:
    child = Agent(
        model=MockModel([ModelResponse(text="42", cost_usd=0.05, tokens_used=10)]),
        system_prompt="child",
    )
    ask = subagent_tool(child, name="ask_child", description="Delegate.")

    def parent_model(messages: list[Message]) -> ModelResponse:
        if any(m.role is Role.TOOL for m in messages):
            return ModelResponse(text="done")  # parent calls cost 0
        return ModelResponse(tool_calls=[ToolCall("ask_child", {"task": "q"}, id="c1")])

    parent = Agent(model=MockModel(parent_model), system_prompt="parent", tools=[ask])
    out = await parent.run("go")

    assert out.answer == "done"
    assert out.cost_usd == 0.05   # child's spend rolled up
    assert out.tokens_used == 10


# ── instrument() ───────────────────────────────────────────────────────────


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ended = False

    def set_attribute(self, *_a: Any) -> None: ...
    def add_event(self, *_a: Any) -> None: ...
    def end(self) -> None:
        self.ended = True

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *_a: Any) -> bool:
        return False


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_span(self, name: str) -> _FakeSpan:
        s = _FakeSpan(name)
        self.spans.append(s)
        return s

    def start_as_current_span(self, name: str) -> _FakeSpan:
        return self.start_span(name)


async def test_instrument_wraps_model_and_preserves_events() -> None:
    seen: list[str] = []
    agent = Agent(
        model=MockModel([ModelResponse(text="hi")]),
        system_prompt="s",
        events=AgentEvents(on_model=lambda *_a: seen.append("model")),
    )
    fake = _FakeTracer()
    returned = instrument(agent, tracer=fake)

    assert returned is agent
    assert isinstance(agent.model, TracingModel)

    out = await agent.run("q")
    assert out.answer == "hi"
    assert "model" in seen  # the pre-existing callback still fires
    assert any(s.name == "agentix.model" for s in fake.spans)  # tracing span created
