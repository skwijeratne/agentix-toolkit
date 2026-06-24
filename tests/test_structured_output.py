"""First-class structured output: Agent(response_model=...) (P21)."""

from __future__ import annotations

import json
from typing import Any

from agentix import Agent, MockModel, ModelResponse, Role

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class FakeModel:
    """A stand-in for a Pydantic model (agentix never imports pydantic)."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return SCHEMA

    @classmethod
    def model_validate_json(cls, s: str) -> FakeModel:
        data = json.loads(s)
        if "answer" not in data:
            raise ValueError("missing 'answer'")
        return cls(**data)


class FakeNativeModel:
    """A model adapter that supports native structured output."""

    def __init__(self) -> None:
        self.bound: dict[str, Any] | None = None

    def with_response_format(self, schema: dict[str, Any]) -> FakeNativeModel:
        self.bound = schema
        return self

    async def __call__(self, messages: Any, *, tools: Any = ()) -> ModelResponse:
        return ModelResponse(text='{"answer": "ok"}')


async def test_pydantic_model_wires_validator_and_schema_prompt() -> None:
    agent = Agent(
        model=MockModel([ModelResponse(text='{"answer": "hi"}')]),
        system_prompt="You are helpful.",
        response_model=FakeModel,
    )
    out = await agent.run("question")

    assert out.status == "completed"
    assert isinstance(out.parsed, FakeModel)
    assert out.parsed.answer == "hi"
    # The schema is injected as a system instruction (provider-agnostic enforcement).
    system = next(m.content for m in out.transcript if m.role is Role.SYSTEM)
    assert "JSON Schema" in system
    assert '"answer"' in system


async def test_raw_json_schema_dict_uses_json_validator() -> None:
    agent = Agent(
        model=MockModel([ModelResponse(text='{"x": 1}')]),
        system_prompt="s",
        response_model=SCHEMA,
    )
    out = await agent.run("q")
    assert out.parsed == {"x": 1}  # json_output parses; dict schema, no model class


async def test_native_enforcement_binds_when_supported() -> None:
    native = FakeNativeModel()
    agent = Agent(model=native, system_prompt="s", response_model=FakeModel)
    assert agent.model.bound == SCHEMA  # with_response_format was applied
    out = await agent.run("q")
    assert out.parsed.answer == "ok"


async def test_explicit_validator_wins_over_response_model() -> None:
    agent = Agent(
        model=MockModel([ModelResponse(text='{"answer": "hi"}')]),
        system_prompt="s",
        response_model=FakeModel,
        output_validator=lambda s: {"custom": s},  # explicit wins
    )
    out = await agent.run("q")
    assert out.parsed == {"custom": '{"answer": "hi"}'}
    # schema prompt is still injected even with an explicit validator
    assert any("JSON Schema" in m.content for m in out.transcript if m.role is Role.SYSTEM)


async def test_validation_failure_reprompts_then_parses() -> None:
    agent = Agent(
        model=MockModel(
            [
                ModelResponse(text='{"nope": 1}'),       # fails: missing 'answer'
                ModelResponse(text='{"answer": "fixed"}'),  # retry succeeds
            ]
        ),
        system_prompt="s",
        response_model=FakeModel,
        max_output_retries=1,
    )
    out = await agent.run("q")
    assert out.status == "completed"
    assert out.parsed.answer == "fixed"
    assert out.steps == 2  # one re-prompt
