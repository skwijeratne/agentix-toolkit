import json

from agentix import (
    Agent,
    AgentPolicy,
    MockModel,
    ModelResponse,
    Role,
    json_output,
    pydantic_output,
    regex_output,
)

# ── the convenience validators ───────────────────────────────────────────


def test_json_output_parses_and_raises() -> None:
    assert json_output('{"a": 1}') == {"a": 1}
    try:
        json_output("not json")
        raise AssertionError("expected error")
    except json.JSONDecodeError:
        pass


def test_regex_output() -> None:
    v = regex_output(r"^\d{4}$")
    assert v("2026") == "2026"
    try:
        v("nope")
        raise AssertionError("expected error")
    except ValueError as e:
        assert "pattern" in str(e)


def test_pydantic_output_duck_typed() -> None:
    # No real pydantic needed: any class with model_validate_json works.
    class FakeModel:
        def __init__(self, value: int) -> None:
            self.value = value

        @classmethod
        def model_validate_json(cls, s: str) -> "FakeModel":
            data = json.loads(s)
            if "value" not in data:
                raise ValueError("missing 'value'")
            return cls(data["value"])

    v = pydantic_output(FakeModel)
    assert v('{"value": 7}').value == 7
    try:
        v("{}")
        raise AssertionError("expected error")
    except ValueError:
        pass


# ── validation + retry through the loop ──────────────────────────────────


async def test_valid_output_sets_parsed() -> None:
    model = MockModel([ModelResponse(text='{"ok": true}')])
    agent = Agent(model=model, system_prompt="sys", output_validator=json_output)
    outcome = await agent.run("give me json")
    assert outcome.status == "completed"
    assert outcome.parsed == {"ok": True}


async def test_invalid_output_retries_then_succeeds() -> None:
    # First answer is malformed; after the retry prompt, the model returns valid JSON.
    model = MockModel(
        [
            ModelResponse(text="not json at all"),
            ModelResponse(text='{"fixed": 1}'),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        output_validator=json_output,
        max_output_retries=2,
    )
    outcome = await agent.run("json please")
    assert outcome.status == "completed"
    assert outcome.parsed == {"fixed": 1}
    assert outcome.steps == 2  # one retry
    # The retry prompt was injected as a user message.
    assert any(
        m.role == Role.USER and "did not pass validation" in m.content
        for m in outcome.transcript
    )


async def test_exhausted_retries_aborts() -> None:
    def always_bad(_messages: object) -> ModelResponse:
        return ModelResponse(text="still not json")

    agent = Agent(
        model=MockModel(always_bad),
        system_prompt="sys",
        output_validator=json_output,
        max_output_retries=2,
        policy=AgentPolicy(max_steps=10),
    )
    outcome = await agent.run("json")
    assert outcome.status == "aborted"
    assert outcome.reason == "output_validation_failed"
    assert outcome.parsed is None
    assert outcome.answer == "still not json"  # last (invalid) answer is preserved


async def test_zero_retries_aborts_immediately() -> None:
    agent = Agent(
        model=MockModel([ModelResponse(text="bad")]),
        system_prompt="sys",
        output_validator=json_output,
        max_output_retries=0,
    )
    outcome = await agent.run("json")
    assert outcome.status == "aborted"
    assert outcome.steps == 1


async def test_async_validator() -> None:
    async def check(answer: str) -> dict:
        return json.loads(answer)

    agent = Agent(model=MockModel([ModelResponse(text='{"a": 2}')]),
                  system_prompt="sys", output_validator=check)
    outcome = await agent.run("go")
    assert outcome.parsed == {"a": 2}


async def test_no_validator_leaves_parsed_none() -> None:
    agent = Agent(model=MockModel([ModelResponse(text="anything")]), system_prompt="sys")
    outcome = await agent.run("go")
    assert outcome.status == "completed"
    assert outcome.parsed is None


async def test_streaming_validates_best_effort_no_retry() -> None:
    from agentix import Done

    model = MockModel([ModelResponse(text='{"x": 1}')])
    agent = Agent(model=model, system_prompt="sys", output_validator=json_output)
    events = [e async for e in agent.stream("go")]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.outcome.parsed == {"x": 1}
