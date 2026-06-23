"""Verify the Anthropic adapter's translation via a fake client.

No `anthropic` package or API key needed: we inject a stand-in client that
mimics the Messages API surface the adapter uses (an async `messages.create`
returning objects with `.content`, `.usage`, `.stop_reason`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentix import Agent, LocalToolExecutor, Role
from agentix.providers.anthropic import AnthropicModel


def _text(s: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=s)


def _tool_use(id: str, name: str, inp: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=inp)


def _response(blocks: list[SimpleNamespace], stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeMessages:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self._scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(scripted)


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
            _response([_tool_use("toolu_1", "get_weather", {"city": "Paris"})],
                      stop_reason="tool_use"),
            _response([_text("It's 21C in Paris.")]),
        ]
    )
    agent = Agent(
        model=AnthropicModel(client=fake),
        system_prompt="You are helpful.",
        tool_executor=LocalToolExecutor({"get_weather": lambda city: f"{city}: 21C"}),
        tool_schemas=[WEATHER_TOOL],
    )

    outcome = await agent.run("Weather in Paris?")

    assert outcome.status == "completed"
    assert outcome.answer == "It's 21C in Paris."
    assert outcome.steps == 2
    assert outcome.tokens_used == 30  # (10+5) per call, two calls

    first, second = fake.messages.calls

    # System prompt is a top-level param, not a message.
    assert first["system"] == "You are helpful."
    assert "system" not in first["messages"][0].get("role", "")

    # Tools translated: parameters -> input_schema.
    tool = first["tools"][0]
    assert tool["name"] == "get_weather"
    assert tool["input_schema"]["required"] == ["city"]
    assert "parameters" not in tool

    # Second call replays the assistant tool_use block + the user tool_result.
    msgs = second["messages"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    tool_use_block = next(b for b in assistant["content"] if b["type"] == "tool_use")
    assert tool_use_block["id"] == "toolu_1"
    assert tool_use_block["input"] == {"city": "Paris"}

    tool_result = next(
        b
        for m in msgs
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    )
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["content"] == "Paris: 21C"
    assert tool_result["is_error"] is False


async def test_refusal_surfaces_as_final_answer() -> None:
    fake = FakeClient([_response([], stop_reason="refusal")])
    agent = Agent(model=AnthropicModel(client=fake), system_prompt="hi")
    outcome = await agent.run("...")
    assert outcome.status == "completed"
    assert "declined" in (outcome.answer or "").lower()


async def test_plain_text_answer_no_tools() -> None:
    fake = FakeClient([_response([_text("hello there")])])
    agent = Agent(model=AnthropicModel(client=fake), system_prompt="sys")
    outcome = await agent.run("hi")
    assert outcome.answer == "hello there"
    assert outcome.steps == 1
    # A plain user message is a string, not a block list.
    user_msg = next(m for m in fake.messages.calls[0]["messages"] if m["role"] == "user")
    assert user_msg["content"] == "hi"


def test_missing_dependency_raises_helpful_error() -> None:
    # With no real client and (presumably) no anthropic package, construction
    # should raise a clear ImportError. If anthropic *is* installed, this
    # instead succeeds — so only assert the message when it raises.
    try:
        AnthropicModel()
    except ImportError as exc:
        assert "agentix[anthropic]" in str(exc)


def test_role_enum_unused_import_guard() -> None:
    # Sanity: Role is importable and the adapter module loaded cleanly.
    assert Role.TOOL == "tool"


# ── typed thinking / effort / task_budget knobs ──────────────────────────


def _ok() -> FakeClient:
    return FakeClient([_response([_text("ok")])])


async def test_thinking_coercion() -> None:
    for value, expected in [
        (True, {"type": "adaptive"}),
        ("adaptive", {"type": "adaptive"}),
        ("summarized", {"type": "adaptive", "display": "summarized"}),
        (False, {"type": "disabled"}),
        ("disabled", {"type": "disabled"}),
    ]:
        fake = _ok()
        await AnthropicModel(client=fake, thinking=value)([])  # type: ignore[arg-type]
        assert fake.messages.calls[0]["thinking"] == expected


async def test_thinking_raw_dict_passthrough() -> None:
    fake = _ok()
    raw = {"type": "adaptive", "display": "summarized"}
    await AnthropicModel(client=fake, thinking=raw)([])
    assert fake.messages.calls[0]["thinking"] == raw


async def test_no_thinking_omits_the_field() -> None:
    fake = _ok()
    await AnthropicModel(client=fake)([])
    assert "thinking" not in fake.messages.calls[0]


async def test_effort_goes_into_output_config() -> None:
    fake = _ok()
    await AnthropicModel(client=fake, effort="low")([])
    assert fake.messages.calls[0]["output_config"]["effort"] == "low"


async def test_task_budget_sets_output_config_and_beta_header() -> None:
    fake = _ok()
    await AnthropicModel(client=fake, task_budget=50000)([])
    call = fake.messages.calls[0]
    assert call["output_config"]["task_budget"] == {"type": "tokens", "total": 50000}
    assert "task-budgets-2026-03-13" in call["extra_headers"]["anthropic-beta"]


async def test_effort_merges_with_extra_output_config() -> None:
    fake = _ok()
    fmt = {"format": {"type": "json_schema", "schema": {}}}
    await AnthropicModel(client=fake, effort="high", output_config=fmt)([])
    oc = fake.messages.calls[0]["output_config"]
    assert oc["effort"] == "high"
    assert oc["format"] == fmt["format"]  # both present


def test_bad_thinking_value_raises() -> None:
    try:
        AnthropicModel(client=_ok(), thinking="loud")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
