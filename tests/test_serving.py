"""Serving helpers: event/outcome serialization + the SSE/NDJSON adapter."""

from __future__ import annotations

import json

import pytest

from agentix import (
    Agent,
    AgentOutcome,
    AnswerDelta,
    Done,
    Message,
    MockModel,
    ModelResponse,
    PendingApproval,
    Role,
    ToolCall,
    ToolFinished,
    ToolStarted,
    tool,
)
from agentix.serving import (
    event_to_dict,
    format_sse,
    ndjson_events,
    outcome_to_payload,
    sse_events,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent() -> Agent:
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
            ModelResponse(text="The answer is 5."),
        ]
    )
    return Agent(model=model, system_prompt="math", tools=[add])


# ── event serialization ────────────────────────────────────────────────────


def test_event_to_dict_for_each_type() -> None:
    assert event_to_dict(AnswerDelta("hi")) == {"type": "answer", "text": "hi"}

    started = ToolStarted(ToolCall("add", {"a": 2, "b": 3}, id="c1"))
    assert event_to_dict(started) == {
        "type": "tool_started",
        "tool": "add",
        "args": {"a": 2, "b": 3},
        "id": "c1",
    }

    msg = Message(Role.TOOL, "5", name="add", meta={"ok": True, "call_id": "c1"})
    assert event_to_dict(ToolFinished(msg)) == {
        "type": "tool_finished",
        "tool": "add",
        "ok": True,
        "content": "5",
    }

    outcome = AgentOutcome(status="completed", answer="done", steps=2, cost_usd=0.01)
    payload = event_to_dict(Done(outcome))
    assert payload["type"] == "done"
    assert payload["outcome"]["status"] == "completed"
    assert payload["outcome"]["answer"] == "done"


def test_outcome_to_payload_includes_pending_when_suspended() -> None:
    outcome = AgentOutcome(
        status="suspended",
        reason="awaiting_confirmation",
        pending=[PendingApproval(ToolCall("send_email", {"to": "x"}, id="c9"), "confirm")],
    )
    p = outcome_to_payload(outcome)
    assert p["status"] == "suspended"
    assert "transcript" not in p
    assert p["pending"] == [
        {"id": "c9", "tool": "send_email", "args": {"to": "x"}, "reason": "confirm"}
    ]


def test_format_sse_shape() -> None:
    out = format_sse({"type": "answer", "text": "hi"}, event="answer")
    assert out == 'event: answer\ndata: {"type": "answer", "text": "hi"}\n\n'


# ── streaming serializers (over a real agent.stream) ───────────────────────


async def test_sse_events_over_a_run() -> None:
    chunks = [c async for c in sse_events(_agent().stream("2+3?"))]
    text = "".join(chunks)
    for kind in ("tool_started", "tool_finished", "answer", "done"):
        assert f"event: {kind}" in text
    assert text.endswith("\n\n")  # well-formed SSE records


async def test_ndjson_events_are_one_object_per_line() -> None:
    lines = [line async for line in ndjson_events(_agent().stream("2+3?"))]
    types = [json.loads(line)["type"] for line in lines]
    assert types[-1] == "done"
    assert "answer" in types
    assert all(line.endswith("\n") for line in lines)


# ── FastAPI / Starlette adapter ────────────────────────────────────────────


async def test_sse_response_sets_media_type_and_streams() -> None:
    pytest.importorskip("starlette")
    from agentix.serving import sse_response

    resp = sse_response(_agent().stream("2+3?"))
    assert resp.media_type == "text/event-stream"
    assert resp.headers["cache-control"] == "no-cache"

    body = "".join([part async for part in resp.body_iterator])
    assert "event: done" in body
