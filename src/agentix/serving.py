"""Serving helpers — turn an ``Agent`` into a streaming HTTP endpoint.

`Agent.stream()` already yields events (answer chunks, tool activity, a final
``Done``). To put an agent behind a web endpoint, you just need to turn those
events into a wire format the browser can read incrementally. This module does
exactly that, in two layers:

* **Dependency-free serializers** — :func:`event_to_dict`, :func:`sse_events`,
  and :func:`ndjson_events` convert the stream into Server-Sent Events (the
  browser ``EventSource`` format) or newline-delimited JSON. No web framework
  required, so they work with any ASGI/WSGI stack.
* **A thin FastAPI/Starlette adapter** — :func:`sse_response` /
  :func:`ndjson_response` wrap those in a ``StreamingResponse`` with the right
  headers. The web dependency is imported lazily, so importing agentix never
  requires it; install it with ``pip install "agentix[serving]"``.

For the request/response (non-streaming) side — including a run that **suspends**
for human approval — :func:`outcome_to_payload` serializes an
:class:`~agentix.types.AgentOutcome` (with any ``pending`` approvals) to a plain
dict you can return as JSON. See ``examples/30_serving_fastapi.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from .streaming import AgentStreamEvent, AnswerDelta, Done, ToolFinished, ToolStarted
from .types import AgentOutcome

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse


def outcome_to_payload(outcome: AgentOutcome) -> dict[str, Any]:
    """A compact, JSON-able view of an outcome for an HTTP response.

    Includes the status, answer, and usage; when the run is ``"suspended"`` it
    also lists the ``pending`` approvals (each with the call ``id`` to pass back
    to ``resume(decisions=…)``). The full transcript is omitted — fetch it from
    your ``Store`` if you need it.
    """
    payload: dict[str, Any] = {
        "status": outcome.status,
        "answer": outcome.answer,
        "reason": outcome.reason,
        "steps": outcome.steps,
        "tokens_used": outcome.tokens_used,
        "cost_usd": outcome.cost_usd,
    }
    if outcome.pending:
        payload["pending"] = [
            {"id": p.call.id, "tool": p.call.name, "args": p.call.args, "reason": p.reason}
            for p in outcome.pending
        ]
    return payload


def event_to_dict(event: AgentStreamEvent) -> dict[str, Any]:
    """Convert one :class:`~agentix.streaming.AgentStreamEvent` to a JSON-able
    dict with a ``type`` discriminator the client can switch on."""
    if isinstance(event, AnswerDelta):
        return {"type": "answer", "text": event.text}
    if isinstance(event, ToolStarted):
        return {
            "type": "tool_started",
            "tool": event.call.name,
            "args": event.call.args,
            "id": event.call.id,
        }
    if isinstance(event, ToolFinished):
        msg = event.result
        return {
            "type": "tool_finished",
            "tool": msg.name,
            "ok": bool(msg.meta.get("ok", True)),
            "content": msg.text,
        }
    if isinstance(event, Done):
        return {"type": "done", "outcome": outcome_to_payload(event.outcome)}
    raise TypeError(f"unknown stream event: {event!r}")  # pragma: no cover


def format_sse(payload: dict[str, Any], *, event: str | None = None) -> str:
    """Format one payload as a Server-Sent Events record."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload)}\n\n"


async def sse_events(events: AsyncIterator[AgentStreamEvent]) -> AsyncIterator[str]:
    """Map an agent event stream to Server-Sent Events text chunks.

    The SSE ``event:`` field carries the event ``type`` (``answer``,
    ``tool_started``, ``tool_finished``, ``done``) so a browser ``EventSource``
    can listen per type.
    """
    async for ev in events:
        payload = event_to_dict(ev)
        yield format_sse(payload, event=payload["type"])


async def ndjson_events(events: AsyncIterator[AgentStreamEvent]) -> AsyncIterator[str]:
    """Map an agent event stream to newline-delimited JSON (one object per line)."""
    async for ev in events:
        yield json.dumps(event_to_dict(ev)) + "\n"


# ── FastAPI / Starlette adapter (lazy import) ──────────────────────────────


def _streaming_response_cls() -> Any:
    try:
        from starlette.responses import StreamingResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Serving responses require Starlette/FastAPI. "
            'Install it with: pip install "agentix[serving]"'
        ) from exc
    return StreamingResponse


def sse_response(
    events: AsyncIterator[AgentStreamEvent],
    *,
    headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """A ``StreamingResponse`` streaming an agent run as Server-Sent Events::

        @app.post("/chat")
        async def chat(body: ChatIn):
            return sse_response(agent.stream(body.message))

    Works with FastAPI or Starlette (``agentix[serving]``).
    """
    cls = _streaming_response_cls()
    hdrs = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **(headers or {})}
    return cast(
        "StreamingResponse",
        cls(sse_events(events), media_type="text/event-stream", headers=hdrs),
    )


def ndjson_response(
    events: AsyncIterator[AgentStreamEvent],
    *,
    headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """A ``StreamingResponse`` streaming an agent run as newline-delimited JSON."""
    cls = _streaming_response_cls()
    hdrs = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **(headers or {})}
    return cast(
        "StreamingResponse",
        cls(ndjson_events(events), media_type="application/x-ndjson", headers=hdrs),
    )
