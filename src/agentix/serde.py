"""JSON (de)serialization for the core types.

The agent transcript isn't plain JSON: ``Message.role`` is an enum and
``Message.meta["tool_calls"]`` holds live :class:`ToolCall` objects (so provider
adapters can replay tool turns). This module is the explicit codec that turns the
transcript — and run state — into JSON-able dicts and back. It's the substance of
persistence, and is reusable for logging, audit export, or wire transfer.
"""

from __future__ import annotations

from typing import Any

from .types import AgentOutcome, Message, Role, ToolCall

SCHEMA_VERSION = 1


def tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "args": call.args, "id": call.id}


def tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(name=d["name"], args=dict(d.get("args") or {}), id=d.get("id"))


def message_to_dict(msg: Message) -> dict[str, Any]:
    meta = dict(msg.meta)
    if "tool_calls" in meta:
        meta = {
            **meta,
            "tool_calls": [tool_call_to_dict(c) for c in meta["tool_calls"]],
        }
    return {
        "role": msg.role.value,
        "content": msg.content,
        "trusted": msg.trusted,
        "name": msg.name,
        "meta": meta,
    }


def message_from_dict(d: dict[str, Any]) -> Message:
    meta = dict(d.get("meta") or {})
    if "tool_calls" in meta:
        meta = {
            **meta,
            "tool_calls": [tool_call_from_dict(c) for c in meta["tool_calls"]],
        }
    return Message(
        role=Role(d["role"]),
        content=d["content"],
        trusted=bool(d.get("trusted", False)),
        name=d.get("name"),
        meta=meta,
    )


def transcript_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def transcript_from_dicts(dicts: list[dict[str, Any]]) -> list[Message]:
    return [message_from_dict(d) for d in dicts]


def outcome_to_dict(outcome: AgentOutcome) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": outcome.status,
        "answer": outcome.answer,
        "reason": outcome.reason,
        "steps": outcome.steps,
        "tokens_used": outcome.tokens_used,
        "transcript": transcript_to_dicts(outcome.transcript),
    }


def outcome_from_dict(d: dict[str, Any]) -> AgentOutcome:
    return AgentOutcome(
        status=d["status"],
        answer=d.get("answer"),
        reason=d.get("reason"),
        steps=int(d.get("steps", 0)),
        tokens_used=int(d.get("tokens_used", 0)),
        transcript=transcript_from_dicts(d.get("transcript") or []),
    )
