"""Shared translation for OpenAI-style chat APIs.

OpenAI, LiteLLM (and any OpenAI-compatible endpoint, e.g. Ollama's ``/v1``) all
speak the same Chat Completions shape: a flat ``messages`` list, ``tools`` as
``{"type": "function", "function": {...}}``, and a response whose
``choices[0].message`` carries ``content`` plus ``tool_calls`` with **JSON-string**
arguments. This module centralises that translation so each adapter stays thin.

The functions accept the framework-agnostic :class:`~agentix.types.Message` /
:class:`~agentix.model.ToolSchema` and return plain ``dict`` / ``list`` that the
SDKs accept directly; :func:`parse_response` reads the SDK's response object
(duck-typed, so a test fake or a raw dict both work) back into a
:class:`~agentix.types.ModelResponse`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..model import ToolSchema
from ..pricing import cost_usd
from ..types import Message, ModelResponse, Role, ToolCall


def to_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """agentix messages -> OpenAI chat ``messages``."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            out.append({"role": "system", "content": msg.content})
        elif msg.role is Role.USER:
            out.append({"role": "user", "content": msg.content})
        elif msg.role is Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            calls: list[ToolCall] = msg.meta.get("tool_calls", [])
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id or call.name,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args),
                        },
                    }
                    for call in calls
                ]
            out.append(entry)
        elif msg.role is Role.TOOL:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.meta.get("call_id") or msg.name,
                    "content": msg.content,
                }
            )
    return out


def to_tools(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    """agentix tool schemas -> OpenAI ``tools`` (``function`` wrappers)."""
    out: list[dict[str, Any]] = []
    for t in tools:
        params = t.get("parameters") or t.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        fn: dict[str, Any] = {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": params,
        }
        if t.get("strict"):  # OpenAI strict structured tool calls
            fn["strict"] = True
        out.append({"type": "function", "function": fn})
    return out


def _coerce_args(raw: Any) -> dict[str, Any]:
    """Tool-call arguments come back as a JSON string; tolerate dicts too."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_response(response: Any, model: str) -> ModelResponse:
    """OpenAI-style response object -> :class:`ModelResponse`.

    Duck-typed: works with the SDK's pydantic objects, a ``SimpleNamespace``
    fake, or a plain dict (LiteLLM also returns dict-likes).
    """
    choice = _get(response, "choices")[0]
    message = _get(choice, "message")

    text = _get(message, "content", default="") or ""
    tool_calls: list[ToolCall] = []
    for tc in _get(message, "tool_calls", default=None) or []:
        fn = _get(tc, "function")
        tool_calls.append(
            ToolCall(
                name=_get(fn, "name"),
                args=_coerce_args(_get(fn, "arguments", default="")),
                id=_get(tc, "id", default=None),
            )
        )

    usage = _get(response, "usage", default=None)
    input_tokens = int(_get(usage, "prompt_tokens", default=0) or 0) if usage else 0
    output_tokens = int(_get(usage, "completion_tokens", default=0) or 0) if usage else 0

    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        tokens_used=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(model, input_tokens, output_tokens),
    )


_SENTINEL = object()


def _get(obj: Any, key: str, *, default: Any = _SENTINEL) -> Any:
    """Read ``key`` from an attribute *or* a mapping; raise if absent and no
    default was supplied (mirrors how the SDKs expose both shapes)."""
    value = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    if value is _SENTINEL:
        raise KeyError(key)
    return value
