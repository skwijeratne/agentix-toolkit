"""Observability hooks.

``AgentEvents`` is a bundle of optional callbacks the loop fires as it runs —
for tracing, logging, and the audit trail the governance story promises. Every
callback may be sync or async; the loop awaits awaitable results. Unset
callbacks are simply skipped.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from .types import AgentOutcome, Message, ModelResponse, ToolCall

_Cb = Optional[Callable[..., Union[None, "Awaitable[None]"]]]


@dataclass
class AgentEvents:
    """Optional lifecycle callbacks. Wire only the ones you need."""

    on_model: _Cb = None          # (messages: list[Message], response: ModelResponse)
    on_tool_call: _Cb = None      # (call: ToolCall)
    on_guard_decision: _Cb = None  # (call: ToolCall, decision)
    on_confirm: _Cb = None        # (call: ToolCall, approved: bool)
    on_tool_result: _Cb = None    # (call: ToolCall, result: Message)
    on_compact: _Cb = None        # (before_count: int, after_count: int)
    on_final: _Cb = None          # (outcome: AgentOutcome)

    async def emit(self, name: str, *args: Any) -> None:
        callback = getattr(self, name, None)
        if callback is None:
            return
        result = callback(*args)
        if inspect.isawaitable(result):
            await result


__all__ = ["AgentEvents", "Message", "ModelResponse", "ToolCall", "AgentOutcome"]
