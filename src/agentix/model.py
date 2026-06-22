"""The model hook.

A ``ModelFn`` is the one piece of provider-specific wiring the loop can't supply:
it calls your LLM, translating :class:`~agentix.types.Message` <-> the provider's
format and parsing tool calls out of the provider's response.

The ``tools`` argument carries JSON-schema tool definitions so the model can
discover what it is allowed to call. The loop populates it from the tool
registry (P2); a model that takes no tools simply ignores it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .types import Message, ModelResponse

# A tool definition as a JSON-schema dict, e.g.
# {"name": "search", "description": "...", "parameters": {...}}.
ToolSchema = dict[str, Any]


@runtime_checkable
class ModelFn(Protocol):
    """Calls your LLM. Receives the full message history and the available tool
    schemas; returns a :class:`ModelResponse`."""

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse: ...
