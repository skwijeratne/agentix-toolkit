"""Streaming types.

Two layers of events:

  * **Model stream** — what a streaming model yields: zero or more
    :class:`TextDelta` chunks, then exactly one :class:`ResponseComplete`
    carrying the full :class:`ModelResponse` (text + tool calls + tokens).
  * **Agent stream** — what :meth:`Agent.stream` yields to *you*:
    :class:`AnswerDelta` (incremental answer text), :class:`ToolStarted` /
    :class:`ToolFinished` around each tool call, and a terminal :class:`Done`
    carrying the full :class:`AgentOutcome`.

A model that supports streaming implements :class:`StreamingModelFn` (an async
``stream`` method). The loop falls back to non-streaming if it doesn't.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .model import ToolSchema
from .types import AgentOutcome, Message, ModelResponse, ToolCall

# ── model-level stream events ─────────────────────────────────────────────


@dataclass
class TextDelta:
    text: str


@dataclass
class ResponseComplete:
    response: ModelResponse


ModelStreamEvent = TextDelta | ResponseComplete


@runtime_checkable
class StreamingModelFn(Protocol):
    """A model that can stream. ``stream`` yields TextDelta chunks then one
    ResponseComplete with the assembled response."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> AsyncIterator[ModelStreamEvent]: ...


# ── agent-level stream events ─────────────────────────────────────────────


@dataclass
class AnswerDelta:
    """Incremental text of the model's response for the current turn."""

    text: str


@dataclass
class ToolStarted:
    call: ToolCall


@dataclass
class ToolFinished:
    result: Message


@dataclass
class Done:
    outcome: AgentOutcome


AgentStreamEvent = AnswerDelta | ToolStarted | ToolFinished | Done


def chunk_text(text: str) -> list[str]:
    """Split text into word-ish chunks for a realistic fake stream."""
    if not text:
        return []
    parts = text.split(" ")
    return [p if i == 0 else " " + p for i, p in enumerate(parts)]
