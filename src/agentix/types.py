"""Core data types shared across the agent loop.

These are plain, framework-agnostic dataclasses. Provider adapters translate
between these and a vendor's wire format; the loop only ever sees these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .content import ContentPart, TextPart


class Role(str, Enum):
    """Author of a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single conversation message.

    ``content`` is either a plain ``str`` (the common case) or a list of
    :class:`~agentix.content.ContentPart` for multimodal input (text interleaved
    with images / documents / audio). Use :attr:`text` for a string view that
    works regardless.

    ``trusted`` marks whether the content originated from the real user
    (an instruction source) rather than from tool output (data to reason
    *about*, never instructions to follow). The loop sets this; guards and
    the security subsystem rely on it.
    """

    role: Role
    content: str | list[ContentPart]
    trusted: bool = False
    name: str | None = None  # tool name, for tool-result messages
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The textual content: the string itself, or the concatenation of the
        :class:`~agentix.content.TextPart` parts (media parts contribute nothing)."""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class ToolResult:
    """The outcome of executing a single :class:`ToolCall`."""

    name: str
    content: str
    call_id: str | None = None
    ok: bool = True


@dataclass
class ModelResponse:
    """What a model adapter returns each turn.

    A response carries assistant ``text`` and/or one or more ``tool_calls``.
    When there are no tool calls the turn is final.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # set by an adapter that knows its model's pricing

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


@dataclass
class AgentOutcome:
    """Terminal result of an agent run."""

    status: str  # "completed" | "aborted" | "refused"
    answer: str | None = None
    parsed: Any = None  # validated/parsed answer, when an output_validator is set
    reason: str | None = None
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    transcript: list[Message] = field(default_factory=list)
