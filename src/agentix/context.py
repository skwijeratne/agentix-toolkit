"""Context management — keep the working transcript from growing unbounded.

A long agentic run accumulates a turn per step; without bounds, memory grows and
the provider context window eventually overflows. A :class:`ContextStrategy` is
applied to the message list *before each model call* and returns a (possibly
smaller) list. Strategies are opt-in: with none, the full transcript is kept.

**Pairing safety.** Providers like Anthropic require every tool result to follow
the assistant ``tool_use`` that produced it. The shipped strategies never split
that pair: :class:`TrimRounds` drops whole rounds (an assistant tool-turn plus
its tool results), and :class:`TruncateToolOutputs` only shrinks content in
place. Write custom strategies with the same invariant.
"""

from __future__ import annotations

from .types import Message, Role


class ContextStrategy:
    """Base strategy. Override :meth:`compact`; the default is a no-op.

    Return the same list object when nothing changed (lets the loop skip the
    ``on_compact`` event)."""

    async def compact(self, messages: list[Message]) -> list[Message]:
        return messages


def _split(
    messages: list[Message],
) -> tuple[list[Message], list[Message], list[list[Message]]]:
    """Partition into (leading system msgs, first user task, [rounds]).

    A *round* is an assistant message plus the tool-result messages that follow
    it — the unit that must be kept or dropped together.
    """
    i = 0
    head: list[Message] = []
    while i < len(messages) and messages[i].role is Role.SYSTEM:
        head.append(messages[i])
        i += 1

    task: list[Message] = []
    if i < len(messages) and messages[i].role is Role.USER:
        task.append(messages[i])
        i += 1

    rounds: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages[i:]:
        if msg.role is Role.ASSISTANT:
            if current:
                rounds.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        rounds.append(current)
    return head, task, rounds


class TrimRounds(ContextStrategy):
    """Keep the system prompt, the user's task, and the most recent
    ``max_rounds`` tool rounds — drop older ones."""

    def __init__(self, max_rounds: int) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.max_rounds = max_rounds

    async def compact(self, messages: list[Message]) -> list[Message]:
        head, task, rounds = _split(messages)
        if len(rounds) <= self.max_rounds:
            return messages  # unchanged
        kept = rounds[-self.max_rounds :]
        return [*head, *task, *(m for r in kept for m in r)]


class TruncateToolOutputs(ContextStrategy):
    """Shrink any tool-result message longer than ``max_chars`` in place.

    Preserves every message and all tool pairing — only the content of large
    tool outputs is clipped. Idempotent (won't re-clip already-clipped text).
    """

    def __init__(self, max_chars: int, *, marker: str = "...[truncated]") -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_chars = max_chars
        self.marker = marker

    async def compact(self, messages: list[Message]) -> list[Message]:
        changed = False
        out: list[Message] = []
        for msg in messages:
            if (
                msg.role is Role.TOOL
                and isinstance(msg.content, str)
                and len(msg.content) > self.max_chars
                and not msg.content.endswith(self.marker)
            ):
                changed = True
                out.append(
                    Message(
                        msg.role,
                        msg.content[: self.max_chars] + self.marker,
                        trusted=msg.trusted,
                        name=msg.name,
                        meta=msg.meta,
                    )
                )
            else:
                out.append(msg)
        return out if changed else messages
