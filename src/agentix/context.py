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

import json
import math
from collections.abc import Callable

from .content import TextPart
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


# ── token counting ─────────────────────────────────────────────────────────

#: Estimates the token count of a string. Any ``Callable[[str], int]`` works —
#: a heuristic, a ``tiktoken`` encoder (``lambda s: len(enc.encode(s))``), or a
#: provider's tokenizer. Strategies take one so trimming can target real tokens.
TokenCounter = Callable[[str], int]


class HeuristicTokenCounter:
    """A dependency-free token estimate: ``ceil(len(text) / chars_per_token)``.

    The 4.0 default is a reasonable average for English prose and code. It is an
    *estimate* — pass a real tokenizer (e.g. ``tiktoken`` for OpenAI, or a
    provider counter) when you need exactness. Erring slightly high keeps
    context-window trimming on the safe side.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self.chars_per_token = chars_per_token

    def __call__(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token)


#: Ready-to-use default counter (4 chars/token heuristic).
approx_token_counter: TokenCounter = HeuristicTokenCounter()


def count_message_tokens(
    message: Message,
    counter: TokenCounter = approx_token_counter,
    *,
    per_message_overhead: int = 4,
    tokens_per_media: int = 600,
) -> int:
    """Estimate the tokens a single message contributes: its text, any tool
    calls it carries, a flat estimate per media part, and a small per-message
    framing overhead (roles/delimiters the provider adds)."""
    total = per_message_overhead
    content = message.content
    if isinstance(content, str):
        total += counter(content)
    else:
        for part in content:
            total += counter(part.text) if isinstance(part, TextPart) else tokens_per_media
    for call in message.meta.get("tool_calls", []):
        total += counter(call.name) + counter(json.dumps(call.args))
    return total


def count_tokens(
    messages: list[Message],
    counter: TokenCounter = approx_token_counter,
    *,
    per_message_overhead: int = 4,
    tokens_per_media: int = 600,
) -> int:
    """Estimate the total tokens of a transcript (sum over its messages)."""
    return sum(
        count_message_tokens(
            m,
            counter,
            per_message_overhead=per_message_overhead,
            tokens_per_media=tokens_per_media,
        )
        for m in messages
    )


class FitContextWindow(ContextStrategy):
    """Keep the transcript under a **token** budget — the unit the model's
    context window is actually measured in (unlike :class:`TrimRounds`, which
    counts rounds, or :class:`TruncateToolOutputs`, which counts characters).

    Always keeps the system prompt(s) and the user's task, then keeps as many of
    the most recent tool rounds as fit ``max_tokens - reserve_tokens`` — dropping
    older whole rounds (never splitting a tool call from its result). Use
    ``reserve_tokens`` to leave room for the model's response. If even the fixed
    prefix plus the latest round overflows, that minimum is kept anyway (a
    required tool turn can't be dropped) — layer :class:`TruncateToolOutputs`
    before this to shrink large tool outputs first.
    """

    def __init__(
        self,
        max_tokens: int,
        counter: TokenCounter = approx_token_counter,
        *,
        reserve_tokens: int = 0,
        per_message_overhead: int = 4,
        tokens_per_media: int = 600,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")
        self.max_tokens = max_tokens
        self.counter = counter
        self.reserve_tokens = reserve_tokens
        self.per_message_overhead = per_message_overhead
        self.tokens_per_media = tokens_per_media

    def _toks(self, messages: list[Message]) -> int:
        return count_tokens(
            messages,
            self.counter,
            per_message_overhead=self.per_message_overhead,
            tokens_per_media=self.tokens_per_media,
        )

    async def compact(self, messages: list[Message]) -> list[Message]:
        budget = self.max_tokens - self.reserve_tokens
        head, task, rounds = _split(messages)
        if not rounds:
            return messages  # nothing droppable

        fixed = [*head, *task]
        running = self._toks(fixed)
        kept: list[list[Message]] = []
        for rnd in reversed(rounds):  # newest first
            cost = self._toks(rnd)
            if kept and running + cost > budget:
                break  # keep at least the most recent round
            running += cost
            kept.append(rnd)
        kept.reverse()

        if len(kept) == len(rounds):
            return messages  # everything fit — unchanged (skip on_compact)
        return [*fixed, *(m for r in kept for m in r)]
