"""A scriptable model for tests and examples — zero provider dependencies.

Drive the loop deterministically by handing :class:`MockModel` either a fixed
list of responses (returned in order) or a callable that computes a response
from the current message history.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

from ..model import ToolSchema
from ..streaming import ModelStreamEvent, ResponseComplete, TextDelta, chunk_text
from ..types import Message, ModelResponse

Responder = Callable[[Sequence[Message]], ModelResponse]


class MockModel:
    """Returns pre-scripted :class:`ModelResponse` objects.

    ``script`` is either:
      * a list of ``ModelResponse`` — popped in order; when exhausted, a final
        empty response is returned so the loop always terminates; or
      * a callable ``(messages) -> ModelResponse`` for dynamic behavior.
    """

    def __init__(self, script: Sequence[ModelResponse] | Responder) -> None:
        self._responder: Responder | None
        if callable(script):
            self._queue = []
            self._responder = script
        else:
            self._queue = list(script)
            self._responder = None
        self.calls = 0

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        self.calls += 1
        if self._responder is not None:
            return self._responder(messages)
        if self._queue:
            return self._queue.pop(0)
        return ModelResponse(text="")  # exhausted -> final, loop terminates

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> AsyncIterator[ModelStreamEvent]:
        # Resolve the response, then replay its text as chunks.
        response = await self(messages, tools=tools)
        for piece in chunk_text(response.text):
            yield TextDelta(piece)
        yield ResponseComplete(response)
