"""Resilient model wrappers.

These wrap a :class:`~agentix.model.ModelFn` and are themselves ``ModelFn``s, so
they drop into ``Agent(model=...)`` and compose
(``FallbackModel([RetryModel(primary), secondary])``).

  * :class:`RetryModel` — retry transient errors with exponential backoff.
  * :class:`FallbackModel` — try each model in order; fall back on failure.

By default they catch ``Exception``; **narrow ``retry_on`` / ``fallback_on`` to
your provider's transient error types** (e.g. ``anthropic.APIStatusError``,
``anthropic.RateLimitError``) so you don't retry/mask real bugs.

Note: these are non-streaming wrappers (no ``stream`` method). Wrapping a
streaming model with them disables streaming — the agent loop transparently
falls back to a single ``__call__`` per turn. Retrying a partially-streamed
response is unsafe, so v1 keeps these one-shot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .model import ModelFn, ToolSchema
from .types import Message, ModelResponse


class RetryModel:
    """Retry a model on transient errors, with exponential backoff."""

    def __init__(
        self,
        model: ModelFn,
        *,
        retries: int = 2,
        backoff: float = 0.5,
        retry_on: Sequence[type[BaseException]] = (Exception,),
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.model = model
        self.retries = retries
        self.backoff = backoff
        self.retry_on = tuple(retry_on)

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await self.model(messages, tools=tools)
            except self.retry_on:
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(self.backoff * (2**attempt))
                attempt += 1


class FallbackModel:
    """Try models in order; on a matching error, fall back to the next.

    Use to escalate (small → big model) or to survive a provider outage. Falls
    back on *exceptions* — a model that returns a (refusal) response is not an
    error here; handle refusals separately."""

    def __init__(
        self,
        models: Sequence[ModelFn],
        *,
        fallback_on: Sequence[type[BaseException]] = (Exception,),
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("FallbackModel needs at least one model")
        self.fallback_on = tuple(fallback_on)

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        last_exc: BaseException | None = None
        for model in self.models:
            try:
                return await model(messages, tools=tools)
            except self.fallback_on as exc:
                last_exc = exc
        assert last_exc is not None  # loop ran at least once
        raise last_exc
