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
import copy
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .model import ModelFn, ToolSchema
from .types import Message, ModelResponse

#: Extracts a server-requested wait (seconds) from a rate-limit error, or None.
RetryAfterFn = Callable[[BaseException], float | None]

#: Called when a retry is scheduled: (exc, delay_seconds, attempt). Sync or async.
OnRetry = Callable[[BaseException, float, int], Awaitable[None] | None]


def _to_seconds(value: Any) -> float | None:
    try:
        return float(value)  # plain seconds (int/str); HTTP-date form is ignored
    except (TypeError, ValueError):
        return None


def default_retry_after(exc: BaseException) -> float | None:
    """Best-effort ``Retry-After`` extraction across SDK/HTTP error shapes.

    Checks ``exc.retry_after`` (some SDKs) then a ``Retry-After`` response header
    (``exc.response.headers``). Returns seconds, or None if absent/unparseable
    (the caller then falls back to exponential backoff). Only the delta-seconds
    form is honored; an HTTP-date ``Retry-After`` is ignored.
    """
    direct = _to_seconds(getattr(exc, "retry_after", None))
    if direct is not None:
        return direct
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        for key in ("retry-after", "Retry-After"):
            try:
                raw = headers.get(key)
            except Exception:  # noqa: BLE001 - tolerate odd header containers
                raw = None
            secs = _to_seconds(raw)
            if secs is not None:
                return secs
    return None


class RetryModel:
    """Retry a model on transient errors.

    Backoff is exponential by default, but **rate-limit aware**: when the error
    carries a ``Retry-After`` (via ``retry_after``), that server-requested delay
    is honored instead of blind backoff (capped at ``max_sleep``). Wire
    ``on_retry`` to surface/log waits. Set ``retry_after=lambda _e: None`` to
    disable and always use exponential backoff.
    """

    def __init__(
        self,
        model: ModelFn,
        *,
        retries: int = 2,
        backoff: float = 0.5,
        retry_on: Sequence[type[BaseException]] = (Exception,),
        retry_after: RetryAfterFn = default_retry_after,
        max_sleep: float = 60.0,
        on_retry: OnRetry | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.model = model
        self.retries = retries
        self.backoff = backoff
        self.retry_on = tuple(retry_on)
        self.retry_after = retry_after
        self.max_sleep = max_sleep
        self.on_retry = on_retry

    def with_response_format(self, schema: dict[str, Any]) -> RetryModel:
        """Delegate structured-output binding to the wrapped model (for
        ``Agent(response_model=…)`` composed with retries)."""
        clone = copy.copy(self)
        clone.model = _bind_format(self.model, schema)
        return clone

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
            except self.retry_on as exc:
                if attempt >= self.retries:
                    raise
                requested = self.retry_after(exc)
                delay = (
                    min(requested, self.max_sleep)
                    if requested is not None
                    else self.backoff * (2**attempt)
                )
                if self.on_retry is not None:
                    result = self.on_retry(exc, delay, attempt)
                    if inspect.isawaitable(result):
                        await result
                await asyncio.sleep(delay)
                attempt += 1


def _bind_format(model: ModelFn, schema: dict[str, Any]) -> ModelFn:
    bind = getattr(model, "with_response_format", None)
    return bind(schema) if bind is not None else model


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

    def with_response_format(self, schema: dict[str, Any]) -> FallbackModel:
        """Bind structured-output to every wrapped model that supports it."""
        clone = copy.copy(self)
        clone.models = [_bind_format(m, schema) for m in self.models]
        return clone

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
