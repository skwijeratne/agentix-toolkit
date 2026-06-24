"""Rate-limit awareness: Retry-After-honoring RetryModel (P22)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import agentix.resilience as rl
from agentix import FallbackModel, RetryModel, default_retry_after
from agentix.types import ModelResponse


class RateLimitError(Exception):
    """Mimics a provider 429 with a Retry-After (attribute and/or header)."""

    def __init__(self, *, retry_after: Any = None, header: Any = None) -> None:
        super().__init__("rate limited")
        if retry_after is not None:
            self.retry_after = retry_after
        if header is not None:
            self.response = SimpleNamespace(headers={"retry-after": header})


class Flaky:
    """Raises the queued errors, then returns ok."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self, messages: Any, *, tools: Any = ()) -> ModelResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ModelResponse(text="ok")


@pytest.fixture
def no_real_sleep(monkeypatch: Any) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(rl.asyncio, "sleep", fake_sleep)
    return slept


# ── default_retry_after extraction ─────────────────────────────────────────


def test_retry_after_from_attribute_header_and_absent() -> None:
    assert default_retry_after(RateLimitError(retry_after=2.5)) == 2.5
    assert default_retry_after(RateLimitError(header="7")) == 7.0
    assert default_retry_after(RateLimitError(header="not-a-number")) is None
    assert default_retry_after(ValueError("boom")) is None


# ── RetryModel behavior ────────────────────────────────────────────────────


async def test_honors_retry_after_over_backoff(no_real_sleep: list[float]) -> None:
    model = Flaky([RateLimitError(retry_after=2.5)])
    out = await RetryModel(model, backoff=0.5)([])
    assert out.text == "ok"
    assert no_real_sleep == [2.5]  # server delay, not 0.5 backoff


async def test_falls_back_to_exponential_backoff(no_real_sleep: list[float]) -> None:
    model = Flaky([ValueError("transient"), ValueError("transient")])
    out = await RetryModel(model, backoff=0.5, retries=3)([])
    assert out.text == "ok"
    assert no_real_sleep == [0.5, 1.0]  # 0.5*2^0, 0.5*2^1


async def test_max_sleep_caps_retry_after(no_real_sleep: list[float]) -> None:
    model = Flaky([RateLimitError(retry_after=120)])
    await RetryModel(model, max_sleep=10)([])
    assert no_real_sleep == [10.0]


async def test_on_retry_callback_fires(no_real_sleep: list[float]) -> None:
    seen: list[tuple[float, int]] = []

    def on_retry(exc: BaseException, delay: float, attempt: int) -> None:
        seen.append((delay, attempt))

    model = Flaky([RateLimitError(retry_after=3)])
    await RetryModel(model, on_retry=on_retry)([])
    assert seen == [(3.0, 0)]


async def test_disable_retry_after_uses_backoff(no_real_sleep: list[float]) -> None:
    model = Flaky([RateLimitError(retry_after=99)])
    await RetryModel(model, backoff=0.25, retry_after=lambda _e: None)([])
    assert no_real_sleep == [0.25]  # ignored the 99s server hint


async def test_exhausted_retries_reraise(no_real_sleep: list[float]) -> None:
    model = Flaky([ValueError("a"), ValueError("b")])
    with pytest.raises(ValueError):
        await RetryModel(model, retries=1)([])  # 1 retry, 2 failures


# ── structured-output delegation through the wrappers ──────────────────────


class _Native:
    def __init__(self) -> None:
        self.bound: dict[str, Any] | None = None

    def with_response_format(self, schema: dict[str, Any]) -> _Native:
        self.bound = schema
        return self

    async def __call__(self, messages: Any, *, tools: Any = ()) -> ModelResponse:
        return ModelResponse(text="ok")


def test_retrymodel_delegates_response_format() -> None:
    native = _Native()
    RetryModel(native).with_response_format({"k": "v"})
    assert native.bound == {"k": "v"}


def test_fallbackmodel_binds_all_supporting_models() -> None:
    a, b = _Native(), _Native()
    FallbackModel([a, b]).with_response_format({"k": "v"})
    assert a.bound == {"k": "v"} and b.bound == {"k": "v"}
