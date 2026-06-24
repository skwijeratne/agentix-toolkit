"""28 — Rate-limit-aware retries.

`RetryModel` retries transient errors, but instead of blind exponential backoff
it honors a provider's **Retry-After** when the error carries one (capped at
`max_sleep`). Wire `on_retry` to surface the wait. Point `retry_on` at your
provider's rate-limit/transient error types in real use.

Run:
    python examples/28_rate_limit.py
"""

from __future__ import annotations

import asyncio

from agentix import RetryModel
from agentix.types import ModelResponse


class RateLimited(Exception):
    """Stands in for a provider 429 carrying a Retry-After (seconds)."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("429 Too Many Requests")
        self.retry_after = retry_after


class FlakyModel:
    """Rate-limited for the first two calls, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, messages, *, tools=()):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= 2:
            raise RateLimited(retry_after=0.2 * self.calls)  # server says "wait N s"
        return ModelResponse(text="Here is your answer.")


async def main() -> None:
    def on_retry(exc: BaseException, delay: float, attempt: int) -> None:
        print(f"  attempt {attempt}: {exc} -> waiting {delay}s (server-requested)")

    model = RetryModel(
        FlakyModel(),
        retries=3,
        retry_on=(RateLimited,),  # narrow this to your SDK's error types
        max_sleep=30,             # never wait longer than this, even if asked to
        on_retry=on_retry,
    )

    print("calling (will be rate-limited twice)...")
    response = await model([])  # called directly; normally Agent(model=model)
    print("succeeded:", response.text)


if __name__ == "__main__":
    asyncio.run(main())
