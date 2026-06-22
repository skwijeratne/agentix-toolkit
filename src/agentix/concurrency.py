"""Concurrency control for running agents at scale.

At "thousands of agents" the scarce resources are the provider connection pool,
provider rate limits, and memory (one transcript per in-flight run). agentix
imposes no cap by default; these primitives let you add one.

  * :class:`Limiter` — a shareable async semaphore. Inject it into an ``Agent``
    (``model_limiter=``) to bound concurrent **model calls** across a whole
    fleet of agents — the right tool for a web server where each request spawns
    its own ``run()`` and there's no single place to gather.
  * :func:`bounded_gather` — run many awaitables with a hard concurrency cap.
    The right tool for batch jobs, where it also bounds peak memory by limiting
    how many runs are alive at once.

Both bind to the event loop on first use — create and share them **within one
running loop** (don't reuse one Limiter across separate ``asyncio.run`` calls).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import TypeVar

_T = TypeVar("_T")


class Limiter(AbstractAsyncContextManager["Limiter"]):
    """An async concurrency limiter usable as an ``async with`` context.

    Share one instance across many agents to cap their combined in-flight
    operations (e.g. concurrent provider requests)::

        limiter = Limiter(50)
        agent = Agent(model=..., system_prompt=..., model_limiter=limiter)
    """

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> Limiter:
        await self._sem.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._sem.release()


async def bounded_gather(
    aws: Sequence[Awaitable[_T]], *, limit: int
) -> list[_T]:
    """Like :func:`asyncio.gather`, but at most ``limit`` awaitables run at once.

    Results are returned in the original order. Use this to launch many agent
    runs without flooding the provider or holding every transcript in memory::

        outcomes = await bounded_gather(
            [agent.run(q) for q in questions], limit=20
        )
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    sem = asyncio.Semaphore(limit)

    async def _run(aw: Awaitable[_T]) -> _T:
        async with sem:
            return await aw

    return await asyncio.gather(*(_run(a) for a in aws))
