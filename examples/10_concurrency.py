"""10 — Running many agents safely.

agentix imposes no concurrency cap by default. Two primitives let you add one:

  * `bounded_gather(aws, limit=N)` — run many agent runs with at most N alive at
    once. Bounds provider load *and* memory (fewer transcripts in flight). Best
    for batch jobs.
  * `Limiter(N)` — a shared async semaphore. Inject it (`model_limiter=`) to cap
    concurrent *model calls* across a whole fleet of agents. Best for a server
    where each request spawns its own run and there's no single gather point.

Also relevant (automatic, nothing to configure): synchronous tool functions run
in a worker thread, so a blocking tool can't stall the event loop and starve the
other agents.

Run:
    PYTHONPATH=src python examples/10_concurrency.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, Limiter, MockModel, ModelResponse, bounded_gather


def make_agent(limiter: Limiter) -> Agent:
    # Each "request" gets its own agent; they share one limiter.
    return Agent(
        model=MockModel([ModelResponse(text="handled")]),
        system_prompt="You are a worker.",
        model_limiter=limiter,
    )


async def main() -> None:
    # Cap concurrent model calls across the fleet at 5, regardless of how many
    # runs we launch.
    limiter = Limiter(5)
    requests = [f"task {i}" for i in range(50)]

    # bounded_gather caps how many runs are alive at once (here, 10).
    outcomes = await bounded_gather(
        [make_agent(limiter).run(req) for req in requests],
        limit=10,
    )

    print(f"ran {len(outcomes)} agents")
    print("all completed:", all(o.status == "completed" for o in outcomes))
    print("first answer:", outcomes[0].answer)


if __name__ == "__main__":
    asyncio.run(main())
