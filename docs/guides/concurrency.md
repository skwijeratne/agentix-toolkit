# Running many agents

Sometimes you need to run lots of agents at once — process a thousand support
tickets, summarize a batch of documents, fan out a job across many workers. Doing
that naively can overwhelm your machine or blow past a provider's rate limits.
agentix gives you simple tools to run many agents **safely**.

## Bounded fan-out

`bounded_gather` runs many async tasks at once, but never more than a set number at
a time. It's like a queue at a busy counter: everyone gets served, but only so many
at once.

```python
from agentix import bounded_gather

async def handle(ticket):
    return await agent.run(ticket)

results = await bounded_gather(
    [handle(t) for t in tickets],
    limit=10,                       # at most 10 running at once
)
```

## Sharing a limit across a fleet

If you have several agents that all call the same provider, you want one shared
speed limit across all of them — not a separate one each. A `Limiter` does that:
create one and pass it to every agent.

```python
from agentix import Agent, Limiter

shared = Limiter(20)     # at most 20 model calls in flight across everything
agent_a = Agent(model=m, system_prompt="...", model_limiter=shared)
agent_b = Agent(model=m, system_prompt="...", model_limiter=shared)
```

This keeps you under provider rate limits even when many agents run together.

→ Runnable example:
[`examples/10_concurrency.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/10_concurrency.py)
