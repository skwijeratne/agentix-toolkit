# Saving & resuming runs

A long agent run can be interrupted — a crash, a deploy, a timeout, or just a
user who closes the tab. **Persistence** lets you save a run's progress and pick it
back up later, instead of starting over.

## Saving progress automatically

Give the agent a **store** (somewhere to save state) and a **run id** (a name for
this run). After every step, the agent saves a checkpoint:

```python
from agentix import Agent, FileStore

agent = Agent(model=m, system_prompt="...", tools=[...], store=FileStore("./runs"))
outcome = await agent.run("Big multi-step task…", run_id="job-123")
```

`FileStore` saves to disk. `MemoryStore` keeps things in memory (handy for tests).
You can also write your own store to save anywhere (a database, cloud storage) —
it's a small interface.

## Resuming

If a run stops partway, resume it later with the same id:

```python
outcome = await agent.resume("job-123")
```

The agent reloads the conversation and continues from where it left off. This is
also how **human approval in web apps** works — a run pauses, gets saved, and
resumes once someone approves. See
**[Cost, budgets & human approval](cost-and-control.md#pausing-for-a-human-without-blocking)**.

→ Runnable example:
[`examples/08_persistence.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/08_persistence.py)
