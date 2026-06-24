# Memory & context

Two related ideas about what the agent "remembers":

- **Context** is the current conversation — everything in *this* run.
- **Memory** is what carries over *between* runs and sessions.

## Keeping the conversation from getting too big

Every AI model can only read so much text at once. That limit is called its
**context window**, and it's measured in **tokens** (a token is roughly ¾ of a
word). A long agent run — lots of tool calls, lots of results — slowly fills that
window, and eventually it overflows and the model errors out.

`FitContextWindow` prevents that. It keeps the conversation under a token budget by
dropping the **oldest** exchanges, while always keeping the original task and never
splitting a tool call from its result:

```python
from agentix import Agent, FitContextWindow

agent = Agent(
    model=m,
    system_prompt="...",
    tools=[...],
    context_strategy=FitContextWindow(max_tokens=180_000, reserve_tokens=4_000),
)
```

`reserve_tokens` leaves room for the model's reply. By default the token count is a
fast estimate; for an exact count you can plug in a real tokenizer (like
`tiktoken`) — see the example.

→ Runnable example:
[`examples/25_token_context.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/25_token_context.py)

## Remembering across sessions

By default, an agent forgets everything once a run ends. **Memory** lets it recall
useful things later — "the user prefers metric units", "last week we decided X".

agentix gives you the *interface*; you bring the storage (a search index, a vector
database, or just a file). A simple keyword-based memory is included so you can
start immediately:

```python
from agentix import Agent, InMemoryMemory

memory = InMemoryMemory()
await memory.write("The user's name is Sanjaya.")

agent = Agent(model=m, system_prompt="...", memory=memory)
# Before each run, relevant memories are looked up and added to the agent's context.
```

Set `remember_exchange=True` to automatically save each finished conversation back
into memory.

!!! warning "Only store trusted content in memory"
    Memories are added to the agent as **trusted** instructions. Don't store raw,
    unchecked tool output there — that would reopen the prompt-injection door the
    [Security model](../security.md) closes.

→ Runnable example:
[`examples/26_memory.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/26_memory.py)
