# Streaming

By default, `agent.run(...)` waits until the agent is completely finished, then
hands you the answer. **Streaming** instead gives you the answer *as it's being
written* — the way you see ChatGPT type out a reply word by word. It makes apps
feel fast and lets you show tool activity live.

## How it works

Use `agent.stream(...)` and loop over the events it sends you. Each event tells you
something happening right now:

```python
from agentix import AnswerDelta, ToolStarted, ToolFinished, Done

async for event in agent.stream("Tell me about Lisbon."):
    if isinstance(event, AnswerDelta):
        print(event.text, end="", flush=True)     # a chunk of the answer
    elif isinstance(event, ToolStarted):
        print(f"\n[using {event.call.name}…]")
    elif isinstance(event, Done):
        print("\nfinished:", event.outcome.status)
```

The event types you'll see:

| Event | Meaning |
|---|---|
| `AnswerDelta` | a small piece of the answer text |
| `ToolStarted` | the agent is about to use a tool |
| `ToolFinished` | a tool just returned |
| `Done` | the run is over; carries the full final `outcome` |

→ Runnable example:
[`examples/09_streaming.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/09_streaming.py)

!!! note "One caveat"
    Because the answer is sent out piece by piece as it's written, a guard that
    edits the *final* answer (like redacting personal data) can't take back text
    that's already been streamed. If you need the user-facing text fully checked
    before it's shown, use `agent.run(...)` instead.
