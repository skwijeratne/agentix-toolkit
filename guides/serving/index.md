# Serving an agent over HTTP

You've built an agent — now you want to put it behind a web address so a browser or app can talk to it. agentix gives you small helpers to turn an agent into a **streaming HTTP endpoint** without writing the plumbing yourself.

Install the extra:

```
pip install "agentix-toolkit[serving]"
```

It works with [FastAPI](https://fastapi.tiangolo.com/) or [Starlette](https://www.starlette.io/). The web dependency is optional — it's only needed when you actually serve.

## Streaming the answer live

For a chat-style UI you want the answer to appear as it's written, not all at once after a long wait. The standard browser way to receive a live stream is **Server-Sent Events (SSE)** — the server keeps one connection open and pushes updates as they happen.

`sse_response` takes an agent's event stream and turns it into exactly that:

```
from fastapi import FastAPI
from pydantic import BaseModel
from agentix.serving import sse_response

app = FastAPI()

class ChatIn(BaseModel):
    message: str

@app.post("/chat")
async def chat(body: ChatIn):
    agent = build_agent()                      # your Agent
    return sse_response(agent.stream(body.message))
```

That's the whole server side. Each event the agent produces is sent to the browser as it happens, tagged by type:

| Event type      | What it carries                       |
| --------------- | ------------------------------------- |
| `answer`        | a chunk of the answer text            |
| `tool_started`  | the agent is calling a tool           |
| `tool_finished` | a tool returned                       |
| `done`          | the run is over, with a small summary |

Prefer plain newline-delimited JSON instead of SSE? Use `ndjson_response` — same idea, one JSON object per line.

## Pausing for human approval (web-friendly)

A streaming connection can't wait around for a human to click "approve" — they might take minutes. For actions that need a person's "yes", use the **pause and resume** pattern instead (see [Cost, budgets & human approval](https://skwijeratne.github.io/agentix-toolkit/guides/cost-and-control/#pausing-for-a-human-without-blocking)).

The request returns immediately with `status="suspended"` and the pending action. `outcome_to_payload` turns that into clean JSON for your response:

```
from agentix.serving import outcome_to_payload

@app.post("/task")
async def task(body: ChatIn):
    outcome = await agent.run(body.message, run_id="task-1")
    return outcome_to_payload(outcome)         # includes `pending` when suspended

@app.post("/approve")
async def approve(body: ApproveIn):            # {"decisions": {"call-id": true}}
    outcome = await agent.resume("task-1", decisions=body.decisions)
    return outcome_to_payload(outcome)
```

## Not using FastAPI?

The serialization itself has no dependencies. `sse_events(agent.stream(...))` and `ndjson_events(...)` are plain async generators of text you can feed into *any* framework's streaming response. `sse_response`/`ndjson_response` are just thin wrappers that add the right headers for FastAPI/Starlette.

→ Runnable example (a full app + a tiny browser client): [`examples/30_serving_fastapi.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/30_serving_fastapi.py)
