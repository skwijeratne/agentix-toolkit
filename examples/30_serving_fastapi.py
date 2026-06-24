"""30 — Serve an agent over HTTP (FastAPI + Server-Sent Events).

Two endpoints show the two shapes of a web agent:

* `POST /chat` — **streams** the answer live as Server-Sent Events, using
  `agentix.serving.sse_response`. Open `/` in a browser to try it.
* `POST /task` + `POST /approve` — a run that **pauses for human approval**
  (`suspend_on_confirm`) returns `status="suspended"` with the pending action;
  `/approve` resumes it. This is the web-friendly human-in-the-loop pattern — the
  request returns immediately instead of blocking on the human.

The agents use `MockModel`, so no API key is needed. To run it you need the
serving extra plus an ASGI server:

    pip install "agentix-toolkit[serving]" fastapi uvicorn
    python examples/30_serving_fastapi.py     # then open http://127.0.0.1:8000/
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agentix import (
    Agent,
    AgentPolicy,
    MemoryStore,
    MockModel,
    ModelResponse,
    Role,
    TierGuard,
    ToolCall,
    tool,
)
from agentix.serving import outcome_to_payload, sse_response
from agentix.types import Message

app = FastAPI(title="agentix serving demo")


# ── streaming chat ──────────────────────────────────────────────────────────


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: 21°C, sunny"


def weather_model(messages: Sequence[Message]) -> ModelResponse:
    if any(m.role is Role.TOOL for m in messages):
        return ModelResponse(text="It looks like a lovely day — 21°C and sunny.")
    return ModelResponse(tool_calls=[ToolCall("get_weather", {"city": "Lisbon"}, id="c1")])


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatIn) -> object:
    agent = Agent(model=MockModel(weather_model), system_prompt="weather", tools=[get_weather])
    return sse_response(agent.stream(body.message))   # streams as Server-Sent Events


# ── human approval (suspend / resume) ───────────────────────────────────────

APPROVAL_STORE = MemoryStore()   # demo only; one shared run id


@tool
def send_email(to: str) -> str:
    """Send an email."""
    return f"sent to {to}"


def email_model(messages: Sequence[Message]) -> ModelResponse:
    if any(m.role is Role.TOOL for m in messages):
        return ModelResponse(text="Done — the email has been sent.")
    return ModelResponse(tool_calls=[ToolCall("send_email", {"to": "boss@example.com"}, id="e1")])


def approval_agent() -> Agent:
    return Agent(
        model=MockModel(email_model),
        system_prompt="assistant",
        tools=[send_email],
        guards=[TierGuard()],
        policy=AgentPolicy(confirm_first={"send_email"}),   # this tool needs a human "yes"
        store=APPROVAL_STORE,
        suspend_on_confirm=True,
    )


@app.post("/task")
async def task(body: ChatIn) -> dict:
    outcome = await approval_agent().run(body.message, run_id="task-1")
    return outcome_to_payload(outcome)   # status="suspended" with `pending` to approve


class ApproveIn(BaseModel):
    decisions: dict[str, bool]            # {call_id: True/False}


@app.post("/approve")
async def approve(body: ApproveIn) -> dict:
    outcome = await approval_agent().resume("task-1", decisions=body.decisions)
    return outcome_to_payload(outcome)


# ── a tiny browser client ───────────────────────────────────────────────────

_PAGE = """
<!doctype html><meta charset="utf-8"><title>agentix serving demo</title>
<h2>Streaming chat</h2>
<button onclick="chat()">Ask about the weather</button>
<pre id="out"></pre>
<script>
async function chat() {
  const out = document.getElementById('out'); out.textContent = '';
  const res = await fetch('/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({message: 'Weather in Lisbon?'})});
  const reader = res.body.getReader(), dec = new TextDecoder();
  for (;;) {
    const {value, done} = await reader.read(); if (done) break;
    for (const line of dec.decode(value).split('\\n')) {
      if (line.startsWith('data:')) {
        const e = JSON.parse(line.slice(5));
        if (e.type === 'answer') out.textContent += e.text;
        if (e.type === 'tool_started') out.textContent += '[using ' + e.tool + '…] ';
      }
    }
  }
}
</script>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
