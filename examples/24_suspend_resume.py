"""24 — Suspendable human-in-the-loop (pause → persist → resume).

A web or serverless agent can't block a request coroutine waiting for a human to
click "approve" minutes later. With `suspend_on_confirm=True`, when a tool needs
confirmation the loop *checkpoints to the store and returns* `status="suspended"`
with the pending approvals — the request can end. A later request (even in a new
process) calls `resume(run_id, decisions=...)` to approve/deny and continue.

This demo uses a scripted model and a FileStore, and resumes on a brand-new
Agent instance to prove the paused state lives entirely in the store.

Run:
    python examples/24_suspend_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Sequence

from agentix import (
    Agent,
    AgentPolicy,
    FileStore,
    MockModel,
    ModelResponse,
    Role,
    TierGuard,
    ToolCall,
    tool,
)
from agentix.types import Message


@tool
def wire_funds(to: str, amount: int) -> str:
    """Wire money to an account."""
    return f"wired ${amount} to {to}"


def model(messages: Sequence[Message]) -> ModelResponse:
    # Stateless, like a real model: answer once the tool result is present.
    if any(m.role is Role.TOOL for m in messages):
        return ModelResponse(text="Done — the transfer is complete.")
    call = ToolCall("wire_funds", {"to": "acct-42", "amount": 9000}, id="c1")
    return ModelResponse(tool_calls=[call])


def build_agent(store: FileStore) -> Agent:
    return Agent(
        model=MockModel(model),
        system_prompt="You are a finance assistant.",
        tools=[wire_funds],
        guards=[TierGuard()],
        policy=AgentPolicy(confirm_first={"wire_funds"}),  # gate the risky tool
        store=store,
        suspend_on_confirm=True,
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)

        # --- Request 1: start the run; it pauses for approval and returns. ---
        first = await build_agent(store).run("Pay invoice 42", run_id="run-1")
        print("after request 1:", first.status)
        for p in first.pending:
            c = p.call
            print(f"  awaiting approval: {c.name}({c.args}) — {p.reason}")

        # ... the HTTP handler returns here; nothing is blocked. The human
        # reviews and approves out-of-band. Later, a *new* process handles the
        # approval callback: ---

        # --- Request 2: a fresh Agent (same store) resumes with the decision. ---
        approved = await build_agent(store).resume("run-1", decisions={"c1": True})
        print("after request 2:", approved.status, "->", approved.answer)

        # Denying instead would have declined the tool and still completed:
        denied = await build_agent(store).resume("run-1", decisions={"c1": False})
        print("if denied:", denied.status, "->", denied.answer)


if __name__ == "__main__":
    asyncio.run(main())
