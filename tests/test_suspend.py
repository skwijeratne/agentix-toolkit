"""Suspendable human-in-the-loop: pause at a confirmation, persist, resume.

The model is a transcript-driven responder (not a fixed queue), so it behaves
like a real stateless model — once a tool result is present it answers, which
lets a *fresh* Agent resume a run persisted by another instance.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agentix import (
    Agent,
    AgentError,
    AgentEvents,
    AgentPolicy,
    MemoryStore,
    MockModel,
    ModelResponse,
    Role,
    TierGuard,
    ToolCall,
    tool,
)
from agentix.types import Message


@tool
def send_email(to: str) -> str:
    """Send an email."""
    return f"sent to {to}"


@tool
def read_inbox() -> str:
    """Read the inbox."""
    return "5 messages"


def _responder(messages: Sequence[Message]) -> ModelResponse:
    # Once any tool result is in the transcript, produce the final answer;
    # otherwise request the email tool.
    if any(m.role is Role.TOOL for m in messages):
        return ModelResponse(text="Done, email sent.")
    return ModelResponse(tool_calls=[ToolCall("send_email", {"to": "a@b.com"}, id="c1")])


def _agent(store: MemoryStore, responder=_responder, **kw: object) -> Agent:  # type: ignore[no-untyped-def]
    return Agent(
        model=MockModel(responder),
        system_prompt="assistant",
        tools=[send_email, read_inbox],
        guards=[TierGuard()],
        policy=AgentPolicy(confirm_first={"send_email"}),
        store=store,
        suspend_on_confirm=True,
        **kw,
    )


async def test_run_suspends_at_confirmation_without_side_effects() -> None:
    store = MemoryStore()
    out = await _agent(store).run("email a@b.com", run_id="r1")

    assert out.status == "suspended"
    assert out.reason == "awaiting_confirmation"
    assert [p.call.name for p in out.pending] == ["send_email"]
    assert out.pending[0].call.id == "c1"
    # The transcript pauses at the assistant tool-turn — no tool ran yet.
    assert out.transcript[-1].role is Role.ASSISTANT
    assert out.transcript[-1].meta["tool_calls"][0].name == "send_email"
    assert not any(m.role is Role.TOOL for m in out.transcript)


async def test_resume_approve_executes_and_completes() -> None:
    store = MemoryStore()
    agent = _agent(store)
    await agent.run("...", run_id="r1")

    out = await agent.resume("r1", decisions={"c1": True})

    assert out.status == "completed"
    assert out.answer == "Done, email sent."
    tool_msgs = [m for m in out.transcript if m.role is Role.TOOL]
    assert tool_msgs and "sent to a@b.com" in tool_msgs[0].content


async def test_resume_deny_declines_then_completes() -> None:
    store = MemoryStore()
    agent = _agent(store)
    await agent.run("...", run_id="r1")

    out = await agent.resume("r1", decisions={"c1": False})

    assert out.status == "completed"
    tool_msgs = [m for m in out.transcript if m.role is Role.TOOL]
    assert "declined" in tool_msgs[0].content.lower()


async def test_resume_without_decision_fails_closed() -> None:
    store = MemoryStore()
    agent = _agent(store)  # no confirm_fn wired
    await agent.run("...", run_id="r1")

    out = await agent.resume("r1")  # no decisions provided

    assert out.status == "completed"
    tool_msgs = [m for m in out.transcript if m.role is Role.TOOL]
    assert "declined" in tool_msgs[0].content.lower()


async def test_resume_on_a_fresh_agent_instance() -> None:
    # Simulates resuming in a different process: a brand-new Agent, same store.
    store = MemoryStore()
    await _agent(store).run("...", run_id="r1")

    fresh = _agent(store)
    out = await fresh.resume("r1", decisions={"c1": True})

    assert out.status == "completed"
    assert out.answer == "Done, email sent."


async def test_mixed_turn_runs_autook_and_gates_only_confirm() -> None:
    def responder(messages: Sequence[Message]) -> ModelResponse:
        if any(m.role is Role.TOOL for m in messages):
            return ModelResponse(text="all done")
        return ModelResponse(
            tool_calls=[
                ToolCall("read_inbox", {}, id="r0"),  # auto_ok
                ToolCall("send_email", {"to": "x"}, id="c1"),  # confirm_first
            ]
        )

    store = MemoryStore()
    agent = _agent(store, responder)
    out = await agent.run("...", run_id="r1")
    # Only the confirm-gated call is pending; nothing executed yet.
    assert [p.call.id for p in out.pending] == ["c1"]
    assert not any(m.role is Role.TOOL for m in out.transcript)

    done = await agent.resume("r1", decisions={"c1": True})
    assert done.status == "completed"
    names = [m.name for m in done.transcript if m.role is Role.TOOL]
    assert names == ["read_inbox", "send_email"]  # both ran, original order


async def test_suspend_requires_store_and_run_id() -> None:
    agent = Agent(
        model=MockModel(_responder),
        system_prompt="s",
        tools=[send_email],
        guards=[TierGuard()],
        policy=AgentPolicy(confirm_first={"send_email"}),
        suspend_on_confirm=True,  # but no store / run_id
    )
    with pytest.raises(AgentError, match="store and a run_id"):
        await agent.run("...")


async def test_on_suspend_event_fires() -> None:
    fired: list[object] = []
    store = MemoryStore()
    agent = _agent(store, events=AgentEvents(on_suspend=lambda o: fired.append(o)))
    await agent.run("...", run_id="r1")
    assert fired and getattr(fired[0], "status", None) == "suspended"
