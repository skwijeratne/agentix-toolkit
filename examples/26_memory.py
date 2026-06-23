"""26 — Cross-session memory.

The agent loop forgets everything between runs. A `Memory` gives it recall across
sessions: relevant records are fetched before a run and injected as system
context. agentix owns the *interface* (`Memory`); the storage is yours — a vector
DB, a search index, or the dependency-free `InMemoryMemory` shown here.

This demo runs two separate "sessions" (imagine different processes): the first
learns a fact and persists memory to a FileStore; the second loads it and recalls
the fact, so the model answers correctly without being told again.

Run:
    python examples/26_memory.py
"""

from __future__ import annotations

import asyncio
import tempfile

from agentix import (
    Agent,
    FileStore,
    InMemoryMemory,
    MockModel,
    ModelResponse,
    Role,
    tool,
)
from agentix.types import Message


@tool
def note_preference(fact: str) -> str:
    """Record a durable fact about the user."""
    return f"noted: {fact}"


def recall_aware_model(messages: list[Message]) -> ModelResponse:
    # A stand-in for a real model: read the recalled memory out of the system
    # context and answer from it.
    context = " ".join(m.content for m in messages if m.role is Role.SYSTEM)
    name = "Sanjaya" if "Sanjaya" in context else "(I don't know yet)"
    return ModelResponse(text=f"Your name is {name}.")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)

        # --- Session 1: learn a fact and persist memory. ---
        mem = InMemoryMemory()
        await mem.write("The user's name is Sanjaya.", metadata={"kind": "profile"})
        await mem.write("The user prefers concise answers.", metadata={"kind": "profile"})
        await store.save("user-memory", {"records": mem.dump()})
        print("session 1: stored", len(mem.records), "memories")

        # --- Session 2: a fresh process. Load memory, ask, recall kicks in. ---
        state = await store.load("user-memory")
        assert state is not None
        loaded = InMemoryMemory.load(state["records"])

        agent = Agent(
            model=MockModel(recall_aware_model),
            system_prompt="You are a helpful assistant.",
            memory=loaded,
            tools=[note_preference],
        )
        outcome = await agent.run("Remind me — what's my name?")
        print("session 2 answer:", outcome.answer)

        # The recalled record was injected as a system message before the model ran:
        recalled = [
            m.content
            for m in outcome.transcript
            if m.role is Role.SYSTEM and "recalled from memory" in m.content
        ]
        print("injected context:\n ", recalled[0].replace("\n", "\n  ") if recalled else "(none)")


if __name__ == "__main__":
    asyncio.run(main())
