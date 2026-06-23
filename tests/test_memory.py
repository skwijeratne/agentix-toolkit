"""Pluggable memory: the InMemoryMemory default + Agent recall/remember wiring."""

from __future__ import annotations

from agentix import (
    Agent,
    AgentPolicy,
    InMemoryMemory,
    Memory,
    MemoryRecord,
    MockModel,
    ModelResponse,
    Role,
    TextPart,
    ToolCall,
    tool,
)


@tool
def noop() -> str:
    """A no-op tool."""
    return "ok"


# ── InMemoryMemory ─────────────────────────────────────────────────────────


def test_satisfies_the_memory_protocol() -> None:
    assert isinstance(InMemoryMemory(), Memory)


async def test_recall_ranks_by_keyword_overlap_and_sets_score() -> None:
    mem = InMemoryMemory(
        [MemoryRecord("apple banana cherry"), MemoryRecord("banana only")]
    )
    out = await mem.recall("banana cherry")
    assert out[0].content == "apple banana cherry"  # overlap 2 beats overlap 1
    assert out[0].score == 2.0
    assert out[1].content == "banana only"


async def test_recall_limit_empty_query_and_no_match() -> None:
    mem = InMemoryMemory([MemoryRecord(f"fact about cats {i}") for i in range(10)])
    assert len(await mem.recall("cats", limit=3)) == 3
    assert await mem.recall("") == []  # no query tokens
    assert await InMemoryMemory([MemoryRecord("apples")]).recall("zebra") == []


async def test_write_and_dump_load_round_trip() -> None:
    mem = InMemoryMemory()
    await mem.write("a fact", metadata={"k": "v"})
    assert mem.records[0].content == "a fact"

    restored = InMemoryMemory.load(mem.dump())
    assert restored.records[0].content == "a fact"
    assert restored.records[0].metadata == {"k": "v"}


# ── Agent integration ──────────────────────────────────────────────────────


async def test_recalled_memory_is_injected_as_system_context() -> None:
    mem = InMemoryMemory([MemoryRecord("The user prefers metric units.")])
    agent = Agent(
        model=MockModel([ModelResponse(text="ok")]),
        system_prompt="sys",
        memory=mem,
    )
    out = await agent.run("convert these to my preferred units")

    sys_blocks = [m.content for m in out.transcript if m.role is Role.SYSTEM]
    assert any("metric units" in c for c in sys_blocks)
    assert len(sys_blocks) == 2  # the prompt + the recalled-memory block


async def test_no_memory_means_a_single_system_message() -> None:
    agent = Agent(model=MockModel([ModelResponse(text="ok")]), system_prompt="sys")
    out = await agent.run("hi")
    assert len([m for m in out.transcript if m.role is Role.SYSTEM]) == 1


async def test_recall_uses_the_query_text_of_multimodal_input() -> None:
    mem = InMemoryMemory([MemoryRecord("Paris is the capital of France.")])
    agent = Agent(
        model=MockModel([ModelResponse(text="ok")]), system_prompt="s", memory=mem
    )
    out = await agent.run([TextPart("Tell me about Paris")])
    assert any(
        "capital of France" in m.content for m in out.transcript if m.role is Role.SYSTEM
    )


async def test_remember_exchange_persists_a_completed_run() -> None:
    mem = InMemoryMemory()
    agent = Agent(
        model=MockModel([ModelResponse(text="42")]),
        system_prompt="s",
        memory=mem,
        remember_exchange=True,
    )
    await agent.run("what is the answer?")

    assert len(mem.records) == 1
    assert "42" in mem.records[0].content
    assert "what is the answer?" in mem.records[0].content
    assert mem.records[0].metadata.get("kind") == "exchange"


async def test_remember_exchange_skips_non_completed_runs() -> None:
    mem = InMemoryMemory()

    def responder(_messages: object) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall("noop", {}, id="c1")])

    agent = Agent(
        model=MockModel(responder),
        system_prompt="s",
        tools=[noop],
        policy=AgentPolicy(max_steps=1),  # never finalizes -> aborted
        memory=mem,
        remember_exchange=True,
    )
    out = await agent.run("x")
    assert out.status == "aborted"
    assert mem.records == []
