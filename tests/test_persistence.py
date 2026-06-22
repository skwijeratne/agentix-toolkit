import json
import tempfile
from pathlib import Path

from agentix import (
    Agent,
    AgentOutcome,
    FileStore,
    MemoryStore,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    message_from_dict,
    message_to_dict,
    outcome_from_dict,
    outcome_to_dict,
    tool,
)

# ── serde codec ──────────────────────────────────────────────────────────


def test_message_round_trip_with_tool_calls() -> None:
    msg = Message(
        Role.ASSISTANT,
        "calling",
        trusted=True,
        meta={"tool_calls": [ToolCall("add", {"a": 1, "b": 2}, id="c1")]},
    )
    d = message_to_dict(msg)
    # JSON-serializable end to end
    d = json.loads(json.dumps(d))
    back = message_from_dict(d)
    assert back.role == Role.ASSISTANT
    assert back.trusted is True
    call = back.meta["tool_calls"][0]
    assert isinstance(call, ToolCall)
    assert call.name == "add" and call.args == {"a": 1, "b": 2} and call.id == "c1"


def test_tool_message_round_trip() -> None:
    msg = Message(Role.TOOL, "result", name="add", meta={"call_id": "c1", "ok": True})
    back = message_from_dict(json.loads(json.dumps(message_to_dict(msg))))
    assert back.role == Role.TOOL
    assert back.trusted is False
    assert back.meta == {"call_id": "c1", "ok": True}


def test_outcome_round_trip() -> None:
    outcome = AgentOutcome(
        status="completed",
        answer="hi",
        steps=2,
        tokens_used=42,
        transcript=[Message(Role.USER, "q", trusted=True)],
    )
    back = outcome_from_dict(json.loads(json.dumps(outcome_to_dict(outcome))))
    assert back.status == "completed"
    assert back.answer == "hi"
    assert back.steps == 2
    assert back.tokens_used == 42
    assert back.transcript[0].content == "q"


# ── stores ───────────────────────────────────────────────────────────────


async def test_memory_store_save_load() -> None:
    store = MemoryStore()
    assert await store.load("x") is None
    await store.save("x", {"a": 1})
    assert await store.load("x") == {"a": 1}


async def test_memory_store_is_isolated_from_caller_mutation() -> None:
    store = MemoryStore()
    state = {"a": [1]}
    await store.save("x", state)
    state["a"].append(2)  # mutate after save
    assert (await store.load("x"))["a"] == [1]  # stored copy unaffected


async def test_file_store_save_load() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = FileStore(d)
        await store.save("run-1", {"steps": 3})
        assert (Path(d) / "run-1.json").exists()
        assert await store.load("run-1") == {"steps": 3}
        assert await store.load("missing") is None


async def test_file_store_sanitizes_run_id() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = FileStore(d)
        await store.save("a/b", {"k": 1})  # no traversal/escape
        assert await store.load("a/b") == {"k": 1}
        assert (Path(d) / "a_b.json").exists()


# ── checkpoint + resume through the Agent ────────────────────────────────


@tool
def add(a: int, b: int) -> int:
    """Add."""
    return a + b


async def test_run_checkpoints_each_step() -> None:
    store = MemoryStore()
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
            ModelResponse(text="done"),
        ]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[add], store=store)
    await agent.run("go", run_id="r1")

    saved = await store.load("r1")
    assert saved is not None
    assert saved["steps"] >= 1
    assert any(m["role"] == "tool" for m in saved["messages"])


async def test_resume_continues_from_checkpoint() -> None:
    store = MemoryStore()

    # Simulate a run that was interrupted after step 1: the transcript holds the
    # user turn, an assistant tool-call turn, and the tool result.
    interrupted = [
        Message(Role.SYSTEM, "sys", trusted=True),
        Message(Role.USER, "2+3?", trusted=True),
        Message(Role.ASSISTANT, "", trusted=True,
                meta={"tool_calls": [ToolCall("add", {"a": 2, "b": 3}, id="c1")]}),
        Message(Role.TOOL, "5", name="add", meta={"call_id": "c1", "ok": True}),
    ]
    await store.save("r2", {
        "run_id": "r2",
        "steps": 1,
        "tokens_used": 10,
        "messages": [message_to_dict(m) for m in interrupted],
    })

    # The model only needs to produce the final answer to finish.
    agent = Agent(model=MockModel([ModelResponse(text="The answer is 5.")]),
                  system_prompt="sys", tools=[add], store=store)
    outcome = await agent.resume("r2")

    assert outcome.status == "completed"
    assert outcome.answer == "The answer is 5."
    assert outcome.steps == 2  # continued from step 1
    assert outcome.tokens_used == 10  # carried forward


async def test_resume_without_store_raises() -> None:
    agent = Agent(model=MockModel([]), system_prompt="sys")
    try:
        await agent.resume("nope")
        raise AssertionError("expected AgentError")
    except Exception as e:
        assert "store" in str(e).lower()


async def test_resume_unknown_run_raises() -> None:
    agent = Agent(model=MockModel([]), system_prompt="sys", store=MemoryStore())
    try:
        await agent.resume("ghost")
        raise AssertionError("expected AgentError")
    except Exception as e:
        assert "ghost" in str(e)
