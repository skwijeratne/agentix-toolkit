from agentix import (
    Agent,
    AgentEvents,
    AgentPolicy,
    ContextStrategy,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    TrimRounds,
    TruncateToolOutputs,
    tool,
)
from agentix.context import _split


def _round(idx: int) -> list[Message]:
    """An assistant tool-turn + its tool result."""
    return [
        Message(Role.ASSISTANT, "", trusted=True,
                meta={"tool_calls": [ToolCall("ping", {}, id=f"c{idx}")]}),
        Message(Role.TOOL, f"pong{idx}", name="ping", meta={"call_id": f"c{idx}", "ok": True}),
    ]


def _transcript(n_rounds: int) -> list[Message]:
    msgs = [Message(Role.SYSTEM, "sys", trusted=True), Message(Role.USER, "task", trusted=True)]
    for i in range(n_rounds):
        msgs.extend(_round(i))
    return msgs


# ── _split / TrimRounds ──────────────────────────────────────────────────


def test_split_separates_head_task_rounds() -> None:
    head, task, rounds = _split(_transcript(3))
    assert [m.role for m in head] == [Role.SYSTEM]
    assert [m.role for m in task] == [Role.USER]
    assert len(rounds) == 3
    # Each round is a complete assistant + tool pair.
    for r in rounds:
        assert r[0].role is Role.ASSISTANT and r[1].role is Role.TOOL


async def test_trim_rounds_keeps_recent_and_preserves_pairing() -> None:
    out = await TrimRounds(2).compact(_transcript(4))
    # system + task + 2 rounds * 2 msgs = 6
    assert len(out) == 6
    assert out[0].role is Role.SYSTEM
    assert out[1].role is Role.USER
    # kept the LAST two rounds (pong2, pong3), pairing intact
    assert out[2].role is Role.ASSISTANT and "tool_calls" in out[2].meta
    assert out[3].content == "pong2"
    assert out[5].content == "pong3"
    # every assistant tool-turn is immediately followed by its tool result
    for i, m in enumerate(out):
        if m.role is Role.ASSISTANT and m.meta.get("tool_calls"):
            assert out[i + 1].role is Role.TOOL


async def test_trim_rounds_noop_under_threshold_returns_same_object() -> None:
    msgs = _transcript(2)
    out = await TrimRounds(5).compact(msgs)
    assert out is msgs  # unchanged identity -> loop skips the on_compact event


def test_trim_rounds_rejects_bad_size() -> None:
    try:
        TrimRounds(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ── TruncateToolOutputs ──────────────────────────────────────────────────


async def test_truncate_clips_long_tool_output() -> None:
    msgs = [
        Message(Role.SYSTEM, "sys", trusted=True),
        Message(Role.TOOL, "x" * 100, name="t", meta={"ok": True}),
        Message(Role.ASSISTANT, "short", trusted=True),
    ]
    out = await TruncateToolOutputs(10).compact(msgs)
    assert out[1].content == "x" * 10 + "...[truncated]"
    assert out[0].content == "sys"        # non-tool untouched
    assert out[2].content == "short"      # non-tool untouched


async def test_truncate_is_idempotent() -> None:
    strat = TruncateToolOutputs(10)
    once = await strat.compact([Message(Role.TOOL, "y" * 100, meta={})])
    twice = await strat.compact(once)
    assert twice is once  # nothing left to clip -> same object
    assert once[0].content.count("...[truncated]") == 1


async def test_truncate_noop_returns_same_object() -> None:
    msgs = [Message(Role.TOOL, "short", meta={})]
    assert await TruncateToolOutputs(100).compact(msgs) is msgs


# ── through the Agent loop ───────────────────────────────────────────────


@tool
def ping() -> str:
    """Ping."""
    return "pong"


async def test_trim_rounds_bounds_transcript_in_loop_and_emits_event() -> None:
    compactions: list[tuple[int, int]] = []
    events = AgentEvents(on_compact=lambda b, a: compactions.append((b, a)))

    # 4 tool rounds, then a final answer.
    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("ping", {}, id=f"c{i}")]) for i in range(4)]
        + [ModelResponse(text="done")]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[ping],
        policy=AgentPolicy(max_steps=10),
        context_strategy=TrimRounds(2),
        events=events,
    )
    outcome = await agent.run("go")

    assert outcome.status == "completed"
    assert outcome.answer == "done"
    # Without compaction the transcript would be 2 + 4*2 + 1 = 11; trimmed it's bounded.
    assert len(outcome.transcript) < 11
    assert compactions  # on_compact fired at least once


async def test_truncate_tool_output_in_loop() -> None:
    @tool
    def big() -> str:
        """Return a big blob."""
        return "Z" * 5000

    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("big", {}, id="c1")]), ModelResponse(text="ok")]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[big],
        context_strategy=TruncateToolOutputs(100),
    )
    outcome = await agent.run("go")
    tool_msg = next(m for m in outcome.transcript if m.role is Role.TOOL)
    assert len(tool_msg.content) < 5000
    assert tool_msg.content.endswith("...[truncated]")


async def test_no_strategy_keeps_full_transcript() -> None:
    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("ping", {}, id="c1")]), ModelResponse(text="done")]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[ping])  # no strategy
    outcome = await agent.run("go")
    # system, user, assistant(toolcall), tool, assistant(final) = 5
    assert len(outcome.transcript) == 5


async def test_custom_strategy_base_is_noop() -> None:
    msgs = _transcript(3)
    assert await ContextStrategy().compact(msgs) is msgs
