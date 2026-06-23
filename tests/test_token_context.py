"""Token-accurate context: counters + the FitContextWindow strategy."""

from __future__ import annotations

import json

import pytest

from agentix import (
    Agent,
    AgentEvents,
    AgentPolicy,
    FitContextWindow,
    HeuristicTokenCounter,
    ImagePart,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    approx_token_counter,
    count_message_tokens,
    count_tokens,
    tool,
)

LEN = len  # a "1 token per character" counter for exact arithmetic


@tool
def noop() -> str:
    """A no-op tool."""
    return "ok"


# ── counters ──────────────────────────────────────────────────────────────


def test_heuristic_counter_default_and_configurable() -> None:
    assert HeuristicTokenCounter()("abcd" * 4) == 4  # 16 chars / 4
    assert HeuristicTokenCounter(chars_per_token=2.0)("abcd") == 2
    assert approx_token_counter("") == 0
    with pytest.raises(ValueError):
        HeuristicTokenCounter(chars_per_token=0)


def test_count_message_tokens_text_overhead_and_tool_calls() -> None:
    msg = Message(Role.USER, "hello")
    assert count_message_tokens(msg, LEN, per_message_overhead=0) == 5
    assert count_message_tokens(msg, LEN, per_message_overhead=3) == 8

    asst = Message(
        Role.ASSISTANT,
        "",
        meta={"tool_calls": [ToolCall("t", {"a": 1}, id="c1")]},
    )
    expected = LEN("t") + LEN(json.dumps({"a": 1}))  # name + serialized args
    assert count_message_tokens(asst, LEN, per_message_overhead=0) == expected


def test_count_message_tokens_media() -> None:
    msg = Message(Role.USER, [ImagePart.from_base64("aGk=", "image/png")])
    got = count_message_tokens(msg, LEN, per_message_overhead=0, tokens_per_media=600)
    assert got == 600  # one media part, no text


def test_count_tokens_sums_transcript() -> None:
    msgs = [Message(Role.SYSTEM, "sys"), Message(Role.USER, "hello")]
    assert count_tokens(msgs, LEN, per_message_overhead=0) == 3 + 5


# ── FitContextWindow ──────────────────────────────────────────────────────


def _round(i: int) -> list[Message]:
    """An assistant tool-turn (cost 3 with LEN: 't' + '{}') + its tool result
    (cost 20: 'R<i>' * 10 == 20 chars)."""
    asst = Message(Role.ASSISTANT, "", meta={"tool_calls": [ToolCall("t", {}, id=f"c{i}")]})
    result = Message(Role.TOOL, f"R{i}" * 10, name="t", meta={"call_id": f"c{i}", "ok": True})
    return [asst, result]


def _transcript(n_rounds: int) -> list[Message]:
    msgs = [Message(Role.SYSTEM, "sys"), Message(Role.USER, "task")]  # fixed = 3 + 4
    for i in range(1, n_rounds + 1):
        msgs += _round(i)
    return msgs


def _fit(max_tokens: int, **kw: object) -> FitContextWindow:
    return FitContextWindow(max_tokens, LEN, per_message_overhead=0, **kw)  # type: ignore[arg-type]


async def test_returns_same_object_when_everything_fits() -> None:
    msgs = _transcript(3)
    out = await _fit(10_000).compact(msgs)
    assert out is msgs  # unchanged identity => loop skips on_compact


async def test_drops_oldest_rounds_to_fit_budget() -> None:
    # fixed=7, each round=23. all=7+69=76; last two=53; last one=30.
    msgs = _transcript(3)
    out = await _fit(60).compact(msgs)

    assert out is not msgs
    tool_payloads = [m.content for m in out if m.role is Role.TOOL]
    assert tool_payloads == ["R2" * 10, "R3" * 10]  # oldest (R1) dropped, order kept
    assert out[0].role is Role.SYSTEM and out[1].role is Role.USER  # prefix kept


async def test_keeps_at_least_the_latest_round_even_if_over_budget() -> None:
    msgs = _transcript(3)
    out = await _fit(5).compact(msgs)  # budget below even the fixed prefix
    tool_payloads = [m.content for m in out if m.role is Role.TOOL]
    assert tool_payloads == ["R3" * 10]  # only the most recent round survives


async def test_reserve_tokens_shrinks_the_budget() -> None:
    msgs = _transcript(3)
    out = await _fit(60, reserve_tokens=10).compact(msgs)  # effective budget 50
    tool_payloads = [m.content for m in out if m.role is Role.TOOL]
    assert tool_payloads == ["R3" * 10]


async def test_kept_messages_preserve_tool_pairing() -> None:
    out = await _fit(60).compact(_transcript(3))
    # Every TOOL result is immediately preceded by its assistant tool-turn.
    for i, m in enumerate(out):
        if m.role is Role.TOOL:
            assert out[i - 1].role is Role.ASSISTANT
            assert out[i - 1].meta["tool_calls"][0].id == m.meta["call_id"]


def test_validation() -> None:
    with pytest.raises(ValueError):
        FitContextWindow(0)
    with pytest.raises(ValueError):
        FitContextWindow(100, reserve_tokens=-1)


async def test_integration_keeps_loop_transcript_bounded() -> None:
    # A model that always calls a tool; the loop trims before each step, so the
    # working transcript stays bounded instead of growing every round.
    def responder(messages: list[Message]) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall("noop", {}, id=f"c{len(messages)}")])

    compactions: list[tuple[int, int]] = []
    agent = Agent(
        model=MockModel(responder),
        system_prompt="sys",
        tools=[noop],
        policy=AgentPolicy(max_steps=8),
        context_strategy=_fit(40),  # holds ~4 rounds of 8 tokens each
        events=AgentEvents(on_compact=lambda b, a: compactions.append((b, a))),
    )
    outcome = await agent.run("task")

    assert outcome.status == "aborted"  # hit max_steps (model never finalizes)
    assert compactions  # trimming actually happened
    # The transcript the loop carried forward stayed bounded (a few rounds, not
    # all 8), proving compaction fed back into the loop.
    assert count_tokens(outcome.transcript, LEN, per_message_overhead=0) <= 60
