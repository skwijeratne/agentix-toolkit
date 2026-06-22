"""12 — Context management (bounding the transcript).

A long agentic run accumulates a turn per step; without bounds, memory grows and
the provider's context window eventually overflows. A `ContextStrategy` is
applied before each model call to keep the working transcript small. Two
pairing-safe strategies ship:

  * `TrimRounds(n)`          — keep system + task + the most recent n tool rounds
  * `TruncateToolOutputs(k)` — clip any tool output longer than k chars

Strategies never split an assistant tool-call from its result (which would break
providers like Anthropic). All dependency-free (MockModel).

Run:
    PYTHONPATH=src python examples/12_context.py
"""

from __future__ import annotations

from agentix import (
    Agent,
    AgentEvents,
    AgentPolicy,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    TrimRounds,
    TruncateToolOutputs,
    tool,
)


@tool
def step() -> str:
    """Do one unit of work."""
    return "step result"


@tool
def fetch() -> str:
    """Fetch a large document."""
    return "DATA " * 1000  # ~5000 chars


def demo_trim_rounds() -> None:
    print("== TrimRounds(2): keep only the last 2 tool rounds ==")
    # 5 tool rounds, then an answer.
    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("step", {}, id=f"c{i}")]) for i in range(5)]
        + [ModelResponse(text="finished")]
    )
    events = AgentEvents(
        on_compact=lambda before, after: print(f"  compacted: {before} -> {after} messages")
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[step],
        policy=AgentPolicy(max_steps=10),
        context_strategy=TrimRounds(2),
        events=events,
    )
    outcome = agent.run_sync("do 5 steps")
    print(f"  final transcript length: {len(outcome.transcript)} (would be 12 without trimming)\n")


def demo_truncate() -> None:
    print("== TruncateToolOutputs(80): clip big tool outputs ==")
    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("fetch", {}, id="c1")]), ModelResponse(text="ok")]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[fetch],
        context_strategy=TruncateToolOutputs(80),
    )
    outcome = agent.run_sync("fetch the doc")
    tool_msg = next(m for m in outcome.transcript if m.role is Role.TOOL)
    print(f"  tool output stored as: {tool_msg.content!r}")
    print(f"  ({len(tool_msg.content)} chars instead of ~5000)")


if __name__ == "__main__":
    demo_trim_rounds()
    demo_truncate()
