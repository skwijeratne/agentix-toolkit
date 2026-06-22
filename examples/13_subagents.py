"""13 — Subagents (delegation).

A subagent is a child agent exposed to a parent as a tool. The parent delegates
a subtask; the child runs its own loop — own model, system prompt, tools — and
returns its answer. Because it's just a tool, it composes with everything else
(guards, the executor, `bounded_gather` for fanout).

All dependency-free (MockModel), so no API key needed.

Run:
    PYTHONPATH=src python examples/13_subagents.py
"""

from __future__ import annotations

from agentix import Agent, MockModel, ModelResponse, Role, ToolCall, subagent_tool, tool


# --- a specialist subagent: a math expert with its own tool ---------------
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def build_math_agent() -> Agent:
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 20, "b": 22}, id="m1")]),
            ModelResponse(text="20 + 22 = 42."),
        ]
    )
    return Agent(model=model, system_prompt="You are a meticulous math expert.", tools=[add])


def main() -> None:
    # Wrap the subagent as a tool the lead agent can call.
    math = subagent_tool(
        build_math_agent(),
        name="math_expert",
        description="Delegate any arithmetic question to a math specialist.",
    )

    # The lead agent delegates, then summarizes.
    lead_model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("math_expert", {"task": "what is 20 + 22?"})]),
            ModelResponse(text="I checked with my math expert: the answer is 42."),
        ]
    )
    lead = Agent(
        model=lead_model,
        system_prompt="You coordinate work and delegate to specialists.",
        tools=[math],
    )

    outcome = lead.run_sync("Please compute 20 + 22.")
    print("lead answer:", outcome.answer)

    delegated = next(m for m in outcome.transcript if m.role == Role.TOOL)
    print("subagent returned:", delegated.content)


if __name__ == "__main__":
    main()
