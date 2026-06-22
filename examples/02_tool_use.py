"""02 — Using a tool.

The model asks to call a tool, the loop runs it, feeds the result back, and the
model produces a final answer using it.

Tools are just callables registered with a LocalToolExecutor by name. (The
ergonomic @tool decorator with auto-generated schemas is coming in P2; for now
we wire the executor directly.)

Run:
    PYTHONPATH=src python examples/02_tool_use.py
"""

from __future__ import annotations

from agentix import Agent, LocalToolExecutor, MockModel, ModelResponse, ToolCall


def add(a: int, b: int) -> int:
    """A plain Python function used as a tool."""
    return a + b


def main() -> None:
    # Turn 1: ask to call `add`. Turn 2: answer with the result.
    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="call-1")]),
            ModelResponse(text="2 + 3 = 5."),
        ]
    )

    executor = LocalToolExecutor({"add": add})

    agent = Agent(
        model=model,
        system_prompt="You can do arithmetic with the `add` tool.",
        tool_executor=executor,
    )

    outcome = agent.run_sync("What is 2 + 3?")

    print("answer:", outcome.answer)   # 2 + 3 = 5.
    print("steps: ", outcome.steps)    # 2 (one tool turn + one answer turn)


if __name__ == "__main__":
    main()
