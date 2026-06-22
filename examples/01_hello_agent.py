"""01 — Hello, agent.

The smallest possible agentix program: a model that just answers, no tools.

We use MockModel (a scripted, dependency-free model) so this runs with zero
setup and no API key. In a real app you'd swap MockModel for a provider adapter.

Run:
    PYTHONPATH=src python examples/01_hello_agent.py
"""

from __future__ import annotations

from agentix import Agent, MockModel, ModelResponse


def main() -> None:
    # A model that returns one final answer (no tool calls -> the loop ends).
    model = MockModel([ModelResponse(text="Hello! I'm a tiny agent.")])

    agent = Agent(model=model, system_prompt="You are a friendly assistant.")

    # run_sync() is the blocking convenience wrapper around the async run().
    outcome = agent.run_sync("Say hi.")

    print("status:", outcome.status)   # completed
    print("answer:", outcome.answer)   # Hello! I'm a tiny agent.
    print("steps: ", outcome.steps)    # 1


if __name__ == "__main__":
    main()
