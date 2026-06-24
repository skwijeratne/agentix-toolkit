"""29 — Record/replay cassettes.

Testing against a live model is slow, costly, and flaky. `CassetteModel` wraps
any model: the first run records each response to a JSON file; later runs replay
from it with no network — deterministic tests of your real prompts/flows.

`mode="auto"` (default) records when the file is missing and replays when it
exists. This demo uses MockModel as the "real" model so it runs without a key.

Run:
    python examples/29_cassettes.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agentix import Agent, CassetteModel, MockModel, ModelResponse, ToolCall, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def real_model() -> MockModel:
    # Stand-in for a live provider (AnthropicModel(), OpenAIModel(), ...).
    return MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
            ModelResponse(text="The answer is 5."),
        ]
    )


async def run_once(model: CassetteModel) -> str:
    agent = Agent(model=model, system_prompt="Do the math.", tools=[add])
    return (await agent.run("What is 2 + 3?")).answer or ""


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "math.json")

        # First run: file missing -> record (calls the "real" model), then save.
        rec = CassetteModel(path, model=real_model(), mode="auto")
        print("mode:", rec.mode)
        print("recorded run ->", await run_once(rec))
        rec.save()
        print("cassette written:", Path(path).exists())

        # Second run: file exists -> replay, no model needed (pass none at all).
        rep = CassetteModel(path, mode="auto")
        print("mode:", rep.mode)
        print("replayed run ->", await run_once(rep))


if __name__ == "__main__":
    asyncio.run(main())
