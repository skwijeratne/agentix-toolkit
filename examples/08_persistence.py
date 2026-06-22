"""08 — Persistence & resume.

agentix doesn't assume a database: persistence is a pluggable `Store`. Core
ships `MemoryStore` (a dict) and `FileStore` (one JSON file per run). Pass a
`store` and a `run_id` and the loop checkpoints after every step — so a run that
is interrupted (crash, timeout, process restart) can be resumed.

This demo forces an "interruption" with a 1-step budget, shows the checkpoint
file on disk, then resumes with a fresh agent to finish the job.

Run:
    PYTHONPATH=src python examples/08_persistence.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentix import Agent, AgentPolicy, FileStore, MockModel, ModelResponse, ToolCall, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = FileStore(d)

        # --- first agent: only allowed one step, so it stops after the tool call
        interrupted = Agent(
            model=MockModel([ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")])]),
            system_prompt="You do math with the add tool.",
            tools=[add],
            store=store,
            policy=AgentPolicy(max_steps=1),
        )
        outcome1 = interrupted.run_sync("What is 2 + 3?", run_id="job-42")
        print("first run status:", outcome1.status, f"({outcome1.reason})")

        checkpoint = Path(d) / "job-42.json"
        print("checkpoint written:", checkpoint.name)
        print("checkpoint steps:", __import__("json").loads(checkpoint.read_text())["steps"])

        # --- later: a fresh agent resumes the same run_id and finishes
        resumed = Agent(
            model=MockModel([ModelResponse(text="2 + 3 = 5.")]),
            system_prompt="You do math with the add tool.",
            tools=[add],
            store=store,
        )
        outcome2 = resumed.resume_sync("job-42")
        print("resumed status:", outcome2.status)
        print("final answer:", outcome2.answer)
        print("total steps:", outcome2.steps)


if __name__ == "__main__":
    main()
