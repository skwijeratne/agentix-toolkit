"""09 — Streaming.

`Agent.stream()` yields events as the run unfolds:
  * `AnswerDelta`  — incremental answer text (print it live)
  * `ToolStarted` / `ToolFinished` — around each tool call
  * `Done`         — the terminal event, carrying the full outcome

Any model with a `stream()` method drives real streaming; models without one
fall back transparently. (MockModel streams here, so no API key is needed.)

Note: `on_answer` guards like PiiRedactionGuard can't un-send already-streamed
deltas — `Done.outcome.answer` is redacted, but the live deltas are raw. Use
`run()` if you need the user-facing text itself redacted.

Run:
    PYTHONPATH=src python examples/09_streaming.py
"""

from __future__ import annotations

import asyncio

from agentix import (
    Agent,
    AnswerDelta,
    Done,
    MockModel,
    ModelResponse,
    ToolCall,
    ToolFinished,
    ToolStarted,
    tool,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: 18C, light rain"


async def main() -> None:
    model = MockModel(
        [
            # Turn 1: call a tool.
            ModelResponse(tool_calls=[ToolCall("get_weather", {"city": "London"})]),
            # Turn 2: stream the final answer.
            ModelResponse(text="It is 18C and lightly raining in London today."),
        ]
    )
    agent = Agent(model=model, system_prompt="You are a weather assistant.", tools=[get_weather])

    print("streaming:\n")
    async for event in agent.stream("What's the weather in London?"):
        if isinstance(event, AnswerDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolStarted):
            print(f"[calling {event.call.name}({event.call.args})]", end="", flush=True)
        elif isinstance(event, ToolFinished):
            print(f"[-> {event.result.content}]\n", end="", flush=True)
        elif isinstance(event, Done):
            print(f"\n\n[done: {event.outcome.status}, {event.outcome.steps} steps]")


if __name__ == "__main__":
    asyncio.run(main())
