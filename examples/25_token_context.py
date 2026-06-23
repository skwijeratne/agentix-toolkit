"""25 — Token-accurate context trimming.

`TrimRounds` counts whole rounds and `TruncateToolOutputs` counts characters —
proxies for what a model's context window actually measures: **tokens**.
`FitContextWindow` trims the transcript to a real token budget, using a pluggable
token counter (a heuristic by default; swap in tiktoken or a provider tokenizer).

Run:
    python examples/25_token_context.py
"""

from __future__ import annotations

import asyncio

from agentix import (
    Agent,
    AgentEvents,
    AgentPolicy,
    FitContextWindow,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
    approx_token_counter,
    count_tokens,
    tool,
)


@tool
def lookup(topic: str) -> str:
    """Look up a topic (returns a chunky result, as real tools do)."""
    return f"Detailed notes about {topic}: " + ("lorem ipsum " * 40)


def model(messages):  # type: ignore[no-untyped-def]
    # Always calls the tool, so the transcript keeps growing — a stand-in for a
    # long agentic run that would otherwise overflow the context window.
    return ModelResponse(tool_calls=[ToolCall("lookup", {"topic": f"item-{len(messages)}"})])


async def main() -> None:
    # 1) Measure a transcript in tokens (default heuristic counter).
    sample = [
        Message(Role.SYSTEM, "You are helpful."),
        Message(Role.USER, "Summarize the last quarter."),
    ]
    print("sample transcript tokens:", count_tokens(sample))
    print("approx tokens for a 400-char string:", approx_token_counter("x" * 400))

    # 2) Cap the working context at a token budget. Older whole rounds are
    # dropped (never splitting a tool call from its result); reserve headroom for
    # the model's reply. The loop trims before every step, so context stays
    # bounded no matter how long the run goes.
    compactions = 0

    def on_compact(before: int, after: int) -> None:
        nonlocal compactions
        compactions += 1

    agent = Agent(
        model=MockModel(model),
        system_prompt="You are a research assistant.",
        tools=[lookup],
        policy=AgentPolicy(max_steps=8),
        context_strategy=FitContextWindow(max_tokens=400, reserve_tokens=64),
        events=AgentEvents(on_compact=on_compact),
    )
    outcome = await agent.run("Research everything.")

    print("steps run:", outcome.steps, "| compactions:", compactions)
    final = count_tokens(outcome.transcript)
    print(f"final transcript tokens: {final} (bounded near the budget, not growing")
    print("  with step count — an untrimmed 8-step run would be ~1000+)")

    # 3) For exact counts, pass any Callable[[str], int] as the counter:
    #
    #   import tiktoken
    #   enc = tiktoken.get_encoding("cl100k_base")
    #   FitContextWindow(max_tokens=180_000, counter=lambda s: len(enc.encode(s)))
    #
    # ...or a provider tokenizer. The default heuristic needs no dependencies.


if __name__ == "__main__":
    asyncio.run(main())
