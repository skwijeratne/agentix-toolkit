# agentix

**A friendly, batteries-included toolkit for building AI agents in Python.**

## What's an "agent," and what does this do?

An *agent* is an AI model that doesn't just answer once — it can **use tools** to
get its job done. You ask a question; the model decides it needs to look
something up; it calls a tool you gave it; it reads the result; and it keeps going
until it has a final answer.

Building that means writing the same loop every single time:

> ask the model → it asks for a tool → run the tool → feed the result back → repeat → final answer.

You also end up re-writing the same safety checks, the same retry logic, the same
cost tracking… for every project.

**agentix gives you that loop, already built.** You bring three things and plug
them in:

1. **a model** — which AI to use (Claude, GPT, Gemini, a local model, …),
2. **tools** — what the agent is allowed to do (look up weather, send an email, run code),
3. **guards** — optional safety checks (ask a human first, block sensitive data).

Everything else — the loop, retries, streaming, saving progress, tracking cost —
is handled for you and is easy to turn on.

```python
from agentix import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: 21°C, sunny"

agent = Agent(model=my_model, system_prompt="Help with the weather.", tools=[get_weather])
outcome = await agent.run("What's the weather in Lisbon?")
print(outcome.answer)
```

## Why you might like it

- **It works with any AI model.** Adapters for Anthropic, OpenAI, Gemini, AWS
  Bedrock, local models via Ollama, and 100+ more through LiteLLM. Switching is a
  one-line change.
- **Safety is built in, but optional.** Turn on protections against prompt
  injection and data leaks, require a human to approve risky actions, or run
  untrusted code in a locked-down sandbox — when you want them.
- **No surprises in production.** Track spending in real dollars, set budgets,
  stream answers as they're written, save a run and resume it later, and trim long
  conversations so they never overflow the model.
- **Small and honest.** The core has **no required dependencies**. You add only
  the pieces you use.

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started.md)**

    Install it and run your first agent in a few minutes — no API key needed.

-   :material-book-open-variant: **[Guides](guides/tools.md)**

    Short, practical walkthroughs of each feature, each with runnable code.

-   :material-shield-check: **[Security model](security.md)**

    How agentix keeps a tool's output from hijacking your agent — in plain terms.

-   :material-api: **[API reference](reference/agent.md)**

    Every class and function, generated from the code.

</div>

!!! note "Status"
    agentix is **alpha** and under active development. The ideas are stable, but
    some names may change before version 1.0.
