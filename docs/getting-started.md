# Getting started

This page gets you from nothing to a working agent in a few minutes. You don't
need an API key for the first example.

## 1. Install

agentix needs **Python 3.10 or newer**. The package is called
`agentix-toolkit` on PyPI, but you `import agentix` in code.

The core has **no required dependencies**. You add *extras* only for the pieces
you actually use (an extra is an optional add-on, written in square brackets).

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add agentix-toolkit                      # just the core
uv add "agentix-toolkit[anthropic]"         # + the Claude adapter
uv add "agentix-toolkit[openai]"            # + the OpenAI adapter
```

Or with pip:

```bash
pip install "agentix-toolkit[anthropic]"
```

Other extras: `gemini`, `bedrock`, `ollama`, `litellm` (model providers), plus
`mcp` (connect to tool servers) and `otel` (tracing). You can combine them:
`agentix-toolkit[anthropic,mcp]`.

## 2. Run an agent with no API key

`MockModel` is a pretend model that returns answers you script in advance. It's
perfect for learning the shape of things and for writing tests — no network, no
key, no cost.

Here the "model" first asks to use a tool, then gives a final answer:

```python
import asyncio
from agentix import Agent, MockModel, ModelResponse, ToolCall, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# A scripted model: first it asks to call `add`, then it answers.
model = MockModel([
    ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3})]),
    ModelResponse(text="The answer is 5."),
])

agent = Agent(model=model, system_prompt="You are helpful.", tools=[add])
outcome = asyncio.run(agent.run("What is 2 + 3?"))
print(outcome.status, "→", outcome.answer)   # completed → The answer is 5.
```

A few things to notice:

- **`@tool`** turns a normal Python function into something the agent can call.
  Its name, the arguments, and the docstring are read automatically so the model
  knows when and how to use it.
- **`agent.run(...)`** runs the whole loop and gives you back an *outcome* — the
  final answer plus useful details (status, steps taken, tokens, cost, and the
  full transcript).
- It's **async** (note the `await` / `asyncio.run`). If you'd rather not deal with
  async, use `agent.run_sync("...")` instead.

## 3. Use a real model

Swap `MockModel` for a real provider's adapter. Nothing else changes — the tools,
the loop, everything stays the same.

```python
import asyncio
from agentix import Agent, tool
from agentix.providers.anthropic import AnthropicModel

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 21°C, partly cloudy"

agent = Agent(
    model=AnthropicModel(),                       # reads ANTHROPIC_API_KEY from the environment
    system_prompt="You are a concise weather assistant.",
    tools=[get_weather],
)
outcome = asyncio.run(agent.run("What's the weather in Paris?"))
print(outcome.answer)
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Want a different provider? See **[Models & providers](guides/providers.md)** — the
swap is one line.

## 4. Turn on safety checks

Tools can do real things (send email, spend money), and tool *results* can contain
sneaky instructions. **Guards** are optional safety checks. `secure_defaults()`
switches on a sensible set in one line, and a *policy* lets you say "always ask me
before sending email":

```python
from agentix import Agent, AgentPolicy, secure_defaults, always_approve

agent = Agent(
    model=my_model,
    system_prompt="...",
    tools=[send_email, read_ticket],
    policy=AgentPolicy(confirm_first={"send_email"}),  # ask a human before sending
    guards=secure_defaults(),
    confirm_fn=always_approve,                          # plug in your real "ask the user" here
)
```

To understand *why* these matter (and what they protect against), read the
**[Security model](security.md)** — it's written in plain language.

## Where to go next

- **[Guides](guides/tools.md)** — one short page per feature, each with runnable code.
- Every example on this site has a matching file in the project's
  [`examples/`](https://github.com/skwijeratne/agentix-toolkit/tree/main/examples)
  folder you can run directly.
