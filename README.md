# agentix

A generic, batteries-included **agent toolkit**. The agent loop, tool-calling,
guards, and observability are wiring you *configure* — not boilerplate you
rewrite for every project.

- **Async-first** core loop with a sync wrapper for scripts.
- **Provider-agnostic**, with a real **Anthropic** adapter included.
- **Define tools** with a `@tool` decorator (JSON schema from type hints).
- **Security as a first-class subsystem**: a trust boundary between user
  instructions and tool data, permission tiers, and PII/injection guards —
  all opt-in config, not baked into the loop.

> Status: **alpha**, under active development. APIs may change before `1.0`.

## Install

```bash
pip install agentix                 # core
pip install "agentix[anthropic]"    # + Anthropic adapter
```

## Quickstart (no provider needed)

```python
import asyncio
from agentix import Agent, LocalToolExecutor, MockModel, ModelResponse, ToolCall

# A scripted model: ask for a tool, then answer with the result.
model = MockModel([
    ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3})]),
    ModelResponse(text="The answer is 5."),
])

executor = LocalToolExecutor({"add": lambda a, b: a + b})

agent = Agent(model=model, system_prompt="You are helpful.", tool_executor=executor)
outcome = asyncio.run(agent.run("What is 2 + 3?"))
print(outcome.status, "->", outcome.answer)   # completed -> The answer is 5.
```

## Why

Everyone re-codes the same agentic loop, tool dispatch, and safety checks.
`agentix` keeps the loop thin and shared, and makes everything load-bearing —
the model, the tools, the guards — injectable and declarative.

## Roadmap

See [`PLAN.md`](./PLAN.md). Done: core types, packaging, and the async loop with
budget guards (P0–P1). Next: the `@tool` decorator and registry (P2), then the
guard subsystem (P3) and the Anthropic adapter (P4).

## License

MIT
