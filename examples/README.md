# agentix examples

Examples 01–04 are dependency-free (they use `MockModel`, so no API key needed).
From the repo root, after `uv sync`:

```bash
uv run python examples/01_hello_agent.py
```

(Without uv: `pip install -e .`, then `python examples/01_hello_agent.py`.)

| File | Shows | Needs |
|------|-------|-------|
| `01_hello_agent.py` | The minimal loop — a model that just answers. | — |
| `02_tool_use.py` | Registering a tool and a tool→answer round trip. | — |
| `03_async_dynamic_loop.py` | An `async` tool + a model that reacts to the conversation. | — |
| `04_policy_and_trust.py` | `AgentPolicy` step budget and the trusted/untrusted boundary. | — |
| `05_anthropic_model.py` | A live Claude-backed agent with a `@tool`. | `agentix[anthropic]` + `ANTHROPIC_API_KEY` |
| `06_tool_decorator.py` | The `@tool` decorator: schema from type hints + docstring, optionals, `Literal`/enum, lists. | — |
| `07_guards.py` | The guard subsystem: tiers, confirmation, PII/injection defense, untrusted-data wrap, audit events. | — |
| `08_persistence.py` | Checkpoint a run to a `FileStore` and `resume` it after an interruption. | — |
| `09_streaming.py` | `Agent.stream()`: live answer deltas + tool events + terminal `Done`. | — |
| `10_concurrency.py` | Running many agents safely: `bounded_gather` + a shared `Limiter`. | — |
| `11_mcp.py` | Connect to an MCP server and use its tools in an agent. | `agentix[mcp,anthropic]` + a server + `ANTHROPIC_API_KEY` |
| `12_context.py` | Bound the transcript with `TrimRounds` / `TruncateToolOutputs`. | — |
| `13_subagents.py` | Delegate a subtask to a child agent (`subagent_tool`); the child's cost/tokens roll up into the parent. | — |
| `14_cost_and_interrupt.py` | USD cost tracking, `max_budget_usd`, and `Interrupt`. | — |
| `15_permissions.py` | Dynamic permissions: `CallbackGuard` (can_use_tool) + `ToolAllowlistGuard`. | — |
| `16_reliability.py` | Output validation + retry; resilient models (`RetryModel`/`FallbackModel`). | — |
| `17_eval.py` | Eval harness: score an agent over golden cases, gate CI on pass rate. | — |
| `18_verification.py` | Self-consistency (`SelfConsistencyModel`) + LLM-as-judge (`JudgeGuard`). | — |
| `19_tracing.py` | OpenTelemetry tracing: model/tool/run spans. | `agentix[otel]` + `opentelemetry-sdk` |
| `20_prompts.py` | Prompt registry/versioning + typed Anthropic reasoning knobs. | — |
| `21_providers.py` | Provider gallery: OpenAI / Gemini / Bedrock / Ollama / LiteLLM, one-line swap. | — (per-provider extra to run live) |
| `22_multimodal.py` | Multimodal input: text + image / PDF / audio parts via `TextPart`/`ImagePart`/`DocumentPart`/`AudioPart`. | — |
| `23_sandbox.py` | `SubprocessExecutor`: run untrusted/model-generated code in an isolated subprocess (no network, rlimits, timeout). | — (POSIX) |
| `24_suspend_resume.py` | Durable human-in-the-loop: `suspend_on_confirm` pauses for approval, persists, and `resume(decisions=…)` continues (even in a new process). | — |
| `25_token_context.py` | Token-accurate context: `count_tokens` + `FitContextWindow` trims the transcript to a real token budget (pluggable counter). | — |
| `26_memory.py` | Cross-session memory: a `Memory` (`InMemoryMemory`) recalled into context across two sessions, persisted via a `FileStore`. | — |
| `27_structured_output.py` | `Agent(response_model=…)`: validated `outcome.parsed`, schema-prompt + native provider enforcement, retry on failure. | — |
| `28_rate_limit.py` | `RetryModel` honoring `Retry-After` (rate-limit-aware) instead of blind backoff, with an `on_retry` hook. | — |
| `29_cassettes.py` | `CassetteModel`: record model responses to a file, then replay them deterministically (no network). | — |
| `30_serving_fastapi.py` | Serve an agent over HTTP: streaming `sse_response` + a suspend/resume approval flow, with a tiny browser client. | `agentix[serving]` + `fastapi` + `uvicorn` |

To run the Anthropic example:

```bash
pip install "agentix[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
python examples/05_anthropic_model.py
```

> These cover P0–P5 (streaming + persistence included). Remaining before a
> `0.1.0` release: real CI, docs site, and publishing to PyPI.
