# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `AnthropicModel` typed reasoning/cost knobs: `thinking` (`True`/`"adaptive"`/
  `"summarized"`/`"disabled"`/dict), `effort` (`low`…`max`), and `task_budget`
  (int; adds the required beta header) — previously only via opaque `extra`.
  Docstring documents refusal-fallback behavior.
- `PromptRegistry`: lightweight in-process prompt versioning with `register` /
  `get` / `rollback` / `render` and `to_dict`/`from_dict` persistence.

## [0.2.1] - 2026-06-23

### Fixed
- `agentix.__version__` now reflects the installed distribution version (derived
  from package metadata) instead of a hardcoded string that could drift. (0.2.0
  shipped reporting `0.1.0`.)

## [0.2.0] - 2026-06-23

### Added
- Subagents: `subagent_tool(agent, ...)` exposes a child agent as a delegable
  tool (its own model/system prompt/tools/guards); composes with the loop and
  `bounded_gather`.
- Cost & control: USD cost tracking (`pricing` module, `cost_usd`, and
  `cost_usd` on `ModelResponse`/`AgentOutcome`; the Anthropic adapter fills
  `input_tokens`/`output_tokens`/`cost_usd`); `AgentPolicy.max_budget_usd`; and
  `Interrupt` to stop a run/stream at a safe boundary.
- Dynamic permissions: `CallbackGuard` (a `can_use_tool`-style per-call callback
  returning allow/deny/confirm) and `ToolAllowlistGuard` (scope a run to a
  subset of tools).
- Output validation + retry: `Agent(output_validator=, max_output_retries=)`
  re-prompts on a failed validation and exposes `AgentOutcome.parsed`. Ships
  `json_output`, `pydantic_output`, `regex_output`.
- Resilient model wrappers: `RetryModel` (backoff) and `FallbackModel`
  (try-next-on-error), composable and drop-in.
- Eval harness (`agentix.evals`): `evaluate(...)` runs an agent over `Case`s and
  returns an `EvalReport` with `pass_rate` / `format_success_rate` /
  `assert_pass_rate()` (gate CI on regressions). Scorers: `exact_match`,
  `contains`, `regex_match`, `predicate`, `llm_judge`.
- `SelfConsistencyModel`: sample a model N times per turn and return the majority
  vote (drop-in `ModelFn`).
- `JudgeGuard`: an LLM reviews the final answer against a rubric and replaces it
  on failure (an `on_answer` safety/on-brand/format gate).
- Anthropic adapter: structured-output passthrough documented
  (`output_config={"format": ...}`) and `strict` tool schemas forwarded.
- OpenTelemetry tracing (`agentix[otel]`): `TracingModel`, `tracing_events`, and
  `trace_run` produce a span tree (run → model/tool spans) for your observability
  stack.

## [0.1.0] - 2026-06-22

Initial release.

### Core
- Async agent loop: `Agent.run` / `run_sync` / `stream` / `resume`, with step and
  token budgets.
- Provider-agnostic `ModelFn`; tool schemas flow to the model.
- `@tool` decorator generating JSON Schema from type hints + docstrings;
  `Tool` / `ToolRegistry`.
- `LocalToolExecutor` — sync tools run off the event loop; real per-call timeouts.

### Security (opt-in guard pipeline)
- Trust boundary between user instructions and tool data.
- Guards: `TierGuard`, `PiiUrlGuard`, `InjectionGuard`, `UntrustedDataGuard`,
  fail-closed `RecipientTrustGuard`, and `PiiRedactionGuard` (answer egress).
- Async-or-sync confirmation; `AgentEvents` audit hooks; `secure_defaults()`.

### Providers & streaming
- Anthropic adapter (`claude-opus-4-8`) with tool use and streaming.
- Streaming events: `AnswerDelta` / `ToolStarted` / `ToolFinished` / `Done`.

### Persistence & scale
- Pluggable `Store` (`MemoryStore`, atomic non-blocking `FileStore`) + JSON codec.
- `Limiter` and `bounded_gather` for fleet backpressure.

### Integrations & context
- MCP client support (`MCPServer`, `agentix[mcp]`): discover an MCP server's tools
  and use them in an agent.
- Context management: `ContextStrategy`, `TrimRounds`, `TruncateToolOutputs`.

### Delegation, cost & control
- Subagents: `subagent_tool` exposes a child agent as a delegable tool.
- Cost: `pricing` module + `cost_usd`; `ModelResponse`/`AgentOutcome` carry
  `cost_usd`; `AgentPolicy.max_budget_usd` aborts a run over budget.
- `Interrupt` stops a run or stream at the next safe boundary.

[Unreleased]: https://github.com/skwijeratne/agentix-toolkit/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/skwijeratne/agentix-toolkit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/skwijeratne/agentix-toolkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/skwijeratne/agentix-toolkit/releases/tag/v0.1.0
