# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/skwijeratne/agentix-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/skwijeratne/agentix-toolkit/releases/tag/v0.1.0
