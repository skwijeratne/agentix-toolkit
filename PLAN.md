# agentix — Plan

A generic, batteries-included **agent toolkit**: the agent loop, tool-calling,
guards, memory, and observability are wiring you *configure*, not boilerplate you
rewrite for every project. Provider-agnostic core with adapters; thin and
composable, not a kitchen-sink framework.

- **Distribution / import name:** `agentix`
- **Reference implementation:** `secure_agent.py` (the security subsystem started here)

## Positioning

A clean, small-core alternative in the space of pydantic-ai / smolagents:
the loop is thin and shared, everything load-bearing is injected. **Security is
one first-class subsystem** (trust boundary, permission tiers, PII/injection
guards) — a real strength, but it sits alongside tools, memory, and
observability rather than being the whole story.

## Decisions locked

- **Async-first.** Core loop is `async`; thin `run_sync()` wrapper for scripts/CLIs.
- **Batteries:** provider-agnostic core + one real adapter (**Anthropic**) + a
  `@tool` decorator that derives JSON schema from type hints & docstrings + a
  `MockModel` for tests.
- **Name:** `agentix`.
- **License:** MIT (scaffolding choice; switch to Apache-2.0 for a patent grant).
- **Two reference fixes carried into the port:** (1) restrictive
  `recipient_is_trusted` default (lands with guards in P3); (2) tool schemas
  flow to the model (done — `ModelFn` takes `tools=`).

## Core design (carried from the reference, generalized)

Kept: thin loop, `AgentPolicy` as data, guards as explicit ordered checkpoints,
the `trusted`/untrusted-data boundary. Changes:

1. **Tools flow to the model.** `ModelFn(messages, *, tools=[...schemas...])`.
2. **A way to *define* tools.** `@tool` → `Tool` (name, description, JSON schema,
   async `run`); a `ToolRegistry` feeds schemas to the model. *(P2)*
3. **Execution is a separate, pluggable boundary.** Registry says *what* exists;
   a `ToolExecutor` *runs* it under policy limits the model can't influence.
   `LocalToolExecutor` ships now; a `SubprocessExecutor` can isolate later.
4. **Guards are a pipeline of uniform objects.** `before_call -> Decision`,
   `after_output -> text`. Security guards are opt-in config. *(P3)*
5. **Async, non-blocking human-in-the-loop.** `async confirm(request) -> bool`. *(P3)*
6. **Observability first-class.** `AgentEvents` callbacks for tracing/audit. *(P3)*

## Package layout (src layout, typed, `py.typed`)

```
agentix/
  pyproject.toml  README.md  LICENSE  CHANGELOG.md  PLAN.md
  src/agentix/
    __init__.py types.py policy.py errors.py
    model.py executors.py agent.py
    tools.py            # P2
    confirm.py events.py guards/   # P3
    providers/ anthropic.py mock.py
    py.typed
  tests/  examples/
```

## Phased build

- **P0 — Scaffold.** ✅ pyproject, src layout, tooling config, `py.typed`,
  core `types`/`errors`, installable package.
- **P1 — Core loop.** ✅ `policy`, `model` protocol (with `tools=`),
  `executors` (`ToolExecutor` + `LocalToolExecutor` w/ timeout), `MockModel`,
  async `Agent` loop + `run_sync`, budget/step guards. Integration tests.
- **P2 — Tools.** ✅ `@tool` decorator, `Tool`, `ToolRegistry` (doubles as the
  executor), JSON-schema generation from type hints + docstrings (primitives,
  `Optional`, `Literal`/enum, `list`/`dict`); `Agent(tools=[...])` derives the
  executor and schemas. Tests + examples 05 (real) & 06 (decorator showcase).
- **P3 — Guards.** ✅ `GuardPipeline` of uniform `Guard` objects with three
  checkpoints: `before_call -> Decision` (tool ingress), `after_output -> text`
  (tool egress), `on_answer -> text` (answer egress to the user). Ships
  `TierGuard`, `PiiUrlGuard`, `InjectionGuard`, `UntrustedDataGuard`, opt-in
  fail-closed `RecipientTrustGuard`, and opt-in `PiiRedactionGuard` (DLP on the
  final answer, with its own tighter patterns). `secure_defaults()` factory,
  async-or-sync `confirm_fn`, `AgentEvents` audit hooks. Guards are opt-in — no
  guards means a clean loop. Tests + example 07.
- **P4 — Anthropic adapter.** ✅ Real tool-use translation behind
  `pip install agentix[anthropic]`; example 05 + fake-client tests.
- **P5 — Polish & ship.**
  - ✅ **Streaming** — `Agent.stream()` yields `AnswerDelta` / `ToolStarted` /
    `ToolFinished` / `Done`; `StreamingModelFn` protocol; streaming in MockModel
    and the Anthropic adapter; transparent fallback for non-streaming models.
  - ✅ **Persistence/resume** — pluggable `Store` (`MemoryStore`, `FileStore`),
    a JSON codec for the core types (`serde`), per-step checkpointing via
    `run(..., run_id=)`, and `resume()` / `resume_sync()`.
  - ✅ Tooling (uv): `uv sync` / `uv run`; CI (`.github/workflows/ci.yml`,
    pytest matrix 3.10–3.13 + ruff + mypy --strict, all blocking) and release
    (`release.yml`, `uv build` + `twine check` + PyPI Trusted Publishing on `v*`
    tags). `LICENSE`, `CHANGELOG.md`, `RELEASING.md`. **Verified green locally:**
    101 pytest, ruff clean, mypy --strict clean (25 files). `uv.lock` committed.
  - ☐ Watch CI go green on push, then tag `v0.1.0` to publish. Optional docs site.
- **P6 — MCP client support.** ✅ `MCPServer` connects to an MCP server
  (stdio / HTTP / SSE, lazy `mcp` import behind `agentix[mcp]`), discovers its
  tools as agentix `Tool`s (`inputSchema` → `parameters`), and routes calls over
  the live session — plugs into `Agent(tools=...)`. Tests + example 11.
  Roadmap for further gaps vs. the Anthropic Agent SDK: see `PLAN.gaps.md`.
- **P7 — Context management.** ✅ Pluggable `ContextStrategy` applied before each
  model call (opt-in). `TrimRounds(n)` (keep system + task + last n tool rounds)
  and `TruncateToolOutputs(k)` (clip long tool outputs), both pairing-safe so
  they never break provider tool_use/tool_result pairing. `on_compact` event.
  Closes the unbounded-transcript memory risk. Tests + example 12.
- **P9 — Subagents.** ✅ `subagent_tool(agent, ...)` exposes a child `Agent` as a
  delegable `Tool`; composes with the loop, guards, and `bounded_gather`. Tests
  + example 13.
- **P10 — Cost + interrupt.** ✅ `pricing` (per-model table + `cost_usd`);
  `ModelResponse`/`AgentOutcome` carry `cost_usd` (Anthropic adapter fills it);
  `AgentPolicy.max_budget_usd` aborts; `Interrupt` stops a run/stream at a safe
  boundary. Tests + example 14. (P8 — permission callbacks — still open; see
  `PLAN.gaps.md`.)

> ⚠️ Streaming caveat: `on_answer` egress guards (PII redaction) can't un-send
> already-streamed deltas — deltas are raw; `Done.outcome.answer` is redacted.
> Use `run()` when the user-facing text itself must be redacted before emission.

## Concurrency hardening (for fleets of agents)

The loop is already per-run isolated (no shared mutable `Agent` state; asyncio's
cooperative scheduling makes sync regions atomic). Three production fixes landed:

- **Sync tools never block the loop.** `LocalToolExecutor` runs synchronous tool
  functions in a worker thread (`asyncio.to_thread`); a blocking tool can't stall
  the event loop and starve other agents. The timeout now actually returns control
  (a timed-out sync tool's thread is orphaned — Python can't kill threads — and
  draws from the default thread pool; size it for your concurrency).
- **`FileStore` is non-blocking + atomic.** I/O is thread-offloaded; writes use
  temp-file + fsync + `os.replace`, so a crash mid-write can't corrupt a
  checkpoint. Cross-process: last-writer-wins per `run_id`, no lock (keep one
  writer per run).
- **Backpressure primitives.** `Limiter(N)` (shared semaphore, inject via
  `model_limiter=` to cap concurrent model calls fleet-wide) and
  `bounded_gather(aws, limit=N)` (cap concurrent runs in batch jobs). Example 10.

> Still on the caller: a real shared store (Redis/DB) for multi-process fleets,
> and context trimming for very long transcripts (unbounded memory per run).

## Open questions

- ~~Tool **schema source**~~ — resolved: **type hints + docstring only** (zero
  deps). Pydantic-model args can be added later as an opt-in if needed.
- **Multiple tool calls per turn:** keep sequential (current) or run
  independent calls concurrently? (Concurrency complicates confirm/guard order.)
  Defer the decision to P3, where the guard ordering constraints become concrete.
