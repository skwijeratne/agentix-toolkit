"""The agent loop.

Small and shared: call the model, and if it asked for tools, run them and feed
the results back, until the model produces a final answer or a budget is hit.
Everything load-bearing — the model, the tools, the policy — is injected. The
loop only enforces the resource budgets here; the security guards plug in
later (P3) without changing this control flow.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any

from .concurrency import Limiter
from .confirm import ConfirmFn
from .content import ContentPart
from .context import ContextStrategy
from .control import Interrupt
from .errors import AgentError
from .events import AgentEvents
from .executors import ToolExecutor
from .guards import Guard, GuardContext, GuardPipeline
from .model import ModelFn, ToolSchema
from .policy import AgentPolicy
from .serde import SCHEMA_VERSION, transcript_from_dicts, transcript_to_dicts
from .store import Store
from .streaming import (
    AgentStreamEvent,
    AnswerDelta,
    Done,
    ModelStreamEvent,
    ResponseComplete,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from .tools import Tool, ToolRegistry
from .types import AgentOutcome, Message, ModelResponse, Role, ToolCall
from .validation import OutputValidator


class Agent:
    """Drives the async agent loop around an injected model and tool executor.

    Minimal usage::

        agent = Agent(model=my_model, system_prompt="...")
        outcome = await agent.run("Summarize today's tickets.")

    The easiest way to add tools is the ``tools=`` argument — pass ``@tool``
    functions (or a :class:`~agentix.tools.ToolRegistry`) and the agent derives
    both the executor and the schemas the model sees::

        agent = Agent(model=m, system_prompt="...", tools=[get_weather, add])

    For full control (e.g. a sandboxed executor) supply ``tool_executor`` and
    ``tool_schemas`` directly instead.
    """

    def __init__(
        self,
        *,
        model: ModelFn,
        system_prompt: str,
        policy: AgentPolicy | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        guards: Iterable[Guard] | None = None,
        confirm_fn: ConfirmFn | None = None,
        events: AgentEvents | None = None,
        store: Store | None = None,
        model_limiter: Limiter | None = None,
        context_strategy: ContextStrategy | None = None,
        output_validator: OutputValidator | None = None,
        max_output_retries: int = 1,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.policy = policy or AgentPolicy()
        self.store = store
        # Optional shared limiter to bound concurrent model calls across a fleet.
        self.model_limiter = model_limiter
        # Optional compaction applied before each model call (opt-in).
        self.context_strategy = context_strategy
        # Optional final-answer validation; on failure, re-prompt up to N times.
        self.output_validator = output_validator
        self.max_output_retries = max_output_retries

        # Guards are opt-in: no guards -> a clean loop. Pass
        # `guards=secure_defaults()` (or your own list) to turn on protections.
        self.guards = GuardPipeline(list(guards)) if guards is not None else GuardPipeline()
        self.confirm_fn = confirm_fn
        self.events = events or AgentEvents()

        # `tools=` is the high-level path: build a registry that serves as both
        # the executor and the schema source. Explicit tool_executor /
        # tool_schemas still win if also provided.
        if tools is not None:
            registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
            if tool_executor is None:
                tool_executor = registry
            if tool_schemas is None:
                tool_schemas = registry.schemas

        self.tool_executor = tool_executor
        self.tool_schemas: list[ToolSchema] = list(tool_schemas or [])

    # ── public entry points ───────────────────────────────────────────────

    async def run(
        self,
        user_request: str | list[ContentPart],
        *,
        run_id: str | None = None,
        interrupt: Interrupt | None = None,
    ) -> AgentOutcome:
        """Run the loop to completion. If ``run_id`` is given and a ``store`` is
        configured, the run is checkpointed after every step (resumable). Pass an
        ``Interrupt`` to stop the run at its next safe boundary."""
        messages = self._seed_messages(user_request)
        return await self._loop(messages, 0, 0, 0.0, run_id, self.store, interrupt)

    async def resume(
        self,
        run_id: str,
        *,
        store: Store | None = None,
        interrupt: Interrupt | None = None,
    ) -> AgentOutcome:
        """Reload a checkpointed run and continue the loop from where it stopped."""
        effective = store or self.store
        if effective is None:
            raise AgentError("resume() requires a store (on the Agent or as an argument)")
        state = await effective.load(run_id)
        if state is None:
            raise AgentError(f"no saved run found for run_id {run_id!r}")
        messages = transcript_from_dicts(state["messages"])
        return await self._loop(
            messages,
            int(state["steps"]),
            int(state["tokens_used"]),
            float(state.get("cost_usd", 0.0)),
            run_id,
            effective,
            interrupt,
        )

    def run_sync(
        self, user_request: str | list[ContentPart], *, run_id: str | None = None
    ) -> AgentOutcome:
        """Blocking convenience wrapper for scripts/CLIs. Do not call from
        inside a running event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(user_request, run_id=run_id))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead."
        )

    def resume_sync(self, run_id: str, *, store: Store | None = None) -> AgentOutcome:
        """Blocking wrapper around :meth:`resume`. Do not call from inside a
        running event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resume(run_id, store=store))
        raise RuntimeError(
            "resume_sync() cannot be called from a running event loop; await resume() instead."
        )

    async def stream(
        self,
        user_request: str | list[ContentPart],
        *,
        run_id: str | None = None,
        interrupt: Interrupt | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Run the loop, yielding events as they happen: ``AnswerDelta`` text
        chunks, ``ToolStarted``/``ToolFinished`` around tool calls, and a final
        ``Done`` carrying the outcome.

        Note: ``on_answer`` egress guards (e.g. PII redaction) cannot un-send
        already-streamed deltas — the deltas are raw, but ``Done.outcome.answer``
        is passed through the guards. Use :meth:`run` if you need the user-facing
        text itself redacted before it is emitted.
        """
        messages = self._seed_messages(user_request)
        steps = 0
        tokens_used = 0
        cost_usd = 0.0

        while steps < self.policy.max_steps:
            if interrupt is not None and interrupt.triggered:
                yield Done(await self._abort("interrupted", steps, tokens_used, cost_usd, messages))
                return
            steps += 1

            messages = await self._compact(messages)
            response: ModelResponse | None = None
            async for event in self._model_stream(messages):
                if isinstance(event, TextDelta):
                    yield AnswerDelta(event.text)
                elif isinstance(event, ResponseComplete):
                    response = event.response
            assert response is not None  # stream always ends with ResponseComplete
            await self.events.emit("on_model", messages, response)

            tokens_used += response.tokens_used
            cost_usd += response.cost_usd
            stop = self._budget_stop(steps, tokens_used, cost_usd)
            if stop is not None:
                yield Done(await self._abort(stop, steps, tokens_used, cost_usd, messages))
                return

            if response.is_final:
                answer = await self.guards.on_answer(response.text, GuardContext(self.policy))
                messages.append(Message(Role.ASSISTANT, answer, trusted=True))
                # Streaming validates best-effort: the answer was already streamed,
                # so we can't retry. `parsed` is None if validation fails. Use
                # run() if you need validation-driven retries.
                parsed: Any = None
                if self.output_validator is not None:
                    try:
                        parsed = await self._validate(answer)
                    except Exception:  # noqa: BLE001 - best-effort in streaming
                        parsed = None
                outcome = AgentOutcome(
                    status="completed",
                    answer=answer,
                    parsed=parsed,
                    steps=steps,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    transcript=messages,
                )
                await self.events.emit("on_final", outcome)
                await self._checkpoint(self.store, run_id, steps, tokens_used, cost_usd, messages)
                yield Done(outcome)
                return

            messages.append(self._assistant_tool_turn(response))
            for call in response.tool_calls:
                yield ToolStarted(call)
                msg = await self._handle_call(call)
                messages.append(msg)
                yield ToolFinished(msg)
            await self._checkpoint(self.store, run_id, steps, tokens_used, cost_usd, messages)

        yield Done(await self._abort("max_steps_reached", steps, tokens_used, cost_usd, messages))

    # ── the core loop ─────────────────────────────────────────────────────

    async def _loop(
        self,
        messages: list[Message],
        steps: int,
        tokens_used: int,
        cost_usd: float,
        run_id: str | None,
        store: Store | None,
        interrupt: Interrupt | None = None,
    ) -> AgentOutcome:
        output_retries = self.max_output_retries
        while steps < self.policy.max_steps:
            # Cooperative interrupt: checked at a safe boundary (between steps).
            if interrupt is not None and interrupt.triggered:
                return await self._abort("interrupted", steps, tokens_used, cost_usd, messages)
            steps += 1

            messages = await self._compact(messages)
            async with self._model_slot():
                response = await self.model(messages, tools=self.tool_schemas)
            await self.events.emit("on_model", messages, response)
            tokens_used += response.tokens_used
            cost_usd += response.cost_usd
            stop = self._budget_stop(steps, tokens_used, cost_usd)
            if stop is not None:
                return await self._abort(stop, steps, tokens_used, cost_usd, messages)

            if response.is_final:
                # GUARD: egress filter on the answer to the user (e.g. PII
                # redaction). No-op when no guard implements on_answer.
                answer = await self.guards.on_answer(
                    response.text, GuardContext(self.policy)
                )
                messages.append(Message(Role.ASSISTANT, answer, trusted=True))

                parsed: Any = None
                if self.output_validator is not None:
                    try:
                        parsed = await self._validate(answer)
                    except Exception as exc:  # noqa: BLE001 - validator failure -> retry/abort
                        if output_retries > 0:
                            output_retries -= 1
                            messages.append(
                                Message(Role.USER, self._retry_prompt(exc), trusted=True)
                            )
                            continue  # re-prompt the model with the validation error
                        outcome = AgentOutcome(
                            status="aborted",
                            reason="output_validation_failed",
                            answer=answer,
                            steps=steps,
                            tokens_used=tokens_used,
                            cost_usd=cost_usd,
                            transcript=messages,
                        )
                        await self.events.emit("on_final", outcome)
                        return outcome

                outcome = AgentOutcome(
                    status="completed",
                    answer=answer,
                    parsed=parsed,
                    steps=steps,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    transcript=messages,
                )
                await self.events.emit("on_final", outcome)
                return outcome

            messages.append(self._assistant_tool_turn(response))
            for call in response.tool_calls:
                messages.append(await self._handle_call(call))

            # Checkpoint after each completed step so a crash mid-next-step can resume.
            await self._checkpoint(store, run_id, steps, tokens_used, cost_usd, messages)

        return await self._abort("max_steps_reached", steps, tokens_used, cost_usd, messages)

    def _budget_stop(self, steps: int, tokens_used: int, cost_usd: float) -> str | None:
        """Return an abort reason if a resource budget is exceeded, else None."""
        if tokens_used > self.policy.max_tokens_budget:
            return "budget_exceeded"
        if self.policy.max_budget_usd is not None and cost_usd > self.policy.max_budget_usd:
            return "budget_usd_exceeded"
        return None

    async def _model_stream(self, messages: list[Message]) -> AsyncIterator[ModelStreamEvent]:
        """Yield model stream events, falling back to a one-shot call for models
        that don't implement ``stream``."""
        streamer = getattr(self.model, "stream", None)
        if streamer is not None:
            # Hold the slot for the whole stream (the connection stays open).
            async with self._model_slot():
                async for event in streamer(messages, tools=self.tool_schemas):
                    yield event
            return
        async with self._model_slot():
            response = await self.model(messages, tools=self.tool_schemas)
        if response.text:
            yield TextDelta(response.text)
        yield ResponseComplete(response)

    def _model_slot(self) -> AbstractAsyncContextManager[Any]:
        """The limiter context, or a no-op when no limiter is configured."""
        return self.model_limiter if self.model_limiter is not None else nullcontext()

    async def _compact(self, messages: list[Message]) -> list[Message]:
        """Apply the context strategy (if any) before a model call."""
        if self.context_strategy is None:
            return messages
        before = len(messages)
        compacted = await self.context_strategy.compact(messages)
        if compacted is not messages:  # strategy signals a change by new identity
            await self.events.emit("on_compact", before, len(compacted))
        return compacted

    async def _validate(self, answer: str) -> Any:
        """Run the output validator; returns the parsed value or raises."""
        assert self.output_validator is not None
        result = self.output_validator(answer)
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _retry_prompt(exc: BaseException) -> str:
        return (
            "Your previous response did not pass validation:\n"
            f"{exc}\n"
            "Correct the issue and respond again with only the valid output."
        )

    def _seed_messages(self, user_request: str | list[ContentPart]) -> list[Message]:
        # Trust boundary: only the system prompt and the genuine user request
        # are trusted as instructions. Tool output never is.
        return [
            Message(Role.SYSTEM, self.system_prompt, trusted=True),
            Message(Role.USER, user_request, trusted=True),
        ]

    @staticmethod
    def _assistant_tool_turn(response: ModelResponse) -> Message:
        # Record the assistant turn that requested the tools. Keep the full
        # ToolCall objects (id + name + args), not just names — provider
        # adapters need them to faithfully replay the turn next round.
        return Message(
            Role.ASSISTANT,
            response.text,
            trusted=True,
            meta={"tool_calls": list(response.tool_calls)},
        )

    async def _checkpoint(
        self,
        store: Store | None,
        run_id: str | None,
        steps: int,
        tokens_used: int,
        cost_usd: float,
        messages: list[Message],
    ) -> None:
        if store is None or run_id is None:
            return
        await store.save(
            run_id,
            {
                "run_id": run_id,
                "schema_version": SCHEMA_VERSION,
                "steps": steps,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "messages": transcript_to_dicts(messages),
            },
        )

    # ── per-call handling ─────────────────────────────────────────────────

    async def _handle_call(self, call: ToolCall) -> Message:
        await self.events.emit("on_tool_call", call)

        if self.tool_executor is None:
            return self._tool_msg(
                call, "REFUSED: no tool executor is configured.", ok=False
            )

        ctx = GuardContext(self.policy)

        # GUARD: pre-execution checks (tiers, PII, recipient-trust, ...).
        decision = await self.guards.before_call(call, ctx)
        await self.events.emit("on_guard_decision", call, decision)
        if decision.is_deny:
            return self._tool_msg(call, f"REFUSED: {decision.reason}", ok=False)
        if decision.is_confirm:
            approved = await self._confirm(call, decision.reason)
            await self.events.emit("on_confirm", call, approved)
            if not approved:
                return self._tool_msg(call, "User declined this action.", ok=False)

        # GUARD: execute inside the executor with policy-enforced limits.
        result = await self.tool_executor(
            call,
            network_allowlist=self.policy.network_allowlist,
            timeout_s=self.policy.tool_timeout_s,
        )

        # GUARD: sanitize output before it re-enters context (injection scan,
        # untrusted-data wrapping). Tool output is data, never instructions.
        content = await self.guards.after_output(call, result.content, ctx)
        msg = self._tool_msg(call, content, ok=result.ok)
        await self.events.emit("on_tool_result", call, msg)
        return msg

    async def _confirm(self, call: ToolCall, reason: str) -> bool:
        # Fail closed: a confirmation was required but no confirmer is wired.
        if self.confirm_fn is None:
            return False
        result = self.confirm_fn(self._describe(call, reason))
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tool_msg(call: ToolCall, content: str, *, ok: bool) -> Message:
        return Message(
            Role.TOOL,
            content,
            trusted=False,
            name=call.name,
            meta={"call_id": call.id, "ok": ok},
        )

    @staticmethod
    def _describe(call: ToolCall, reason: str = "") -> str:
        args_preview = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        prefix = f"{reason}. " if reason else ""
        return f"{prefix}About to run '{call.name}' with: {args_preview}. Approve?"

    async def _abort(
        self,
        reason: str,
        steps: int,
        tokens: int,
        cost_usd: float,
        transcript: list[Message],
    ) -> AgentOutcome:
        outcome = AgentOutcome(
            status="aborted",
            reason=reason,
            steps=steps,
            tokens_used=tokens,
            cost_usd=cost_usd,
            transcript=transcript,
        )
        await self.events.emit("on_final", outcome)
        return outcome
