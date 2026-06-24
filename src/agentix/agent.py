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
import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
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
from .memory import Memory, MemoryRecord
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
from .types import AgentOutcome, Message, ModelResponse, PendingApproval, Role, ToolCall
from .validation import OutputValidator, json_output, pydantic_output


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
        suspend_on_confirm: bool = False,
        events: AgentEvents | None = None,
        store: Store | None = None,
        model_limiter: Limiter | None = None,
        context_strategy: ContextStrategy | None = None,
        output_validator: OutputValidator | None = None,
        max_output_retries: int = 1,
        response_model: Any = None,
        memory: Memory | None = None,
        memory_limit: int = 5,
        remember_exchange: bool = False,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.policy = policy or AgentPolicy()
        self.store = store
        # Optional cross-session memory: recalled before each run/stream and
        # injected as system context; set remember_exchange to also persist each
        # completed exchange. Recall happens on run()/stream(), not resume().
        self.memory = memory
        self.memory_limit = memory_limit
        self.remember_exchange = remember_exchange
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
        # When True, a tool requiring confirmation pauses the run: the loop
        # checkpoints and returns status="suspended" instead of awaiting
        # confirm_fn inline — so a web/serverless caller can persist and resume on
        # a later request (see resume(decisions=...)). Requires a store + run_id.
        # Applies to run()/resume(); stream() still confirms inline.
        self.suspend_on_confirm = suspend_on_confirm
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

        # `response_model` is the one-knob structured-output path. It (a) derives
        # an output_validator so `outcome.parsed` is the typed/validated value
        # (with re-prompt-on-failure via max_output_retries), (b) injects the JSON
        # schema as a system instruction (works for *any* model), and (c) turns on
        # native provider enforcement when the adapter supports it.
        self.response_model = response_model
        self.response_schema: dict[str, Any] | None = None
        if response_model is not None:
            if isinstance(response_model, dict):
                self.response_schema = response_model
                if self.output_validator is None:
                    self.output_validator = json_output
            else:  # a Pydantic model class (duck-typed; agentix never imports it)
                self.response_schema = response_model.model_json_schema()
                if self.output_validator is None:
                    self.output_validator = pydantic_output(response_model)
            bind = getattr(self.model, "with_response_format", None)
            if bind is not None:  # native enforcement (output_config / response_format)
                self.model = bind(self.response_schema)

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
        messages = await self._seed_messages(user_request)
        outcome = await self._loop(messages, 0, 0, 0.0, run_id, self.store, interrupt)
        await self._remember(user_request, outcome)
        return outcome

    async def resume(
        self,
        run_id: str,
        *,
        decisions: Mapping[str, bool] | None = None,
        store: Store | None = None,
        interrupt: Interrupt | None = None,
    ) -> AgentOutcome:
        """Reload a checkpointed run and continue the loop from where it stopped.

        For a run **suspended** awaiting confirmation (``suspend_on_confirm``),
        pass ``decisions`` mapping each pending ``call.id`` to ``True`` (approve)
        or ``False`` (deny). A pending call with no entry is denied (fail closed).
        This may be called on a *fresh* ``Agent`` in a later process — the paused
        state lives entirely in the store.
        """
        effective = store or self.store
        if effective is None:
            raise AgentError("resume() requires a store (on the Agent or as an argument)")
        state = await effective.load(run_id)
        if state is None:
            raise AgentError(f"no saved run found for run_id {run_id!r}")
        messages = transcript_from_dicts(state["messages"])
        steps = int(state["steps"])
        tokens_used = int(state["tokens_used"])
        cost_usd = float(state.get("cost_usd", 0.0))

        # If the transcript tail is an assistant tool-turn with no results yet,
        # this run was suspended for approval — finish that turn with `decisions`
        # before continuing the loop.
        if self._has_unfinished_tool_turn(messages):
            calls: list[ToolCall] = list(messages[-1].meta.get("tool_calls", []))
            for call in calls:
                msg = await self._handle_call(call, approvals=decisions)
                messages.append(msg)
                tokens_used += _tool_msg_tokens(msg)
                cost_usd += _tool_msg_cost(msg)
            await self._checkpoint(effective, run_id, steps, tokens_used, cost_usd, messages)

        return await self._loop(
            messages, steps, tokens_used, cost_usd, run_id, effective, interrupt
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

    def resume_sync(
        self,
        run_id: str,
        *,
        decisions: Mapping[str, bool] | None = None,
        store: Store | None = None,
    ) -> AgentOutcome:
        """Blocking wrapper around :meth:`resume`. Do not call from inside a
        running event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resume(run_id, decisions=decisions, store=store))
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
        messages = await self._seed_messages(user_request)
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
                tokens_used += _tool_msg_tokens(msg)
                cost_usd += _tool_msg_cost(msg)
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

            # Suspend path: if any call needs human confirmation, pause *before*
            # executing anything (no partial side effects). Checkpoint with the
            # assistant tool-turn as the tail so resume() can finish it later.
            if self.suspend_on_confirm:
                pending = await self._pending_approvals(response.tool_calls)
                if pending:
                    if store is None or run_id is None:
                        raise AgentError(
                            "suspend_on_confirm requires a store and a run_id so the "
                            "paused run can be persisted and resumed; pass run_id= to "
                            "run() on an Agent constructed with store=."
                        )
                    await self._checkpoint(store, run_id, steps, tokens_used, cost_usd, messages)
                    return await self._suspend(pending, steps, tokens_used, cost_usd, messages)

            for call in response.tool_calls:
                msg = await self._handle_call(call)
                messages.append(msg)
                tokens_used += _tool_msg_tokens(msg)
                cost_usd += _tool_msg_cost(msg)

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

    async def _seed_messages(
        self, user_request: str | list[ContentPart]
    ) -> list[Message]:
        # Trust boundary: only the system prompt and the genuine user request
        # are trusted as instructions. Tool output never is.
        system_text = self.system_prompt
        if self.response_schema is not None:
            system_text += "\n\n" + _schema_instruction(self.response_schema)
        messages = [Message(Role.SYSTEM, system_text, trusted=True)]

        # Cross-session memory: recall records relevant to this request and
        # inject them as (trusted) system context before the user's turn.
        if self.memory is not None:
            query = Message(Role.USER, user_request).text
            recalled = await self.memory.recall(query, limit=self.memory_limit)
            if recalled:
                messages.append(
                    Message(Role.SYSTEM, _format_memories(recalled), trusted=True)
                )

        messages.append(Message(Role.USER, user_request, trusted=True))
        return messages

    async def _remember(
        self, user_request: str | list[ContentPart], outcome: AgentOutcome
    ) -> None:
        """Persist a completed exchange to memory (opt-in)."""
        if self.memory is None or not self.remember_exchange:
            return
        if outcome.status != "completed" or outcome.answer is None:
            return
        request = Message(Role.USER, user_request).text
        await self.memory.write(
            f"User asked: {request}\nAssistant answered: {outcome.answer}",
            metadata={"kind": "exchange"},
        )

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

    async def _handle_call(
        self, call: ToolCall, *, approvals: Mapping[str, bool] | None = None
    ) -> Message:
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
            approved = await self._resolve_confirm(call, decision.reason, approvals)
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
        msg = self._tool_msg(
            call, content, ok=result.ok,
            cost_usd=result.cost_usd, tokens_used=result.tokens_used,
        )
        await self.events.emit("on_tool_result", call, msg)
        return msg

    async def _resolve_confirm(
        self, call: ToolCall, reason: str, approvals: Mapping[str, bool] | None
    ) -> bool:
        # On resume, a pre-supplied human decision wins; otherwise ask confirm_fn.
        # A pending call with no decision falls through and (absent confirm_fn)
        # fails closed.
        cid = call.id
        if approvals and cid is not None and cid in approvals:
            return bool(approvals[cid])
        return await self._confirm(call, reason)

    async def _confirm(self, call: ToolCall, reason: str) -> bool:
        # Fail closed: a confirmation was required but no confirmer is wired.
        if self.confirm_fn is None:
            return False
        result = self.confirm_fn(self._describe(call, reason))
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def _pending_approvals(self, calls: Sequence[ToolCall]) -> list[PendingApproval]:
        """Classify a turn's calls (before_call only) and return those that need
        human confirmation. Does not emit events or execute — the real handling
        happens in :meth:`_handle_call` once the run resumes."""
        if self.tool_executor is None:
            return []
        pending: list[PendingApproval] = []
        for call in calls:
            decision = await self.guards.before_call(call, GuardContext(self.policy))
            if decision.is_confirm:
                pending.append(PendingApproval(call, decision.reason))
        return pending

    async def _suspend(
        self,
        pending: list[PendingApproval],
        steps: int,
        tokens: int,
        cost_usd: float,
        transcript: list[Message],
    ) -> AgentOutcome:
        outcome = AgentOutcome(
            status="suspended",
            reason="awaiting_confirmation",
            steps=steps,
            tokens_used=tokens,
            cost_usd=cost_usd,
            transcript=transcript,
            pending=pending,
        )
        await self.events.emit("on_suspend", outcome)
        return outcome

    @staticmethod
    def _has_unfinished_tool_turn(messages: list[Message]) -> bool:
        """True if the transcript tail is an assistant tool-turn with no results
        yet — the signature of a run suspended for confirmation."""
        return bool(messages) and messages[-1].role is Role.ASSISTANT and bool(
            messages[-1].meta.get("tool_calls")
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tool_msg(
        call: ToolCall,
        content: str,
        *,
        ok: bool,
        cost_usd: float = 0.0,
        tokens_used: int = 0,
    ) -> Message:
        return Message(
            Role.TOOL,
            content,
            trusted=False,
            name=call.name,
            meta={
                "call_id": call.id,
                "ok": ok,
                "cost_usd": cost_usd,
                "tokens_used": tokens_used,
            },
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


def _format_memories(records: list[MemoryRecord]) -> str:
    """Render recalled memory records as a system-context block."""
    lines = "\n".join(f"- {r.content}" for r in records)
    return f"Relevant information recalled from memory:\n{lines}"


def _tool_msg_cost(msg: Message) -> float:
    return float(msg.meta.get("cost_usd", 0.0) or 0.0)


def _tool_msg_tokens(msg: Message) -> int:
    return int(msg.meta.get("tokens_used", 0) or 0)


def _schema_instruction(schema: dict[str, Any]) -> str:
    """A provider-agnostic instruction to emit JSON conforming to ``schema``."""
    return (
        "Respond with ONLY a single JSON object that conforms to this JSON "
        "Schema — no prose, no markdown code fences:\n"
        f"{json.dumps(schema)}"
    )
