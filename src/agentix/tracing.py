"""OpenTelemetry tracing.

Turn an agent run into a span tree so you can see latency, tokens, cost, and
tool/guard activity in your existing observability stack. Three pieces, all
opt-in and composable:

  * :class:`TracingModel` — wrap a model; each call becomes a ``agentix.model``
    span with token/cost attributes.
  * :func:`tracing_events` — an :class:`~agentix.events.AgentEvents` that opens a
    ``agentix.tool.<name>`` span per tool call (with guard/confirm sub-events).
  * :func:`trace_run` — an async context manager for the root ``agentix.run``
    span, under which the model/tool spans nest.

Usage::

    agent = Agent(model=TracingModel(my_model), system_prompt="...",
                  tools=[...], events=tracing_events())
    async with trace_run():
        outcome = await agent.run("...")

Requires ``opentelemetry-api`` (``pip install "agentix[otel]"``); you configure
the TracerProvider/exporter in your app. The ``opentelemetry`` import is
deferred, so importing agentix never requires it.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from .events import AgentEvents
from .model import ModelFn, ToolSchema
from .types import Message, ModelResponse, ToolCall

if TYPE_CHECKING:
    from .agent import Agent


def _default_tracer() -> Any:
    try:
        from opentelemetry import trace
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "OpenTelemetry tracing requires opentelemetry-api. "
            'Install it with: pip install "agentix[otel]"'
        ) from exc
    return trace.get_tracer("agentix")


class TracingModel:
    """Wrap a model so each call is recorded as a span with token/cost attrs."""

    def __init__(self, model: ModelFn, *, tracer: Any = None) -> None:
        self.model = model
        self._tracer = tracer

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        tracer = self._tracer if self._tracer is not None else _default_tracer()
        with tracer.start_as_current_span("agentix.model") as span:
            try:
                response = await self.model(messages, tools=tools)
            except Exception as exc:  # noqa: BLE001 - record then re-raise
                span.record_exception(exc)
                raise
            span.set_attribute("agentix.tokens_used", response.tokens_used)
            span.set_attribute("agentix.input_tokens", response.input_tokens)
            span.set_attribute("agentix.output_tokens", response.output_tokens)
            span.set_attribute("agentix.cost_usd", response.cost_usd)
            span.set_attribute("agentix.is_final", response.is_final)
            span.set_attribute("agentix.num_tool_calls", len(response.tool_calls))
            return response


def tracing_events(*, tracer: Any = None) -> AgentEvents:
    """An ``AgentEvents`` that spans each tool call (with guard/confirm events).

    Tool spans nest under whatever span is current (e.g. the :func:`trace_run`
    root). Create a fresh instance per agent."""
    the_tracer = tracer if tracer is not None else _default_tracer()
    spans: dict[int, Any] = {}

    def on_tool_call(call: ToolCall) -> None:
        span = the_tracer.start_span(f"agentix.tool.{call.name}")
        span.set_attribute("agentix.tool.name", call.name)
        spans[id(call)] = span

    def on_guard_decision(call: ToolCall, decision: Any) -> None:
        span = spans.get(id(call))
        if span is not None:
            span.add_event(
                "guard_decision",
                {"decision": decision.type.value, "reason": decision.reason},
            )

    def on_confirm(call: ToolCall, approved: bool) -> None:
        span = spans.get(id(call))
        if span is not None:
            span.add_event("confirm", {"approved": approved})

    def on_tool_result(call: ToolCall, result: Message) -> None:
        span = spans.pop(id(call), None)
        if span is not None:
            span.set_attribute("agentix.tool.ok", bool(result.meta.get("ok", True)))
            span.end()

    return AgentEvents(
        on_tool_call=on_tool_call,
        on_guard_decision=on_guard_decision,
        on_confirm=on_confirm,
        on_tool_result=on_tool_result,
    )


@asynccontextmanager
async def trace_run(name: str = "agentix.run", *, tracer: Any = None) -> AsyncIterator[Any]:
    """Open the root span for a run; yields the span so you can add attributes."""
    the_tracer = tracer if tracer is not None else _default_tracer()
    with the_tracer.start_as_current_span(name) as span:
        yield span


def instrument(agent: Agent, *, tracer: Any = None) -> Agent:
    """Wire tracing into an existing agent in one call: wrap its model in
    :class:`TracingModel` and merge :func:`tracing_events` into its events (any
    callbacks you already set are preserved and run alongside). Mutates and
    returns the agent. Still wrap the run in :func:`trace_run` for a root span.

        agent = instrument(Agent(model=m, system_prompt="...", tools=[...]))
        async with trace_run():
            await agent.run("...")
    """
    if not isinstance(agent.model, TracingModel):
        agent.model = TracingModel(agent.model, tracer=tracer)
    agent.events = _merge_events(agent.events, tracing_events(tracer=tracer))
    return agent


def _merge_events(a: AgentEvents, b: AgentEvents) -> AgentEvents:
    """Combine two AgentEvents so both callbacks fire for each hook."""
    merged: dict[str, Any] = {}
    for f in dataclasses.fields(AgentEvents):
        merged[f.name] = _chain(getattr(a, f.name), getattr(b, f.name))
    return AgentEvents(**merged)


def _chain(
    ca: Callable[..., Any] | None, cb: Callable[..., Any] | None
) -> Callable[..., Awaitable[None]] | Callable[..., Any] | None:
    if ca is None:
        return cb
    if cb is None:
        return ca

    async def combined(*args: Any) -> None:
        for c in (ca, cb):
            result = c(*args)
            if inspect.isawaitable(result):
                await result

    return combined
