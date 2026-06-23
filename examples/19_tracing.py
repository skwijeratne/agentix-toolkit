"""19 — OpenTelemetry tracing.

Turn an agent run into a span tree (root run -> model calls + tool calls) that
flows to your observability stack. agentix provides:

  * TracingModel(model)   — a span per model call (tokens, cost, latency)
  * tracing_events()      — a span per tool call (+ guard/confirm sub-events)
  * trace_run()           — the root run span

You configure the exporter (this demo prints spans to the console). All
dependency-free for agentix itself except the model is a MockModel.

Requirements:
  * pip install "agentix[otel]" opentelemetry-sdk

Run:
    python examples/19_tracing.py
    # or, with uv:  uv run --with opentelemetry-sdk python examples/19_tracing.py
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from agentix import (
    Agent,
    MockModel,
    ModelResponse,
    ToolCall,
    TracingModel,
    tool,
    trace_run,
    tracing_events,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: 18C"


async def main() -> None:
    # Configure OpenTelemetry to print spans to the console (your app would use
    # an OTLP exporter to send them to Jaeger/Honeycomb/Datadog/etc.).
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    model = TracingModel(
        MockModel(
            [
                ModelResponse(tool_calls=[ToolCall("get_weather", {"city": "Oslo"})], tokens_used=120, cost_usd=0.001),
                ModelResponse(text="It's 18C in Oslo.", tokens_used=40, cost_usd=0.0005),
            ]
        )
    )
    agent = Agent(
        model=model,
        system_prompt="You are a weather assistant.",
        tools=[get_weather],
        events=tracing_events(),  # tool spans
    )

    async with trace_run() as span:
        outcome = await agent.run("Weather in Oslo?")
        span.set_attribute("agentix.status", outcome.status)

    print("\n--- agent answer:", outcome.answer)


if __name__ == "__main__":
    asyncio.run(main())
