# Observability

When an agent does something surprising, you want to see *what happened* — which
tools it called, how long each step took, how many tokens it used, where the time
and money went. **Observability** is that visibility into a run.

## Audit hooks

The simplest option: `AgentEvents` lets you attach small callbacks that fire at key
moments (a tool is called, a guard makes a decision, the run finishes). Use them to
log, build an audit trail, or update a UI:

```python
from agentix import Agent, AgentEvents

def on_tool(call):
    print("tool:", call.name, call.args)

agent = Agent(model=m, system_prompt="...", tools=[...],
              events=AgentEvents(on_tool_call=on_tool))
```

## Tracing with OpenTelemetry

For real apps, you'll want **tracing**: a timeline of the run as nested "spans"
(the run contains model calls and tool calls, each with timing and token/cost
details), sent to a dashboard you already use.

agentix speaks **OpenTelemetry**, the industry-standard format that tools like
Jaeger, Honeycomb, and Datadog understand. Turn it on for an existing agent with a
single call:

```python
from agentix import instrument, trace_run

agent = instrument(agent)         # wraps the model + tools with tracing
async with trace_run():
    await agent.run("...")
```

`instrument(agent)` adds tracing without removing any callbacks you already set —
they keep working alongside it. Install the extra with `agentix-toolkit[otel]`, and
configure where traces go (the exporter) in your app as usual.

→ Runnable example:
[`examples/19_tracing.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/19_tracing.py)
