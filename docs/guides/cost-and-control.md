# Cost, budgets & human approval

This page covers staying in control of a run: knowing what it costs, setting
limits, stopping it, and pausing for a human.

## Tracking cost

Every run reports how much it used — in tokens **and** in real US dollars:

```python
outcome = await agent.run("...")
print(outcome.tokens_used, "tokens")
print(f"${outcome.cost_usd:.4f}")
```

The dollar figure is based on a built-in price table for popular models. If you
use a model that isn't listed, register its price with `register_price(...)`, and
costs from subagents roll up into the parent's total automatically.

## Setting budgets

A **policy** lets you cap a run so it can't run away. The agent stops cleanly when
it hits a limit:

```python
from agentix import AgentPolicy

policy = AgentPolicy(
    max_steps=25,            # at most 25 trips through the loop
    max_budget_usd=0.50,     # stop if it would cost more than 50 cents
)
agent = Agent(model=m, system_prompt="...", tools=[...], policy=policy)
```

→ Runnable example:
[`examples/14_cost_and_interrupt.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/14_cost_and_interrupt.py)

## Stopping a run early

Pass an `Interrupt` and trigger it (from another task, a timeout, a UI button) to
stop the agent at the next safe point:

```python
from agentix import Interrupt

stop = Interrupt()
outcome = await agent.run("...", interrupt=stop)
# elsewhere: stop.trigger()
```

## Pausing for a human (without blocking)

Some actions need a person's "yes" — sending money, deleting data. The simplest
way is to mark a tool **confirm-first** and provide a way to ask:

```python
from agentix import AgentPolicy, console_confirm

agent = Agent(
    model=m, system_prompt="...", tools=[wire_money],
    policy=AgentPolicy(confirm_first={"wire_money"}),
    confirm_fn=console_confirm,    # asks on the terminal
)
```

That works great for scripts. But in a **web app**, you can't keep a request
hanging while someone decides — they might take minutes, on a different page.

For that, turn on **suspend-and-resume**. When the agent hits an action needing
approval, it **saves its state and returns right away** with a `"suspended"`
status. Later — even in a different process — you approve or deny, and it picks up
exactly where it left off:

```python
agent = Agent(model=m, system_prompt="...", tools=[wire_money],
              policy=AgentPolicy(confirm_first={"wire_money"}),
              store=my_store, suspend_on_confirm=True)

outcome = await agent.run("Pay invoice 42", run_id="run-1")
if outcome.status == "suspended":
    # show outcome.pending to the user, get their decision, then later:
    outcome = await agent.resume("run-1", decisions={"c1": True})   # True = approved
```

→ Runnable example:
[`examples/24_suspend_resume.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/24_suspend_resume.py)
