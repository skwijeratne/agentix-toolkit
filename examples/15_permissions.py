"""15 — Dynamic permissions (can_use_tool).

Beyond static permission tiers, you often want a per-call decision based on the
*arguments* or external state: "refunds over $1000 need a manager", "this run
may only use read tools". Two guards cover this:

  * `CallbackGuard(check)` — your callback returns allow / deny / confirm per call.
  * `ToolAllowlistGuard({...})` — scope a run to a subset of tools.

They compose with the rest of the pipeline (most restrictive wins). Dependency-
free (MockModel).

Run:
    PYTHONPATH=src python examples/15_permissions.py
"""

from __future__ import annotations

from agentix import (
    Agent,
    CallbackGuard,
    Decision,
    MockModel,
    ModelResponse,
    Role,
    ToolAllowlistGuard,
    ToolCall,
    always_approve,
    tool,
)


@tool
def refund(amount: int, customer: str) -> str:
    """Issue a refund to a customer."""
    return f"refunded ${amount} to {customer}"


@tool
def delete_account(id: int) -> str:
    """Permanently delete an account."""
    return f"deleted account {id}"


# A per-call policy: small refunds auto-approve, medium need confirmation,
# large are denied outright — decided from the call's arguments.
def refund_policy(call: ToolCall, ctx) -> Decision:
    if call.name == "refund":
        amount = call.args.get("amount", 0)
        if amount > 1000:
            return Decision.deny("refunds over $1000 require a manager")
        if amount > 100:
            return Decision.confirm(f"approve a ${amount} refund?")
    return Decision.allow()


def _tool_out(outcome) -> str:
    return next(m for m in outcome.transcript if m.role == Role.TOOL).content


def demo_callback() -> None:
    print("== CallbackGuard: refund policy by amount ==")
    for amount in (50, 500, 5000):
        model = MockModel(
            [ModelResponse(tool_calls=[ToolCall("refund", {"amount": amount, "customer": "Acme"})]),
             ModelResponse(text="done")]
        )
        agent = Agent(
            model=model, system_prompt="sys", tools=[refund],
            guards=[CallbackGuard(refund_policy)],
            confirm_fn=always_approve,  # stand-in for a real approval prompt
        )
        print(f"  ${amount:<5} -> {_tool_out(agent.run_sync('refund'))}")


def demo_allowlist() -> None:
    print("\n== ToolAllowlistGuard: this run may only refund ==")
    model = MockModel(
        [ModelResponse(tool_calls=[ToolCall("delete_account", {"id": 7})]),
         ModelResponse(text="done")]
    )
    agent = Agent(
        model=model, system_prompt="sys", tools=[refund, delete_account],
        guards=[ToolAllowlistGuard({"refund"})],
    )
    print(" ", _tool_out(agent.run_sync("delete account 7")))


if __name__ == "__main__":
    demo_callback()
    demo_allowlist()
