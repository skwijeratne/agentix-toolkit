from agentix import (
    Agent,
    AgentPolicy,
    CallbackGuard,
    Decision,
    MockModel,
    ModelResponse,
    Role,
    ToolAllowlistGuard,
    ToolCall,
    always_approve,
    always_deny,
    tool,
)
from agentix.guards import GuardContext


def _ctx() -> GuardContext:
    return GuardContext(AgentPolicy())


# ── CallbackGuard ────────────────────────────────────────────────────────


async def test_callback_returning_decision() -> None:
    def check(call: ToolCall, ctx: GuardContext) -> Decision:
        return Decision.deny("nope") if call.name == "danger" else Decision.allow()

    guard = CallbackGuard(check)
    assert (await guard.before_call(ToolCall("danger", {}), _ctx())).is_deny
    assert (await guard.before_call(ToolCall("safe", {}), _ctx())).is_allow


async def test_callback_returning_bool() -> None:
    guard = CallbackGuard(lambda call, ctx: call.name == "ok")
    assert (await guard.before_call(ToolCall("ok", {}), _ctx())).is_allow
    assert (await guard.before_call(ToolCall("no", {}), _ctx())).is_deny


async def test_async_callback_is_awaited() -> None:
    async def check(call: ToolCall, ctx: GuardContext) -> Decision:
        return Decision.confirm("are you sure?")

    d = await CallbackGuard(check).before_call(ToolCall("x", {}), _ctx())
    assert d.is_confirm
    assert d.reason == "are you sure?"


async def test_callback_can_inspect_args() -> None:
    def check(call: ToolCall, ctx: GuardContext) -> Decision:
        amount = call.args.get("amount", 0)
        if amount > 1000:
            return Decision.deny("too large")
        if amount > 100:
            return Decision.confirm("needs approval")
        return Decision.allow()

    g = CallbackGuard(check)
    assert (await g.before_call(ToolCall("refund", {"amount": 50}), _ctx())).is_allow
    assert (await g.before_call(ToolCall("refund", {"amount": 500}), _ctx())).is_confirm
    assert (await g.before_call(ToolCall("refund", {"amount": 5000}), _ctx())).is_deny


async def test_callback_bad_return_type_raises() -> None:
    g = CallbackGuard(lambda call, ctx: "yes")  # type: ignore[arg-type,return-value]
    try:
        await g.before_call(ToolCall("x", {}), _ctx())
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "Decision or bool" in str(e)


# ── ToolAllowlistGuard ───────────────────────────────────────────────────


async def test_allowlist_allows_listed_denies_others() -> None:
    g = ToolAllowlistGuard({"read", "search"})
    assert (await g.before_call(ToolCall("read", {}), _ctx())).is_allow
    d = await g.before_call(ToolCall("delete", {}), _ctx())
    assert d.is_deny
    assert "allowed tool set" in d.reason


# ── through the Agent loop ───────────────────────────────────────────────


@tool
def refund(amount: int, customer: str) -> str:
    """Issue a refund."""
    return f"refunded {amount} to {customer}"


async def test_callback_guard_denies_in_loop() -> None:
    def check(call: ToolCall, ctx: GuardContext) -> Decision:
        if call.name == "refund" and call.args["amount"] > 1000:
            return Decision.deny("over the limit")
        return Decision.allow()

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("refund", {"amount": 5000, "customer": "x"})]),
            ModelResponse(text="ok"),
        ]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[refund], guards=[CallbackGuard(check)])
    outcome = await agent.run("refund 5000")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert "REFUSED" in tool_msg.content
    assert "over the limit" in tool_msg.content
    assert "refunded" not in tool_msg.content  # never ran


async def test_callback_guard_confirm_path_in_loop() -> None:
    def check(call: ToolCall, ctx: GuardContext) -> Decision:
        return Decision.confirm("approve this refund?")

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("refund", {"amount": 200, "customer": "y"})]),
            ModelResponse(text="done"),
        ]
    )
    # Declined:
    declined = Agent(model=MockModel(
        [ModelResponse(tool_calls=[ToolCall("refund", {"amount": 200, "customer": "y"})]),
         ModelResponse(text="done")]),
        system_prompt="sys", tools=[refund],
        guards=[CallbackGuard(check)], confirm_fn=always_deny)
    out_d = await declined.run("refund")
    assert "declined" in next(m for m in out_d.transcript if m.role == Role.TOOL).content.lower()

    # Approved:
    approved = Agent(model=model, system_prompt="sys", tools=[refund],
                     guards=[CallbackGuard(check)], confirm_fn=always_approve)
    out_a = await approved.run("refund")
    assert "refunded 200" in next(m for m in out_a.transcript if m.role == Role.TOOL).content


async def test_allowlist_guard_in_loop() -> None:
    @tool
    def delete_account(id: int) -> str:
        """Delete an account."""
        return "deleted"

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("delete_account", {"id": 1})]),
            ModelResponse(text="ok"),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[refund, delete_account],
        guards=[ToolAllowlistGuard({"refund"})],  # delete_account not allowed
    )
    outcome = await agent.run("delete account 1")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert "REFUSED" in tool_msg.content
    assert "deleted" not in tool_msg.content


async def test_callback_composes_with_other_guards() -> None:
    # CallbackGuard allows, but a prohibited tier still denies (most restrictive wins).
    from agentix import secure_defaults

    @tool
    def wire(amount: int) -> str:
        """Wire money."""
        return "sent"

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("wire", {"amount": 1})]),
            ModelResponse(text="ok"),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tools=[wire],
        policy=AgentPolicy(prohibited={"wire"}),
        guards=[CallbackGuard(lambda c, ctx: True), *secure_defaults()],
    )
    outcome = await agent.run("wire 1")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert "REFUSED" in tool_msg.content  # tier deny wins over the callback's allow
