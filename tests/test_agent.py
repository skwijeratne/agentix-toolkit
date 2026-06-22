from __future__ import annotations

from agentix import (
    Agent,
    AgentPolicy,
    LocalToolExecutor,
    Message,
    MockModel,
    ModelResponse,
    Role,
    ToolCall,
)


def _scripted(*responses: ModelResponse) -> MockModel:
    return MockModel(list(responses))


async def test_single_turn_final_answer() -> None:
    agent = Agent(model=_scripted(ModelResponse(text="hi there")), system_prompt="sys")
    outcome = await agent.run("hello")
    assert outcome.status == "completed"
    assert outcome.answer == "hi there"
    assert outcome.steps == 1


async def test_multi_step_tool_then_answer() -> None:
    model = _scripted(
        ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
        ModelResponse(text="The answer is 5."),
    )
    executor = LocalToolExecutor({"add": lambda a, b: a + b})
    agent = Agent(model=model, system_prompt="sys", tool_executor=executor)

    outcome = await agent.run("2+3?")
    assert outcome.status == "completed"
    assert outcome.answer == "The answer is 5."
    assert outcome.steps == 2

    # The tool result re-entered context as an untrusted TOOL message.
    tool_msgs = [m for m in outcome.transcript if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "5"
    assert tool_msgs[0].trusted is False
    assert tool_msgs[0].meta["ok"] is True


async def test_async_tool_is_awaited() -> None:
    async def slow_echo(text: str) -> str:
        return f"echo:{text}"

    model = _scripted(
        ModelResponse(tool_calls=[ToolCall("echo", {"text": "hi"})]),
        ModelResponse(text="done"),
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tool_executor=LocalToolExecutor({"echo": slow_echo}),
    )
    outcome = await agent.run("go")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert tool_msg.content == "echo:hi"


async def test_unknown_tool_is_surfaced_not_raised() -> None:
    model = _scripted(
        ModelResponse(tool_calls=[ToolCall("missing", {})]),
        ModelResponse(text="handled"),
    )
    agent = Agent(
        model=model,
        system_prompt="sys",
        tool_executor=LocalToolExecutor({}),
    )
    outcome = await agent.run("go")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert "unknown tool" in tool_msg.content
    assert tool_msg.meta["ok"] is False


async def test_no_executor_refuses_tool_call() -> None:
    model = _scripted(
        ModelResponse(tool_calls=[ToolCall("add", {"a": 1, "b": 1})]),
        ModelResponse(text="ok"),
    )
    agent = Agent(model=model, system_prompt="sys")  # no executor
    outcome = await agent.run("go")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert "no tool executor" in tool_msg.content.lower()
    assert tool_msg.meta["ok"] is False


async def test_budget_exceeded_aborts() -> None:
    model = _scripted(ModelResponse(text="x", tokens_used=500))
    agent = Agent(
        model=model,
        system_prompt="sys",
        policy=AgentPolicy(max_tokens_budget=100),
    )
    outcome = await agent.run("go")
    assert outcome.status == "aborted"
    assert outcome.reason == "budget_exceeded"


async def test_max_steps_aborts() -> None:
    # Model that always asks for a tool -> never terminates on its own.
    def always_tool(_messages: object) -> ModelResponse:
        return ModelResponse(tool_calls=[ToolCall("noop", {})])

    agent = Agent(
        model=MockModel(always_tool),
        system_prompt="sys",
        tool_executor=LocalToolExecutor({"noop": lambda: "ok"}),
        policy=AgentPolicy(max_steps=3),
    )
    outcome = await agent.run("go")
    assert outcome.status == "aborted"
    assert outcome.reason == "max_steps_reached"
    assert outcome.steps == 3


async def test_tool_schemas_are_passed_to_model() -> None:
    seen: list[object] = []

    def recorder(_messages: object) -> ModelResponse:
        return ModelResponse(text="done")

    class RecordingModel(MockModel):
        async def __call__(self, messages, *, tools=()):  # type: ignore[override]
            seen.append(list(tools))
            return await super().__call__(messages, tools=tools)

    schemas = [{"name": "add", "description": "add two ints", "parameters": {}}]
    agent = Agent(
        model=RecordingModel([ModelResponse(text="done")]),
        system_prompt="sys",
        tool_schemas=schemas,
    )
    await agent.run("go")
    assert seen == [schemas]


def test_run_sync_wrapper() -> None:
    agent = Agent(model=_scripted(ModelResponse(text="sync ok")), system_prompt="sys")
    outcome = agent.run_sync("hello")
    assert outcome.answer == "sync ok"


def test_system_and_user_messages_are_trusted() -> None:
    agent = Agent(model=_scripted(ModelResponse(text="ok")), system_prompt="sys-prompt")
    outcome = agent.run_sync("the request")
    sys_msg = outcome.transcript[0]
    user_msg = outcome.transcript[1]
    assert isinstance(sys_msg, Message)
    assert sys_msg.role == Role.SYSTEM and sys_msg.trusted is True
    assert user_msg.role == Role.USER and user_msg.trusted is True
