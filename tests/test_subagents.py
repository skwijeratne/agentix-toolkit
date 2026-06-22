from agentix import (
    Agent,
    MockModel,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
    subagent_tool,
    tool,
)


def _math_subagent() -> Agent:
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 20, "b": 22}, id="c1")]),
            ModelResponse(text="The result is 42."),
        ]
    )
    return Agent(model=model, system_prompt="You are a math expert.", tools=[add])


def test_subagent_tool_builds_a_tool() -> None:
    t = subagent_tool(_math_subagent(), name="math", description="Delegate math.")
    assert isinstance(t, Tool)
    assert t.name == "math"
    assert t.description == "Delegate math."
    assert t.parameters["required"] == ["task"]
    assert t.parameters["properties"]["task"]["type"] == "string"


async def test_subagent_runs_and_returns_answer() -> None:
    t = subagent_tool(_math_subagent(), name="math", description="Delegate math.")
    result = await t.func(task="what is 20 + 22?")
    assert result == "The result is 42."


async def test_parent_delegates_to_subagent() -> None:
    math = subagent_tool(_math_subagent(), name="math_expert", description="Delegate math.")

    parent_model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("math_expert", {"task": "20+22"}, id="p1")]),
            ModelResponse(text="My assistant says it's 42."),
        ]
    )
    lead = Agent(model=parent_model, system_prompt="You delegate.", tools=[math])
    outcome = await lead.run("Add 20 and 22 using your math expert.")

    assert outcome.answer == "My assistant says it's 42."
    # The subagent's answer came back through the tool boundary.
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert tool_msg.content == "The result is 42."


async def test_subagent_custom_input_name() -> None:
    t = subagent_tool(
        _math_subagent(), name="m", description="d", input_name="question"
    )
    assert t.parameters["required"] == ["question"]
    assert await t.func(question="20+22") == "The result is 42."
