from typing import Literal

from agentix import Agent, MockModel, ModelResponse, Role, Tool, ToolCall, ToolRegistry, tool


def test_bare_decorator_generates_schema_from_hints_and_docstring() -> None:
    @tool
    def get_weather(city: str, days: int = 1) -> str:
        """Get the weather forecast for a city.

        Args:
            city: City name, e.g. 'Paris'.
            days: How many days to forecast.
        """
        return f"{city} for {days}d"

    assert isinstance(get_weather, Tool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the weather forecast for a city."

    params = get_weather.parameters
    assert params["type"] == "object"
    assert params["properties"]["city"] == {
        "type": "string",
        "description": "City name, e.g. 'Paris'.",
    }
    assert params["properties"]["days"]["type"] == "integer"
    # `city` is required (no default); `days` is optional (has a default).
    assert params["required"] == ["city"]


def test_decorated_function_is_still_callable() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert add(2, 3) == 5  # works as a plain function too


def test_type_mapping() -> None:
    @tool
    def f(s: str, i: int, x: float, b: bool, items: list[str]) -> None:
        """Types."""

    p = f.parameters["properties"]
    assert p["s"] == {"type": "string"}
    assert p["i"] == {"type": "integer"}
    assert p["x"] == {"type": "number"}
    assert p["b"] == {"type": "boolean"}
    assert p["items"] == {"type": "array", "items": {"type": "string"}}


def test_optional_and_literal() -> None:
    @tool
    def search(query: str, sort: Literal["new", "top"] = "new", limit: int | None = None) -> str:
        """Search.

        Args:
            query: the query
            sort: ordering
        """
        return query

    p = search.parameters
    assert p["properties"]["sort"] == {
        "type": "string",
        "enum": ["new", "top"],
        "description": "ordering",
    }
    # Optional[int] -> integer, and not required (has default None)
    assert p["properties"]["limit"] == {"type": "integer"}
    assert p["required"] == ["query"]


def test_decorator_with_overrides() -> None:
    @tool(name="weather", description="custom description")
    def get_weather(city: str) -> str:
        """ignored summary."""
        return city

    assert get_weather.name == "weather"
    assert get_weather.description == "custom description"


def test_registry_schemas_and_execution() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg = ToolRegistry([add])
    assert "add" in reg
    assert len(reg) == 1
    assert reg.schemas[0]["name"] == "add"


async def test_registry_executes_as_tool_executor() -> None:
    @tool
    def echo(text: str) -> str:
        """Echo."""
        return f"echo:{text}"

    reg = ToolRegistry([echo])
    result = await reg(ToolCall("echo", {"text": "hi"}, id="c1"))
    assert result.ok is True
    assert result.content == "echo:hi"


async def test_registry_unknown_tool_is_surfaced() -> None:
    reg = ToolRegistry()
    result = await reg(ToolCall("nope", {}))
    assert result.ok is False
    assert "unknown tool" in result.content


async def test_agent_with_tools_argument_end_to_end() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("add", {"a": 2, "b": 3}, id="c1")]),
            ModelResponse(text="The sum is 5."),
        ]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[add])

    # The agent derived schemas for the model...
    assert agent.tool_schemas[0]["name"] == "add"

    outcome = await agent.run("2+3?")
    assert outcome.answer == "The sum is 5."
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert tool_msg.content == "5"


async def test_agent_accepts_bare_functions_in_tools() -> None:
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    model = MockModel(
        [
            ModelResponse(tool_calls=[ToolCall("multiply", {"a": 4, "b": 5})]),
            ModelResponse(text="20"),
        ]
    )
    agent = Agent(model=model, system_prompt="sys", tools=[multiply])
    assert agent.tool_schemas[0]["name"] == "multiply"
    outcome = await agent.run("go")
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert tool_msg.content == "20"
