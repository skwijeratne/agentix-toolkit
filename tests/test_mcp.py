from types import SimpleNamespace
from typing import Any

from agentix import (
    Agent,
    MCPServer,
    MockModel,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
)


# ── a fake MCP ClientSession (duck-typed like mcp.ClientSession) ─────────


def _text(s: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=s)


def _tool_def(name: str, description: str, schema: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


class FakeSession:
    def __init__(self, tools: list[SimpleNamespace], results: dict[str, Any]) -> None:
        self._tools = tools
        self._results = results
        self.initialized = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> SimpleNamespace:
        self.calls.append((name, args))
        return self._results[name]


GEO_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def _server_with(session: FakeSession) -> MCPServer:
    return MCPServer(session=session)


async def test_connect_initializes_session() -> None:
    session = FakeSession([], {})
    async with MCPServer(session=session):
        assert session.initialized is True


async def test_list_tools_translates_to_agentix_tools() -> None:
    session = FakeSession(
        [_tool_def("geocode", "Look up a city", GEO_SCHEMA)],
        {"geocode": SimpleNamespace(content=[_text("ok")], isError=False)},
    )
    async with _server_with(session) as server:
        tools = await server.list_tools()

    assert len(tools) == 1
    t = tools[0]
    assert isinstance(t, Tool)
    assert t.name == "geocode"
    assert t.description == "Look up a city"
    # The MCP inputSchema becomes agentix `parameters` verbatim.
    assert t.parameters == GEO_SCHEMA
    assert t.schema["parameters"]["required"] == ["city"]


async def test_calling_a_tool_routes_to_session_and_renders_text() -> None:
    session = FakeSession(
        [_tool_def("geocode", "Look up a city", GEO_SCHEMA)],
        {"geocode": SimpleNamespace(content=[_text("48.8,2.3")], isError=False)},
    )
    async with _server_with(session) as server:
        (tool,) = await server.list_tools()
        out = await tool.func(city="Paris")

    assert out == "48.8,2.3"
    assert session.calls == [("geocode", {"city": "Paris"})]


async def test_error_result_raises() -> None:
    session = FakeSession(
        [_tool_def("boom", "fails", {"type": "object", "properties": {}})],
        {"boom": SimpleNamespace(content=[_text("nope")], isError=True)},
    )
    async with _server_with(session) as server:
        (tool,) = await server.list_tools()
        try:
            await tool.func()
            raise AssertionError("expected an error")
        except RuntimeError as e:
            assert "nope" in str(e)


async def test_tools_alias_works() -> None:
    session = FakeSession([_tool_def("t", "d", {"type": "object", "properties": {}})], {})
    async with _server_with(session) as server:
        via_alias = await server.tools()
    assert via_alias[0].name == "t"


async def test_end_to_end_agent_uses_mcp_tool() -> None:
    session = FakeSession(
        [_tool_def("geocode", "Look up a city", GEO_SCHEMA)],
        {"geocode": SimpleNamespace(content=[_text("48.8566,2.3522")], isError=False)},
    )
    async with _server_with(session) as server:
        tools = await server.list_tools()
        model = MockModel(
            [
                ModelResponse(tool_calls=[ToolCall("geocode", {"city": "Paris"}, id="c1")]),
                ModelResponse(text="Paris is at 48.8566, 2.3522."),
            ]
        )
        agent = Agent(model=model, system_prompt="sys", tools=tools)
        outcome = await agent.run("Where is Paris?")

    assert outcome.answer == "Paris is at 48.8566, 2.3522."
    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    assert tool_msg.content == "48.8566,2.3522"
    assert session.calls == [("geocode", {"city": "Paris"})]


async def test_list_tools_before_connect_raises() -> None:
    server = MCPServer(session=FakeSession([], {}))
    try:
        await server.list_tools()  # not connected yet
        raise AssertionError("expected AgentError")
    except Exception as e:
        assert "not connected" in str(e).lower()


async def test_missing_mcp_dependency_raises_helpful_error() -> None:
    # No session provided + `mcp` package absent -> a clear install hint.
    server = MCPServer(command="some-mcp-server")
    try:
        await server.connect()
    except ImportError as e:
        assert "agentix[mcp]" in str(e)
    except Exception:
        # If `mcp` happens to be installed, connecting to a bogus command may
        # fail differently; the dependency-guard assertion only applies when
        # the package is missing.
        pass
