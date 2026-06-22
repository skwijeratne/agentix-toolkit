"""MCP (Model Context Protocol) client support.

Connect to an MCP server, discover its tools, and expose them as agentix
:class:`~agentix.tools.Tool` objects that route calls back over the live
connection. The discovered tools plug straight into ``Agent(tools=...)``::

    async with MCPServer(command="my-mcp-server", args=["--flag"]) as server:
        agent = Agent(model=..., system_prompt="...", tools=await server.list_tools())
        await agent.run("...")            # run INSIDE the block

The tools are only valid while the connection is open, so keep the agent run
inside the ``async with``. Requires the ``mcp`` package (lazy-imported):
``pip install "agentix[mcp]"``.

MCP tools already carry a JSON Schema (``inputSchema``), which is exactly
agentix's ``parameters`` convention — so they flow through to any provider
adapter (e.g. Anthropic's ``input_schema``) unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any

from .errors import AgentError
from .tools import Tool

__all__ = ["MCPServer"]

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _render_content(result: Any) -> str:
    """Turn an MCP CallToolResult into a string; raise on an error result."""
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{getattr(block, 'type', 'unknown')} content]")
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        # Surfaced by the executor as a failed tool result (ok=False).
        raise RuntimeError(text or "MCP tool returned an error")
    return text


class MCPServer:
    """A connection to a single MCP server.

    Construct with a transport (``command=`` for stdio, ``url=`` for HTTP/SSE),
    or pass a pre-built ``session`` (anything exposing async ``initialize()`` /
    ``list_tools()`` / ``call_tool(name, args)``) for testing or to reuse an
    existing client. Use as an async context manager.
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        transport: str | None = None,
        name: str | None = None,
        session: Any = None,
    ) -> None:
        self.name = name
        self._command = command
        self._args = list(args or [])
        self._env = env
        self._url = url
        self._transport = transport or (
            "stdio" if command else "http" if url else None
        )
        self._provided_session = session
        self._session: Any = None
        self._stack: AsyncExitStack | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def __aenter__(self) -> MCPServer:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def connect(self) -> None:
        self._session = self._provided_session or await self._open_session()
        await self._session.initialize()

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._session = None

    async def _open_session(self) -> Any:
        self._stack = AsyncExitStack()
        try:
            from mcp import ClientSession
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'MCP support requires the "mcp" package. '
                'Install it with: pip install "agentix[mcp]"'
            ) from exc

        if self._transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            if not self._command:
                raise AgentError("stdio MCP server requires command=")
            params = StdioServerParameters(
                command=self._command, args=self._args, env=self._env
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif self._transport in ("http", "streamable-http"):
            from mcp.client.streamable_http import streamablehttp_client

            if not self._url:
                raise AgentError("http MCP server requires url=")
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._url)
            )
        elif self._transport == "sse":
            from mcp.client.sse import sse_client

            if not self._url:
                raise AgentError("sse MCP server requires url=")
            read, write = await self._stack.enter_async_context(sse_client(self._url))
        else:
            raise AgentError(f"unknown MCP transport: {self._transport!r}")

        return await self._stack.enter_async_context(ClientSession(read, write))

    # ── tool discovery ────────────────────────────────────────────────────

    async def list_tools(self) -> list[Tool]:
        """Discover the server's tools as agentix :class:`Tool` objects."""
        if self._session is None:
            raise AgentError(
                "MCPServer is not connected; use `async with` or call connect()"
            )
        result = await self._session.list_tools()
        return [self._to_tool(t) for t in result.tools]

    # Friendly alias for the common `Agent(tools=await server.tools())` hookup.
    tools = list_tools

    def _to_tool(self, mcp_tool: Any) -> Tool:
        session = self._session
        tool_name = mcp_tool.name

        async def _call(**kwargs: Any) -> str:
            return _render_content(await session.call_tool(tool_name, kwargs))

        return Tool(
            _call,
            name=tool_name,
            description=getattr(mcp_tool, "description", "") or "",
            parameters=getattr(mcp_tool, "inputSchema", None) or dict(_EMPTY_OBJECT_SCHEMA),
        )
