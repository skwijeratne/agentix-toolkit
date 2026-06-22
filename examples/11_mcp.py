"""11 — Using tools from an MCP server.

MCP (Model Context Protocol) servers expose tools over a transport. agentix can
connect to one, discover its tools, and hand them to an agent like any other —
the tool calls route back over the MCP connection.

This example connects to the reference filesystem server over stdio and lets a
Claude-backed agent use it. The discovered tools are only valid while the
connection is open, so the agent run happens inside the `async with` block.

Requirements:
  * pip install "agentix[mcp,anthropic]"
  * a runnable MCP server — e.g. Node's filesystem server:
      npx -y @modelcontextprotocol/server-filesystem /tmp
  * export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/11_mcp.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, MCPServer
from agentix.providers.anthropic import AnthropicModel


async def main() -> None:
    # Launch the filesystem MCP server as a subprocess over stdio, scoped to /tmp.
    async with MCPServer(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        name="filesystem",
    ) as server:
        tools = await server.list_tools()
        print("discovered MCP tools:", [t.name for t in tools])

        agent = Agent(
            model=AnthropicModel(max_tokens=1024),
            system_prompt="You can read and write files via the provided tools.",
            tools=tools,  # MCP tools plug straight in
        )

        # Must run inside the `async with` — the tools route to the live server.
        outcome = await agent.run("List the files in /tmp and summarize what you see.")
        print("\nanswer:", outcome.answer)


if __name__ == "__main__":
    asyncio.run(main())
