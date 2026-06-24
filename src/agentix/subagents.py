"""Subagents — delegate a subtask to a child agent.

A subagent is just a child :class:`~agentix.agent.Agent` exposed to a parent as
a :class:`~agentix.tools.Tool`. When the parent calls it, the child runs its own
loop — with its own model, system prompt, tools, and guards — and returns its
final answer. Because it's an ordinary tool, the parent's loop, guards, and
executor treat it like any other; subagents compose with ``Limiter`` for fanout.

    research = subagent_tool(researcher_agent, name="research",
                             description="Delegate research questions.")
    lead = Agent(model=m, system_prompt="...", tools=[research])

The child's token/cost spend **rolls up** into the parent run: the tool returns a
:class:`~agentix.types.ToolResult` carrying the child's ``cost_usd`` /
``tokens_used``, which the loop adds to the parent's ``AgentOutcome`` totals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tools import Tool
from .types import ToolResult

if TYPE_CHECKING:
    from .agent import Agent


def subagent_tool(
    agent: Agent,
    *,
    name: str,
    description: str,
    input_name: str = "task",
) -> Tool:
    """Wrap an :class:`Agent` as a delegable tool.

    The generated tool takes a single string argument (``input_name``, default
    ``"task"``) — the instruction to delegate — and returns the child's answer.
    """

    async def _delegate(**kwargs: Any) -> ToolResult:
        task = kwargs[input_name]
        outcome = await agent.run(str(task))
        # Return a ToolResult so the child's spend rolls up into the parent.
        return ToolResult(
            name=name,
            content=outcome.answer or "",
            cost_usd=outcome.cost_usd,
            tokens_used=outcome.tokens_used,
        )

    return Tool(
        _delegate,
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                input_name: {
                    "type": "string",
                    "description": "The task or question to delegate to this subagent.",
                }
            },
            "required": [input_name],
        },
    )
