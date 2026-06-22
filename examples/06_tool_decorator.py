"""06 — Defining tools with the @tool decorator.

This is the P2 payoff: instead of hand-writing a JSON schema and registering a
function by name, you decorate a typed function once. The decorator derives the
tool's name, description (from the docstring), and parameter schema (from the
type hints) — and the function stays callable.

Shows: basic types, optional parameters, Literal/enum, and list types; then
runs an agent that calls a decorated tool (MockModel, so no API key needed).

Run:
    PYTHONPATH=src python examples/06_tool_decorator.py
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from agentix import Agent, MockModel, ModelResponse, Role, ToolCall, tool


@tool
def search_products(
    query: str,
    sort: Literal["relevance", "price", "rating"] = "relevance",
    max_results: int = 10,
    tags: Optional[list[str]] = None,
) -> str:
    """Search the product catalog.

    Args:
        query: Free-text search query.
        sort: How to order results.
        max_results: Maximum number of results to return.
        tags: Optional tags to filter by.
    """
    # A real tool would query a database; we fake a result.
    extra = f" tags={tags}" if tags else ""
    return f"Top {max_results} '{query}' products by {sort}{extra}: [Widget, Gadget]"


def show_generated_schema() -> None:
    print("== generated schema for `search_products` ==")
    print(json.dumps(search_products.schema, indent=2))
    print("\nrequired params:", search_products.parameters.get("required"))
    print("still callable directly:", search_products("drills", max_results=2))


def run_agent_with_the_tool() -> None:
    print("\n== agent using the decorated tool ==")
    model = MockModel(
        [
            # The model decides to call the tool...
            ModelResponse(
                tool_calls=[ToolCall("search_products", {"query": "drills", "sort": "price"})]
            ),
            # ...then answers using the result.
            ModelResponse(text="I found Widget and Gadget, sorted by price."),
        ]
    )

    # Just pass the decorated function(s) — the agent builds the executor and
    # the schemas the model sees from them. No manual wiring.
    agent = Agent(
        model=model,
        system_prompt="You help users shop. Use search_products when relevant.",
        tools=[search_products],
    )

    outcome = agent.run_sync("Find me some drills, cheapest first.")
    print("answer:", outcome.answer)

    tool_msg = next(m for m in outcome.transcript if m.role == Role.TOOL)
    print("tool returned:", tool_msg.content)


if __name__ == "__main__":
    show_generated_schema()
    run_agent_with_the_tool()
