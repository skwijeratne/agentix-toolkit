# Tools

A **tool** is something your agent can *do* beyond talking — look up the weather,
search a database, send an email, run a calculation. You write a normal Python
function; agentix shows it to the model and runs it when the model asks.

## The `@tool` decorator

Put `@tool` on a function and you're done. agentix reads the function's name, its
arguments (and their types), and its docstring so the model knows what the tool is
for and how to call it.

```python
from agentix import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Paris".
    """
    return f"{city}: 21°C, sunny"
```

Pass your tools to the agent and it handles the rest — deciding when to call them,
running them, and feeding the results back into the conversation:

```python
agent = Agent(model=m, system_prompt="Help with the weather.", tools=[get_weather])
```

The types you annotate (like `city: str`) become the rules the model follows when
calling the tool, so it sends the right kind of data. Lists, optional arguments,
and fixed choices (`Literal["a", "b"]`) all work.

→ Runnable example:
[`examples/06_tool_decorator.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/06_tool_decorator.py)

## Tools from a tool server (MCP)

**MCP** (Model Context Protocol) is a shared standard for tool servers — ready-made
collections of tools (for files, GitHub, databases, and more) that any agent can
connect to. agentix can connect to an MCP server and use its tools just like your
own:

```python
from agentix import MCPServer

server = MCPServer(...)        # point at a running MCP server
agent = Agent(model=m, system_prompt="...", tools=await server.tools())
```

Install the extra with `agentix-toolkit[mcp]`.

→ Runnable example:
[`examples/11_mcp.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/11_mcp.py)

## One agent as another agent's tool (subagents)

Sometimes a job has a sub-job best handled by a *specialist* — a research agent, a
math agent. You can wrap a whole agent as a tool and hand it to a "lead" agent.
When the lead calls it, the specialist runs its own loop and returns an answer.

```python
from agentix import subagent_tool

research = subagent_tool(researcher_agent, name="research",
                         description="Delegate research questions.")
lead = Agent(model=m, system_prompt="You coordinate specialists.", tools=[research])
```

The specialist's spending (tokens and cost) automatically **adds into** the lead
agent's totals, so your final cost number includes everything.

→ Runnable example:
[`examples/13_subagents.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/13_subagents.py)
