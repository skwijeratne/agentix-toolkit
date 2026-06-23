"""23 — Sandboxed execution of untrusted / model-generated code.

`LocalToolExecutor` runs tools in-process — great for trusted tools, but it can't
contain code you don't trust and can't honour `network_allowlist`. `SubprocessExecutor`
runs each tool as a separate OS process with real limits: no network (Linux
netns, fail-closed), CPU/memory/file-size/process rlimits, an isolated temp
working directory, a scrubbed environment, and a hard timeout.

Run:
    python examples/23_sandbox.py
"""

from __future__ import annotations

import asyncio
import sys

from agentix import Command, SubprocessExecutor
from agentix.types import ToolCall

# A "run this Python" tool: argv runs the interpreter reading code from stdin.
run_python = {"run_python": Command(argv=[sys.executable, "-"], stdin="code")}


async def main() -> None:
    # network_allowlist is what the Agent loop passes from AgentPolicy. Empty =>
    # egress denied (enforced via a network namespace; the call fails closed if
    # the host can't provide one). A non-empty list => network allowed.
    allow_net = ["pypi.org"]

    executor = SubprocessExecutor(run_python)

    # 1) Run some code; capture stdout.
    res = await executor(
        ToolCall("run_python", {"code": "print(6 * 7)"}),
        network_allowlist=allow_net,
    )
    print("1) output:", res.content.strip(), "| ok:", res.ok)

    # 2) Secrets in this process do NOT leak into the child by default.
    import os

    os.environ["MY_API_KEY"] = "sk-secret"
    peek = "import os; print(os.environ.get('MY_API_KEY', 'HIDDEN'))"
    res = await executor(
        ToolCall("run_python", {"code": peek}),
        network_allowlist=allow_net,
    )
    print("2) child sees the key?:", res.content.strip())

    # 3) A runaway is killed by the timeout (not the full 30s).
    res = await executor(
        ToolCall("run_python", {"code": "import time; time.sleep(30)"}),
        network_allowlist=allow_net,
        timeout_s=0.5,
    )
    print("3) runaway:", res.content.strip(), "| ok:", res.ok)

    # 4) The headline guarantee: deny network. With the default policy
    # (require_network_isolation=True), if the host can't isolate the network the
    # tool REFUSES rather than running untrusted code with egress.
    fetch = "import urllib.request; urllib.request.urlopen('http://example.com')"
    res = await executor(
        ToolCall("run_python", {"code": fetch}),
        network_allowlist=[],  # deny all egress
    )
    print("4) network-denied result ok?:", res.ok)
    print("   ->", res.content.strip()[:120])

    # Wiring into an Agent: pass the executor + tool_schemas, and the loop hands
    # it AgentPolicy.network_allowlist / tool_timeout_s automatically:
    #
    #   agent = Agent(
    #       model=my_model,
    #       system_prompt="Use run_python to compute things.",
    #       tool_executor=SubprocessExecutor(run_python),
    #       tool_schemas=[{"name": "run_python", "description": "...",
    #                      "parameters": {"type": "object",
    #                                     "properties": {"code": {"type": "string"}},
    #                                     "required": ["code"]}}],
    #       policy=AgentPolicy(network_allowlist=[]),  # no egress for tools
    #   )


if __name__ == "__main__":
    asyncio.run(main())
