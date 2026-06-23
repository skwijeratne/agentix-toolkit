"""Tool execution — the boundary where tool calls actually run.

The executor is deliberately separate from the model and from tool *definitions*:
the registry says *what* tools exist, the executor says *how* (and under what
limits) they run. Keeping execution here is what lets you drop in a sandboxed
executor later without touching the loop. The loop passes the policy's
``network_allowlist`` and ``timeout_s`` so those limits can't be influenced by
the model.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from .types import ToolCall, ToolResult

ToolFn = Callable[..., object | Awaitable[object]]


@runtime_checkable
class ToolExecutor(Protocol):
    """Executes a single tool call under policy-enforced limits."""

    async def __call__(
        self,
        call: ToolCall,
        *,
        network_allowlist: Sequence[str] = (),
        timeout_s: float = 30.0,
    ) -> ToolResult: ...


class LocalToolExecutor:
    """Runs tools in-process by dispatching to a mapping of name -> callable.

    Callables may be sync or async and are invoked with the tool-call args as
    keyword arguments. Each call is bounded by ``timeout_s``. This executor does
    NOT sandbox the network or filesystem and cannot honour ``network_allowlist``
    (an in-process call can open any socket) — it's the default for trusted,
    in-process tools. For untrusted tools or model-generated code, use
    :class:`~agentix.sandbox.SubprocessExecutor`, which runs each call in an
    isolated subprocess and enforces the network/resource limits.

    Concurrency: **synchronous** tool functions are run in a worker thread
    (via :func:`asyncio.to_thread`) so a blocking tool can't stall the event
    loop and starve other concurrently-running agents — and so ``timeout_s``
    can actually return control (a blocking sync call cannot be timed out while
    it holds the loop). Note: a timed-out sync tool's *thread* keeps running in
    the background until it returns — Python can't forcibly kill a thread — and
    sync tools draw from the default thread-pool, so size that pool for your
    concurrency (``loop.set_default_executor(...)``). Async tools run inline.
    """

    def __init__(self, tools: Mapping[str, ToolFn]) -> None:
        self._tools: dict[str, ToolFn] = dict(tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    async def __call__(
        self,
        call: ToolCall,
        *,
        network_allowlist: Sequence[str] = (),
        timeout_s: float = 30.0,
    ) -> ToolResult:
        fn = self._tools.get(call.name)
        if fn is None:
            return ToolResult(call.name, f"unknown tool: {call.name}", call.id, ok=False)

        try:
            if inspect.iscoroutinefunction(fn):
                # Async tool: runs inline; wait_for cancels it cleanly on timeout.
                value = await asyncio.wait_for(fn(**call.args), timeout=timeout_s)
            else:
                # Sync tool: run in a worker thread so it never blocks the loop.
                # Threads can't be cancelled, so on timeout we orphan the thread
                # (it finishes in the background) and return control to the loop.
                task: asyncio.Future[object] = asyncio.ensure_future(
                    asyncio.to_thread(fn, **call.args)
                )
                done, _pending = await asyncio.wait({task}, timeout=timeout_s)
                if not done:
                    # Orphan the thread; retrieve its eventual result/exception so
                    # it isn't reported as "never retrieved" (skip if cancelled).
                    task.add_done_callback(lambda t: t.cancelled() or t.exception())
                    raise asyncio.TimeoutError
                value = task.result()
                if inspect.isawaitable(value):  # sync fn that returned a coroutine
                    value = await value
        except asyncio.TimeoutError:
            return ToolResult(
                call.name, f"tool timed out after {timeout_s}s", call.id, ok=False
            )
        except Exception as exc:  # noqa: BLE001 — surface as data, don't crash the loop
            return ToolResult(call.name, f"ERROR running tool: {exc}", call.id, ok=False)

        return ToolResult(call.name, str(value), call.id, ok=True)
