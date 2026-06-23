"""Dynamic permission guards.

``CallbackGuard`` is agentix's ``can_use_tool``: a per-call callback that decides
allow / deny / ask based on whatever you like — the tool, its args, the user's
role, external state, a rate limiter. ``ToolAllowlistGuard`` is the declarative
"this agent may only use these tools" case.

Both are ordinary guards, so they compose with the rest of the pipeline
(tiers, PII, injection). The pipeline is AND-ed: the most restrictive guard wins
(first deny stops; any confirm asks).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable

from ..types import ToolCall
from .base import Decision, Guard, GuardContext

#: A permission callback returns a Decision, or a bool (True=allow, False=deny).
#: It may be sync or async.
PermissionResult = Decision | bool
PermissionCheck = Callable[
    [ToolCall, GuardContext], PermissionResult | Awaitable[PermissionResult]
]


def _as_decision(result: PermissionResult) -> Decision:
    if isinstance(result, Decision):
        return result
    if result is True:
        return Decision.allow()
    if result is False:
        return Decision.deny("denied by the permission callback")
    raise TypeError(
        f"permission callback must return a Decision or bool, got {type(result).__name__}"
    )


class CallbackGuard(Guard):
    """Decide each tool call with a user-supplied callback (``can_use_tool``).

    The callback receives the :class:`ToolCall` and :class:`GuardContext` and
    returns a :class:`Decision` (``Decision.allow()`` / ``.deny(reason)`` /
    ``.confirm(reason)``) or a plain ``bool``. Sync or async::

        async def can_use(call, ctx):
            if call.name == "refund" and call.args["amount"] > 1000:
                return Decision.deny("refunds over $1000 need a manager")
            return Decision.allow()

        Agent(..., guards=[CallbackGuard(can_use)], confirm_fn=...)
    """

    def __init__(self, check: PermissionCheck) -> None:
        self._check = check

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        result = self._check(call, ctx)
        resolved: PermissionResult
        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = result
        return _as_decision(resolved)


class ToolAllowlistGuard(Guard):
    """Allow only the named tools; deny any other tool call.

    Useful to scope a run to a subset of registered tools (or to cleanly reject
    a tool the model hallucinated)."""

    def __init__(self, allowed: Iterable[str]) -> None:
        self._allowed = set(allowed)

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        if call.name in self._allowed:
            return Decision.allow()
        return Decision.deny(f"'{call.name}' is not in this agent's allowed tool set")
