"""Recipient-trust guard.

Defends against "send data to an endpoint that came from untrusted content": if
a tool call carries a recipient/endpoint argument, the call is denied unless an
injected predicate confirms the recipient was genuinely user-supplied.

**Fail-closed by default.** With no predicate, nothing is a trusted recipient —
this matches the documented intent of the reference (whose default mistakenly
trusted everything). Supply ``is_trusted`` to whitelist legitimate destinations.
"""

from __future__ import annotations

from collections.abc import Callable

from ..types import ToolCall
from .base import Decision, Guard, GuardContext

#: Arg names that indicate the call sends data somewhere.
_RECIPIENT_KEYS = {
    "to",
    "recipient",
    "recipients",
    "endpoint",
    "url",
    "destination",
    "email",
    "address",
    "webhook",
}

TrustPredicate = Callable[[ToolCall], bool]


class RecipientTrustGuard(Guard):
    def __init__(self, is_trusted: TrustPredicate | None = None) -> None:
        self._is_trusted = is_trusted

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        if not self._has_recipient(call):
            return Decision.allow()
        trusted = self._is_trusted(call) if self._is_trusted is not None else False
        if not trusted:
            return Decision.deny(
                "recipient/endpoint was not verified as user-supplied; not "
                "sending data to destinations that may come from untrusted content"
            )
        return Decision.allow()

    @staticmethod
    def _has_recipient(call: ToolCall) -> bool:
        return any(key.lower() in _RECIPIENT_KEYS for key in call.args)
