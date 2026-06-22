"""Permission-tier guard.

Reads the per-tool tiers from the :class:`~agentix.policy.AgentPolicy`:
``prohibited`` tools are denied outright; ``confirm_first`` (and anything when
``default_deny`` is set) requires explicit human approval.
"""

from __future__ import annotations

from ..policy import Tier
from ..types import ToolCall
from .base import Decision, Guard, GuardContext


class TierGuard(Guard):
    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        tier = ctx.policy.tier_for(call.name)
        if tier is Tier.PROHIBITED:
            return Decision.deny(
                f"'{call.name}' is not permitted for the agent to perform; "
                "the user must do it themselves"
            )
        if tier is Tier.CONFIRM_FIRST:
            return Decision.confirm(f"run '{call.name}'")
        return Decision.allow()
