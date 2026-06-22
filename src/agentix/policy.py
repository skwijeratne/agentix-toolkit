"""Declarative agent policy.

None of this generalizes across applications — it depends entirely on what your
agent can touch — so it lives here as plain config rather than baked into the
loop. The loop reads it; it never hard-codes any of these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(Enum):
    """Permission tier for a tool action (enforced by the guard subsystem)."""

    PROHIBITED = "prohibited"        # never auto-perform; hand back to a human
    CONFIRM_FIRST = "confirm_first"  # require explicit per-action human "yes"
    AUTO_OK = "auto_ok"              # side-effect-free; proceed


@dataclass
class AgentPolicy:
    """All policy in one place: resource budgets, tool tiers, and the inputs the
    security subsystem reads. Safe to construct with no arguments for an
    unrestricted toolkit run; tighten as needed."""

    # Tool permission tiers, keyed by tool name (used by the guard subsystem).
    prohibited: set[str] = field(default_factory=set)
    confirm_first: set[str] = field(default_factory=set)
    # Anything not listed is AUTO_OK unless default_deny is True.
    default_deny: bool = False

    # Resource guards (enforced by the loop).
    max_steps: int = 25
    max_tokens_budget: int = 200_000
    tool_timeout_s: float = 30.0

    # Sandbox egress restriction, passed to the tool executor.
    network_allowlist: list[str] = field(default_factory=list)

    # Privacy: regexes that must NOT appear in outbound URLs / query strings.
    pii_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b\d{3}-\d{2}-\d{4}\b",             # SSN-like
            r"\b(?:\d[ -]*?){13,16}\b",           # card-like
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",  # email
        ]
    )

    def tier_for(self, tool_name: str) -> Tier:
        if tool_name in self.prohibited:
            return Tier.PROHIBITED
        if tool_name in self.confirm_first:
            return Tier.CONFIRM_FIRST
        if self.default_deny:
            return Tier.CONFIRM_FIRST
        return Tier.AUTO_OK
