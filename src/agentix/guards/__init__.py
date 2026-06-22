"""The guard subsystem — agentix's security checkpoints.

Guards are opt-in: an ``Agent`` with no ``guards`` runs a clean loop. Pass
``guards=secure_defaults()`` (or your own list) to turn on the protections.
"""

from __future__ import annotations

from ..policy import AgentPolicy
from .base import Decision, DecisionType, Guard, GuardContext, GuardPipeline
from .injection import (
    InjectionDetector,
    InjectionGuard,
    UntrustedDataGuard,
    default_injection_detector,
    wrap_as_untrusted_data,
)
from .pii import DEFAULT_REDACTION_PATTERNS, PiiRedactionGuard, PiiUrlGuard
from .tiers import TierGuard
from .trust import RecipientTrustGuard, TrustPredicate

__all__ = [
    "DEFAULT_REDACTION_PATTERNS",
    "Decision",
    "DecisionType",
    "Guard",
    "GuardContext",
    "GuardPipeline",
    "InjectionDetector",
    "InjectionGuard",
    "PiiRedactionGuard",
    "PiiUrlGuard",
    "RecipientTrustGuard",
    "TierGuard",
    "TrustPredicate",
    "UntrustedDataGuard",
    "default_injection_detector",
    "secure_defaults",
    "wrap_as_untrusted_data",
]


def secure_defaults(policy: AgentPolicy | None = None) -> list[Guard]:
    """A conservative, non-destructive default pipeline:

      * :class:`TierGuard` — enforce prohibited / confirm-first tiers.
      * :class:`PiiUrlGuard` — block PII in URLs/query strings.
      * :class:`InjectionGuard` — flag injection-like tool output.
      * :class:`UntrustedDataGuard` — wrap tool output as untrusted data.

    :class:`RecipientTrustGuard` is intentionally **not** included — it needs an
    app-specific trust predicate; add it explicitly when relevant.
    """
    return [TierGuard(), PiiUrlGuard(), InjectionGuard(), UntrustedDataGuard()]
