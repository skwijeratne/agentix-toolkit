"""PII guards.

``PiiUrlGuard`` (``before_call``) refuses a tool call whose URL/query-like
arguments contain PII — personal data must never end up in query strings, where
it leaks into logs and referrers.

``PiiRedactionGuard`` (``on_answer``) masks PII in the model's final answer to
the user — a last-line DLP filter on what leaves the loop. It uses its own,
tighter pattern set (the URL patterns are tuned for *detection*, which is too
loose for redacting free text without false positives).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ..types import ToolCall
from .base import Decision, Guard, GuardContext

# Arg names that strongly imply a URL/endpoint even without a scheme.
_URLISH_KEYS = {"url", "endpoint", "href", "link", "query"}

#: Tighter patterns for redacting free text (require boundaries / a real TLD).
DEFAULT_REDACTION_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                   # SSN
    r"\b(?:\d[ -]?){13,19}\b",                                  # card number
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",     # email
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",  # US phone
]


class PiiUrlGuard(Guard):
    def __init__(self, patterns: Optional[Sequence[str]] = None) -> None:
        # If None, the policy's pii_patterns are used at call time.
        self._patterns = list(patterns) if patterns is not None else None
        self._compiled_for: Optional[tuple[str, ...]] = None
        self._compiled: list[re.Pattern[str]] = []

    def _compile(self, patterns: Sequence[str]) -> list[re.Pattern[str]]:
        key = tuple(patterns)
        if key != self._compiled_for:
            self._compiled = [re.compile(p) for p in patterns]
            self._compiled_for = key
        return self._compiled

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        patterns = self._patterns if self._patterns is not None else ctx.policy.pii_patterns
        compiled = self._compile(patterns)
        for key, val in call.args.items():
            if not isinstance(val, str):
                continue
            looks_like_url = "://" in val or "?" in val or key.lower() in _URLISH_KEYS
            if looks_like_url and any(c.search(val) for c in compiled):
                return Decision.deny(
                    f"PII-like value in URL/query arg '{key}'; never place "
                    "personal data in query strings"
                )
        return Decision.allow()


class PiiRedactionGuard(Guard):
    """Masks PII in the final answer before the user sees it.

    Opt-in (not in :func:`secure_defaults`) because redacting user-facing text
    is an application/compliance choice and can mask data the user legitimately
    asked for. Configure ``patterns`` and ``mask`` for your domain.
    """

    def __init__(
        self,
        patterns: Optional[Sequence[str]] = None,
        *,
        mask: str = "[REDACTED]",
    ) -> None:
        self._compiled = [
            re.compile(p) for p in (patterns or DEFAULT_REDACTION_PATTERNS)
        ]
        self.mask = mask

    async def on_answer(self, answer: str, ctx: GuardContext) -> str:
        for pattern in self._compiled:
            answer = pattern.sub(self.mask, answer)
        return answer
