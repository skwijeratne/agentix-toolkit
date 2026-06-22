"""Exception hierarchy for agentix.

All library-raised errors derive from :class:`AgentError` so callers can catch
the whole family with one ``except``.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agentix errors."""


class BudgetExceeded(AgentError):
    """Raised/recorded when a run exceeds its token or step budget."""


class GuardError(AgentError):
    """A guard refused a tool call. Carries a human-readable reason."""


class ToolError(AgentError):
    """A tool failed to execute. Surfaced to the model as data, not a crash."""
