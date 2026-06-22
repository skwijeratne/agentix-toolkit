"""Human-in-the-loop confirmation.

A ``ConfirmFn`` receives a human-readable description of a pending action and
returns True only on an explicit "yes". It may be sync or async — async lets a
web/server app await a real user decision without blocking the event loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Union

#: Returns True to approve. Sync or async; the loop awaits awaitable results.
ConfirmFn = Callable[[str], Union[bool, "Awaitable[bool]"]]


def always_approve(description: str) -> bool:
    """Approve everything. For tests/automation only — never in production."""
    return True


def always_deny(description: str) -> bool:
    """Decline everything. The safe default when no human is available."""
    return False


def console_confirm(description: str) -> bool:
    """Prompt on the terminal. Blocks for real stdin input — CLI use only."""
    answer = input(f"{description} [y/N] ").strip().lower()
    return answer in {"y", "yes"}
