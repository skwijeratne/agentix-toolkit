"""Guard primitives: the uniform checkpoint the loop runs around every tool call.

A :class:`Guard` exposes two optional hooks:

  * ``before_call`` — inspect a pending :class:`~agentix.types.ToolCall` and
    return a :class:`Decision` (allow / deny / confirm).
  * ``after_output`` — transform a tool's output text before it re-enters the
    model's context (e.g. neutralize injection, mark as untrusted data).

A :class:`GuardPipeline` runs an ordered list of guards. ``before_call`` stops
at the first ``deny``; any ``confirm`` along the way means the loop must get a
human "yes" before executing. This replaces the reference's hard-coded ``if``
ladder with composable, swappable objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..policy import AgentPolicy
from ..types import ToolCall


class DecisionType(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Decision:
    """A guard's verdict on a pending tool call."""

    type: DecisionType
    reason: str = ""

    @classmethod
    def allow(cls) -> "Decision":
        return cls(DecisionType.ALLOW)

    @classmethod
    def deny(cls, reason: str) -> "Decision":
        return cls(DecisionType.DENY, reason)

    @classmethod
    def confirm(cls, reason: str = "") -> "Decision":
        return cls(DecisionType.CONFIRM, reason)

    @property
    def is_allow(self) -> bool:
        return self.type is DecisionType.ALLOW

    @property
    def is_deny(self) -> bool:
        return self.type is DecisionType.DENY

    @property
    def is_confirm(self) -> bool:
        return self.type is DecisionType.CONFIRM


@dataclass
class GuardContext:
    """Read-only context handed to every guard for a given call."""

    policy: AgentPolicy


class Guard:
    """Base guard. Subclass and override the hooks you need; defaults are no-ops
    (allow / pass-through), so a guard only implements what it cares about.

    Three checkpoints, covering both boundaries:
      * ``before_call`` — a pending tool call (ingress to a tool).
      * ``after_output`` — a tool's result re-entering context (egress from a tool).
      * ``on_answer``    — the model's final answer leaving for the user (egress
        to the user). Use it for redaction / DLP on what the user sees.
    """

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        return Decision.allow()

    async def after_output(self, call: ToolCall, content: str, ctx: GuardContext) -> str:
        return content

    async def on_answer(self, answer: str, ctx: GuardContext) -> str:
        return answer


class GuardPipeline:
    """Runs an ordered list of guards as a single checkpoint."""

    def __init__(self, guards: Sequence[Guard] = ()) -> None:
        self.guards: list[Guard] = list(guards)

    def __len__(self) -> int:
        return len(self.guards)

    async def before_call(self, call: ToolCall, ctx: GuardContext) -> Decision:
        confirm_reasons: list[str] = []
        for guard in self.guards:
            decision = await guard.before_call(call, ctx)
            if decision.is_deny:
                return decision  # first deny wins, fail closed
            if decision.is_confirm and decision.reason:
                confirm_reasons.append(decision.reason)
            elif decision.is_confirm:
                confirm_reasons.append(f"run '{call.name}'")
        if confirm_reasons:
            return Decision.confirm("; ".join(confirm_reasons))
        return Decision.allow()

    async def after_output(self, call: ToolCall, content: str, ctx: GuardContext) -> str:
        for guard in self.guards:
            content = await guard.after_output(call, content, ctx)
        return content

    async def on_answer(self, answer: str, ctx: GuardContext) -> str:
        for guard in self.guards:
            answer = await guard.on_answer(answer, ctx)
        return answer
