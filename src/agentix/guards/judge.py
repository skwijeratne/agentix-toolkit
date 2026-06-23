"""LLM-as-judge guard.

``JudgeGuard`` runs an LLM over the final answer (via the ``on_answer`` egress
hook) and **replaces it if it fails a rubric** — an output gate for safety,
tone/on-brand, or format ("jailbreaks and off-brand content").

It judges the answer text **in isolation** (the ``on_answer`` hook only sees the
answer, not the task), so use it for answer-only checks. For *task-faithfulness*
judging — where the judge needs the original request — use the eval
``llm_judge`` scorer, which has the full case. Adds one model call per final
answer; mind the cost/latency.
"""

from __future__ import annotations

from ..model import ModelFn
from ..types import Message, Role
from .base import Guard, GuardContext

_JUDGE_SYSTEM = (
    "You are a strict content reviewer. Given a rubric and a response, decide "
    "whether the response satisfies the rubric. Reply with PASS or FAIL as the "
    "very first word, then a brief reason."
)


class JudgeGuard(Guard):
    def __init__(
        self,
        model: ModelFn,
        *,
        rubric: str,
        replacement: str = "[This response was withheld because it did not meet policy.]",
    ) -> None:
        self.model = model
        self.rubric = rubric
        self.replacement = replacement

    async def on_answer(self, answer: str, ctx: GuardContext) -> str:
        messages = [
            Message(Role.SYSTEM, _JUDGE_SYSTEM, trusted=True),
            Message(
                Role.USER,
                f"Rubric: {self.rubric}\n\nResponse to review:\n{answer}",
                trusted=True,
            ),
        ]
        response = await self.model(messages)
        text = response.text.strip()
        first = text.split()[0].lower() if text.split() else ""
        if not first.startswith("pass"):
            return self.replacement
        return answer
