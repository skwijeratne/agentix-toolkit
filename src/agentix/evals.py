"""Evaluation harness.

Run an agent over a dataset of golden cases, score each, and get a report you
can **assert on in CI** — so a prompt or model change that regresses quality
fails the build instead of shipping.

    cases = [
        Case("What is 2+2?", expected="4"),
        Case("Capital of France?", expected="Paris"),
    ]
    report = await evaluate(cases, agent, scorer=contains())
    report.assert_pass_rate(0.9)   # raises if quality dropped

Scorers are callables ``(outcome, case) -> bool | Score`` (sync or async). Ships
exact-match / contains / regex / predicate / LLM-as-judge; write your own for
anything deterministic. `MockModel` (deterministic) + `Store` (transcript
snapshots) make eval runs reproducible.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .concurrency import bounded_gather
from .model import ModelFn
from .types import AgentOutcome, Message, Role

__all__ = [
    "Case",
    "CaseResult",
    "EvalReport",
    "Score",
    "Scorer",
    "contains",
    "evaluate",
    "exact_match",
    "llm_judge",
    "predicate",
    "regex_match",
]


@dataclass
class Score:
    passed: bool
    score: float = 0.0
    detail: str = ""


#: A scorer judges one outcome. Returns a bool or a Score; sync or async.
Scorer = Callable[
    ["AgentOutcome", "Case"], bool | Score | Awaitable[bool | Score]
]


@dataclass
class Case:
    """One eval case: an input, an optional expected answer, an optional
    per-case scorer (overrides the default), and metadata."""

    input: str
    expected: Any = None
    scorer: Scorer | None = None
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case: Case
    outcome: AgentOutcome | None
    score: Score
    error: str | None = None


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.score.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def format_success_rate(self) -> float:
        """Fraction of runs that completed (not aborted / errored) — a proxy for
        output-format success when an ``output_validator`` is configured."""
        if not self.results:
            return 0.0
        ok = sum(
            1
            for r in self.results
            if r.outcome is not None and r.outcome.status == "completed"
        )
        return ok / self.total

    def summary(self) -> str:
        lines = [
            f"Eval: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.1%}) | format-success {self.format_success_rate:.1%}"
        ]
        for i, r in enumerate(self.results):
            mark = "✓" if r.score.passed else "✗"
            label = r.case.id or f"case-{i}"
            detail = f"  — {r.score.detail}" if (not r.score.passed and r.score.detail) else ""
            lines.append(f"  {mark} {label}{detail}")
        return "\n".join(lines)

    def assert_pass_rate(self, minimum: float) -> None:
        """Raise AssertionError if the pass rate is below ``minimum`` (for CI)."""
        if self.pass_rate < minimum:
            raise AssertionError(
                f"pass rate {self.pass_rate:.1%} < required {minimum:.1%}\n{self.summary()}"
            )


# ── the runner ───────────────────────────────────────────────────────────

AgentFactory = Callable[[Case], Agent]


def _coerce(result: bool | Score) -> Score:
    if isinstance(result, Score):
        return result
    return Score(passed=result, score=1.0 if result else 0.0)


async def evaluate(
    dataset: Sequence[Case],
    agent: Agent | AgentFactory,
    *,
    scorer: Scorer | None = None,
    concurrency: int = 1,
) -> EvalReport:
    """Run ``agent`` over ``dataset`` and score each case.

    ``agent`` may be a single :class:`Agent` (reused — fine for stateless models)
    or a factory ``Callable[[Case], Agent]`` (a fresh agent per case — needed for
    stateful models like a scripted ``MockModel``). ``scorer`` is the default;
    a case's own ``scorer`` overrides it. ``concurrency`` runs cases in parallel.
    """
    if scorer is None and any(c.scorer is None for c in dataset):
        raise ValueError("every case needs a scorer (pass `scorer=` or set Case.scorer)")

    resolve: AgentFactory = agent if not isinstance(agent, Agent) else (lambda _case: agent)

    async def _run_one(case: Case) -> CaseResult:
        the_agent = resolve(case)
        try:
            outcome = await the_agent.run(case.input)
        except Exception as exc:  # noqa: BLE001 - a failed run is a failed case
            return CaseResult(case, None, Score(False, 0.0, f"run error: {exc}"), str(exc))

        case_scorer = case.scorer or scorer
        assert case_scorer is not None  # validated above
        result = case_scorer(outcome, case)
        if inspect.isawaitable(result):
            result = await result
        return CaseResult(case, outcome, _coerce(result))

    results = await bounded_gather([_run_one(c) for c in dataset], limit=concurrency)
    return EvalReport(results)


# ── built-in scorers ─────────────────────────────────────────────────────


def exact_match(*, strip: bool = True, case_sensitive: bool = False) -> Scorer:
    """Pass if the answer equals ``case.expected`` exactly."""

    def _norm(s: str) -> str:
        s = s.strip() if strip else s
        return s if case_sensitive else s.lower()

    def _score(outcome: AgentOutcome, case: Case) -> Score:
        answer = outcome.answer or ""
        passed = _norm(answer) == _norm(str(case.expected))
        return Score(passed, 1.0 if passed else 0.0, f"got {answer!r}")

    return _score


def contains(*, case_sensitive: bool = False) -> Scorer:
    """Pass if ``case.expected`` appears in the answer."""

    def _score(outcome: AgentOutcome, case: Case) -> Score:
        answer = outcome.answer or ""
        needle, hay = str(case.expected), answer
        if not case_sensitive:
            needle, hay = needle.lower(), hay.lower()
        passed = needle in hay
        return Score(passed, 1.0 if passed else 0.0, f"got {answer!r}")

    return _score


def regex_match(pattern: str | None = None) -> Scorer:
    """Pass if the answer matches ``pattern`` (or ``case.expected`` if omitted)."""
    import re

    def _score(outcome: AgentOutcome, case: Case) -> Score:
        pat = pattern if pattern is not None else str(case.expected)
        passed = re.search(pat, outcome.answer or "") is not None
        return Score(passed, 1.0 if passed else 0.0, f"pattern {pat!r}")

    return _score


def predicate(fn: Callable[[AgentOutcome, Case], bool]) -> Scorer:
    """Wrap an arbitrary ``(outcome, case) -> bool`` check."""

    def _score(outcome: AgentOutcome, case: Case) -> Score:
        passed = fn(outcome, case)
        return Score(passed, 1.0 if passed else 0.0)

    return _score


_JUDGE_SYSTEM = (
    "You are a strict evaluator. Decide whether the candidate answer correctly "
    "and faithfully satisfies the task. Reply with PASS or FAIL as the first "
    "word, then a brief reason."
)


def llm_judge(
    model: ModelFn,
    *,
    rubric: str | None = None,
) -> Scorer:
    """LLM-as-judge: a model scores the answer's correctness/faithfulness."""

    async def _score(outcome: AgentOutcome, case: Case) -> Score:
        parts = [f"Task: {case.input}"]
        if case.expected is not None:
            parts.append(f"Expected: {case.expected}")
        if rubric:
            parts.append(f"Rubric: {rubric}")
        parts.append(f"Candidate answer: {outcome.answer}")
        messages = [
            Message(Role.SYSTEM, _JUDGE_SYSTEM, trusted=True),
            Message(Role.USER, "\n".join(parts), trusted=True),
        ]
        response = await model(messages)
        text = response.text.strip()
        first = text.split()[0].lower() if text.split() else ""
        passed = first.startswith("pass")
        return Score(passed, 1.0 if passed else 0.0, text[:200])

    return _score
