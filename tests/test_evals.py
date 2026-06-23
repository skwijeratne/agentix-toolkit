from agentix import (
    Agent,
    Case,
    EvalReport,
    MockModel,
    ModelResponse,
    Score,
    contains,
    evaluate,
    exact_match,
    llm_judge,
    predicate,
    regex_match,
)


# Build a per-case agent whose model answers from a lookup (deterministic).
def make_factory(answers: dict[str, str]):
    def factory(case: Case) -> Agent:
        return Agent(
            model=MockModel([ModelResponse(text=answers.get(case.input, ""))]),
            system_prompt="sys",
        )

    return factory


# ── scorers in isolation ─────────────────────────────────────────────────


def _outcome(answer: str):
    from agentix import AgentOutcome

    return AgentOutcome(status="completed", answer=answer)


def test_exact_match_scorer() -> None:
    s = exact_match()
    assert s(_outcome("Paris"), Case("q", expected="paris")).passed  # case-insensitive
    assert not s(_outcome("London"), Case("q", expected="Paris")).passed


def test_contains_scorer() -> None:
    s = contains()
    assert s(_outcome("The answer is 4."), Case("q", expected="4")).passed
    assert not s(_outcome("nope"), Case("q", expected="4")).passed


def test_regex_scorer() -> None:
    s = regex_match(r"\b\d{4}\b")
    assert s(_outcome("year 2026 ok"), Case("q")).passed
    assert not s(_outcome("no digits"), Case("q")).passed


def test_predicate_scorer() -> None:
    s = predicate(lambda outcome, case: outcome.answer == "yes")
    assert s(_outcome("yes"), Case("q")).passed
    assert not s(_outcome("no"), Case("q")).passed


# ── evaluate + report ────────────────────────────────────────────────────


async def test_evaluate_pass_rate_and_report() -> None:
    answers = {"2+2?": "4", "capital of France?": "Paris", "color of sky?": "green"}
    cases = [
        Case("2+2?", expected="4", id="math"),
        Case("capital of France?", expected="Paris", id="geo"),
        Case("color of sky?", expected="blue", id="sky"),  # model answers wrong
    ]
    report = await evaluate(cases, make_factory(answers), scorer=contains())

    assert isinstance(report, EvalReport)
    assert report.total == 3
    assert report.passed == 2
    assert abs(report.pass_rate - 2 / 3) < 1e-9
    assert report.format_success_rate == 1.0  # all runs completed
    assert "2/3 passed" in report.summary()


async def test_per_case_scorer_overrides_default() -> None:
    answers = {"give a year": "2026"}
    cases = [Case("give a year", id="y", scorer=regex_match(r"^\d{4}$"))]
    report = await evaluate(cases, make_factory(answers), scorer=exact_match())
    assert report.passed == 1  # used the per-case regex, not exact_match


async def test_assert_pass_rate_passes_and_fails() -> None:
    answers = {"q1": "a", "q2": "b"}
    cases = [Case("q1", expected="a"), Case("q2", expected="WRONG")]
    report = await evaluate(cases, make_factory(answers), scorer=exact_match())
    report.assert_pass_rate(0.5)  # 50% — ok
    try:
        report.assert_pass_rate(0.9)  # too high
        raise AssertionError("expected the assertion to fail")
    except AssertionError as e:
        assert "pass rate" in str(e)


async def test_run_error_is_a_failed_case() -> None:
    class BoomModel:
        async def __call__(self, messages, *, tools=()):
            raise RuntimeError("boom")

    def factory(case: Case) -> Agent:
        return Agent(model=BoomModel(), system_prompt="sys")

    report = await evaluate([Case("x", expected="y")], factory, scorer=exact_match())
    assert report.passed == 0
    assert report.results[0].outcome is None
    assert "boom" in (report.results[0].error or "")
    assert report.format_success_rate == 0.0


async def test_missing_scorer_raises() -> None:
    try:
        await evaluate([Case("x")], make_factory({}))  # no default, no per-case
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "scorer" in str(e)


async def test_concurrency_runs_all() -> None:
    answers = {f"q{i}": "ok" for i in range(10)}
    cases = [Case(f"q{i}", expected="ok") for i in range(10)]
    report = await evaluate(cases, make_factory(answers), scorer=exact_match(), concurrency=4)
    assert report.total == 10 and report.passed == 10


async def test_single_agent_instance_reused() -> None:
    # A stateless model can be shared (no per-case factory needed).
    class EchoOK:
        async def __call__(self, messages, *, tools=()):
            return ModelResponse(text="ok")

    agent = Agent(model=EchoOK(), system_prompt="sys")
    report = await evaluate([Case("a", expected="ok"), Case("b", expected="ok")],
                            agent, scorer=exact_match())
    assert report.passed == 2


# ── LLM-as-judge ─────────────────────────────────────────────────────────


async def test_llm_judge_scorer() -> None:
    # A deterministic "judge" model: PASS for the right answer, FAIL otherwise.
    def judge_model_factory(verdict: str):
        class JudgeModel:
            async def __call__(self, messages, *, tools=()):
                return ModelResponse(text=f"{verdict} — looks correct")

        return JudgeModel()

    passing = llm_judge(judge_model_factory("PASS"))
    failing = llm_judge(judge_model_factory("FAIL"))
    assert (await passing(_outcome("Paris"), Case("capital?"))).passed
    assert not (await failing(_outcome("London"), Case("capital?"))).passed


async def test_score_coercion_from_bool() -> None:
    # A scorer may return a plain bool; the harness coerces it to a Score.
    report = await evaluate(
        [Case("q", expected="ok")],
        make_factory({"q": "ok"}),
        scorer=predicate(lambda outcome, case: outcome.answer == case.expected),
    )
    assert report.passed == 1
    assert isinstance(report.results[0].score, Score)
