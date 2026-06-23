from agentix import Agent, FallbackModel, MockModel, ModelResponse, RetryModel


class FlakyModel:
    """Fails the first `fail_times` calls, then returns `text`."""

    def __init__(self, fail_times: int, text: str = "ok", exc: type[Exception] = RuntimeError):
        self.fail_times = fail_times
        self.text = text
        self.exc = exc
        self.calls = 0

    async def __call__(self, messages, *, tools=()):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"transient failure {self.calls}")
        return ModelResponse(text=self.text)


# ── RetryModel ───────────────────────────────────────────────────────────


async def test_retry_recovers_after_transient_failures() -> None:
    flaky = FlakyModel(fail_times=2, text="recovered")
    model = RetryModel(flaky, retries=3, backoff=0.0)
    resp = await model([])
    assert resp.text == "recovered"
    assert flaky.calls == 3  # 2 failures + 1 success


async def test_retry_gives_up_after_exhausting() -> None:
    flaky = FlakyModel(fail_times=5)
    model = RetryModel(flaky, retries=2, backoff=0.0)
    try:
        await model([])
        raise AssertionError("expected the error to propagate")
    except RuntimeError:
        pass
    assert flaky.calls == 3  # initial + 2 retries


async def test_retry_only_catches_listed_exceptions() -> None:
    flaky = FlakyModel(fail_times=1, exc=ValueError)
    model = RetryModel(flaky, retries=3, backoff=0.0, retry_on=(RuntimeError,))
    try:
        await model([])
        raise AssertionError("ValueError should not be retried")
    except ValueError:
        pass
    assert flaky.calls == 1  # not retried


def test_retry_rejects_bad_config() -> None:
    try:
        RetryModel(MockModel([]), retries=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ── FallbackModel ────────────────────────────────────────────────────────


async def test_fallback_uses_first_working_model() -> None:
    primary = FlakyModel(fail_times=99)  # always fails
    secondary = MockModel([ModelResponse(text="from secondary")])
    model = FallbackModel([primary, secondary])
    resp = await model([])
    assert resp.text == "from secondary"


async def test_fallback_prefers_primary_when_it_works() -> None:
    primary = MockModel([ModelResponse(text="from primary")])
    secondary = MockModel([ModelResponse(text="from secondary")])
    resp = await FallbackModel([primary, secondary])([])
    assert resp.text == "from primary"


async def test_fallback_raises_if_all_fail() -> None:
    model = FallbackModel([FlakyModel(99), FlakyModel(99)])
    try:
        await model([])
        raise AssertionError("expected the last error to propagate")
    except RuntimeError:
        pass


def test_fallback_needs_a_model() -> None:
    try:
        FallbackModel([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ── composition + use in an Agent ────────────────────────────────────────


async def test_compose_retry_and_fallback_in_agent() -> None:
    # primary always fails even with retries; fall back to a working model.
    primary = RetryModel(FlakyModel(99), retries=1, backoff=0.0)
    secondary = MockModel([ModelResponse(text="answer")])
    agent = Agent(model=FallbackModel([primary, secondary]), system_prompt="sys")
    outcome = await agent.run("go")
    assert outcome.answer == "answer"
