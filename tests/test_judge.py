from agentix import Agent, AgentPolicy, JudgeGuard, MockModel, ModelResponse, Role
from agentix.guards import GuardContext


def _ctx() -> GuardContext:
    return GuardContext(AgentPolicy())


class VerdictModel:
    """A stand-in judge model that always returns the given verdict."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    async def __call__(self, messages, *, tools=()):
        return ModelResponse(text=f"{self.verdict} — reviewed")


async def test_judge_passes_keeps_answer() -> None:
    guard = JudgeGuard(VerdictModel("PASS"), rubric="must be polite")
    out = await guard.on_answer("Hello, happy to help!", _ctx())
    assert out == "Hello, happy to help!"


async def test_judge_fails_replaces_answer() -> None:
    guard = JudgeGuard(VerdictModel("FAIL"), rubric="must be on-brand")
    out = await guard.on_answer("something off-brand", _ctx())
    assert out.startswith("[This response was withheld")


async def test_judge_custom_replacement() -> None:
    guard = JudgeGuard(VerdictModel("FAIL"), rubric="r", replacement="REDACTED")
    out = await guard.on_answer("bad", _ctx())
    assert out == "REDACTED"


async def test_judge_guard_in_loop_replaces_final_answer() -> None:
    # The agent produces an answer; the judge fails it; the user sees the
    # replacement.
    model = MockModel([ModelResponse(text="off-brand rant")])
    agent = Agent(
        model=model,
        system_prompt="sys",
        guards=[JudgeGuard(VerdictModel("FAIL"), rubric="must be professional")],
    )
    outcome = await agent.run("say something")
    assert "withheld" in (outcome.answer or "")
    # The transcript's final assistant message is the replacement (what the user saw).
    assert "withheld" in outcome.transcript[-1].content


async def test_judge_guard_in_loop_passes_answer_through() -> None:
    model = MockModel([ModelResponse(text="a professional, helpful reply")])
    agent = Agent(
        model=model,
        system_prompt="sys",
        guards=[JudgeGuard(VerdictModel("PASS"), rubric="must be professional")],
    )
    outcome = await agent.run("say something")
    assert outcome.answer == "a professional, helpful reply"
    assert outcome.transcript[-1].role == Role.ASSISTANT
