from agentix import Agent, ModelResponse, SelfConsistencyModel, ToolCall


class ScriptedModel:
    """Returns the given responses in order (one per call)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self, messages, *, tools=()):
        r = self._responses[self.calls]
        self.calls += 1
        return r


async def test_majority_vote_on_text() -> None:
    model = ScriptedModel(
        [ModelResponse(text="42"), ModelResponse(text="42"), ModelResponse(text="41")]
    )
    sc = SelfConsistencyModel(model, samples=3)
    resp = await sc([])
    assert resp.text == "42"  # 2 vs 1
    assert model.calls == 3


async def test_normalizes_whitespace_and_case_when_voting() -> None:
    model = ScriptedModel(
        [
            ModelResponse(text="The Answer"),
            ModelResponse(text="the answer"),
            ModelResponse(text="something else"),
        ]
    )
    resp = await SelfConsistencyModel(model, samples=3)([])
    assert resp.text in ("The Answer", "the answer")  # the two grouped, won


async def test_votes_on_tool_calls() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall("search", {"q": "x"})]),
            ModelResponse(tool_calls=[ToolCall("search", {"q": "x"})]),
            ModelResponse(tool_calls=[ToolCall("search", {"q": "y"})]),
        ]
    )
    resp = await SelfConsistencyModel(model, samples=3)([])
    assert resp.tool_calls[0].args == {"q": "x"}


async def test_aggregates_cost_across_samples() -> None:
    model = ScriptedModel(
        [
            ModelResponse(text="a", tokens_used=10, cost_usd=0.01),
            ModelResponse(text="a", tokens_used=10, cost_usd=0.01),
            ModelResponse(text="b", tokens_used=10, cost_usd=0.01),
        ]
    )
    resp = await SelfConsistencyModel(model, samples=3)([])
    assert resp.tokens_used == 30  # paid for all 3
    assert abs(resp.cost_usd - 0.03) < 1e-9


def test_rejects_bad_samples() -> None:
    try:
        SelfConsistencyModel(ScriptedModel([]), samples=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


async def test_works_as_agent_model() -> None:
    # 3 samples of the final answer; majority wins, then the loop completes.
    model = ScriptedModel([ModelResponse(text="done")] * 3)
    agent = Agent(model=SelfConsistencyModel(model, samples=3), system_prompt="sys")
    outcome = await agent.run("go")
    assert outcome.answer == "done"
    assert outcome.tokens_used == 0  # these mock responses carry no tokens
