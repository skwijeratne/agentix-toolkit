"""16 — Reliability: output validation + retry, and resilient models.

Two production fears, addressed:

  * "malformed output crashes my downstream code" → `output_validator` validates
    the final answer; on failure the agent re-prompts the model with the error
    (up to `max_output_retries`), and exposes the parsed value as `outcome.parsed`.
  * "the provider had a blip / I want to escalate" → `RetryModel` (backoff) and
    `FallbackModel` (try the next model) wrap any model and compose.

Dependency-free (MockModel).

Run:
    PYTHONPATH=src python examples/16_reliability.py
"""

from __future__ import annotations

from agentix import (
    Agent,
    FallbackModel,
    MockModel,
    ModelResponse,
    RetryModel,
    json_output,
)


def demo_validation_retry() -> None:
    print("== output validation + retry ==")
    # First answer is malformed JSON; after the retry prompt the model fixes it.
    model = MockModel(
        [
            ModelResponse(text="here's your data: {oops not json}"),
            ModelResponse(text='{"order_id": 42, "status": "shipped"}'),
        ]
    )
    agent = Agent(
        model=model,
        system_prompt="Return only a JSON object.",
        output_validator=json_output,
        max_output_retries=2,
    )
    outcome = agent.run_sync("give me the order as JSON")
    print(f"  status={outcome.status}, steps={outcome.steps} (1 retry)")
    print(f"  parsed (a real dict, safe to use): {outcome.parsed}")
    print(f"  parsed['status'] = {outcome.parsed['status']}\n")


def demo_fallback() -> None:
    print("== resilient model: retry + fallback ==")

    class OutageModel:
        """Simulates a provider outage — always errors."""
        async def __call__(self, messages, *, tools=()):
            raise RuntimeError("503 Service Unavailable")

    # Try the primary (with 1 retry); if it still fails, fall back to a backup.
    model = FallbackModel(
        [
            RetryModel(OutageModel(), retries=1, backoff=0.0),
            MockModel([ModelResponse(text="answer from the backup model")]),
        ]
    )
    agent = Agent(model=model, system_prompt="sys")
    outcome = agent.run_sync("anything")
    print(f"  primary was down; served by fallback -> {outcome.answer!r}")


if __name__ == "__main__":
    demo_validation_retry()
    demo_fallback()
