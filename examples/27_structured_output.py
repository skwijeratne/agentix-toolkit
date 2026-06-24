"""27 — First-class structured output.

`Agent(response_model=…)` is one knob that wires the whole structured-output
path: it validates the final answer (so `outcome.parsed` is typed), re-prompts on
failure (`max_output_retries`), injects the JSON schema as an instruction so any
model conforms, and turns on **native** provider enforcement when the adapter
supports it (Anthropic `output_config.format`, OpenAI `response_format`, …).

This demo is dependency-free (a raw JSON-Schema dict + MockModel). With Pydantic,
pass the model class instead and `outcome.parsed` is a validated instance.

Run:
    python examples/27_structured_output.py
"""

from __future__ import annotations

import asyncio

from agentix import Agent, MockModel, ModelResponse, Role

PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


async def main() -> None:
    agent = Agent(
        model=MockModel([ModelResponse(text='{"name": "Ada", "age": 36}')]),
        system_prompt="Extract the person as JSON.",
        response_model=PERSON_SCHEMA,   # a dict schema; or a Pydantic model class
    )
    outcome = await agent.run("Ada Lovelace, 36 years old.")

    print("status:", outcome.status)
    print("parsed:", outcome.parsed)          # {'name': 'Ada', 'age': 36}, validated
    print("typed access:", outcome.parsed["name"])

    # The schema was injected as a system instruction (so ANY model conforms):
    system = next(m.content for m in outcome.transcript if m.role is Role.SYSTEM)
    print("\nsystem instruction included schema:", "JSON Schema" in system)

    # With Pydantic + a real provider, the *same* one knob also enforces natively:
    #
    #   from pydantic import BaseModel
    #   from agentix.providers.openai import OpenAIModel
    #
    #   class Person(BaseModel):
    #       name: str
    #       age: int
    #
    #   agent = Agent(model=OpenAIModel(), system_prompt="...", response_model=Person)
    #   outcome = await agent.run("...")
    #   person: Person = outcome.parsed     # validated instance; provider-enforced


if __name__ == "__main__":
    asyncio.run(main())
