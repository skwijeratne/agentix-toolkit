"""Self-consistency.

``SelfConsistencyModel`` wraps a model and, on each turn, samples it N times and
returns the **majority vote** — a simple, effective way to damp non-determinism
on hard reasoning steps. It's a ``ModelFn``, so it drops into ``Agent(model=...)``.

Cost: it makes N model calls per turn. The returned response aggregates the
token/USD spend across all N samples, so the agent's budgets and
``outcome.cost_usd`` reflect the real cost. Pair with a ``Limiter`` to bound the
extra concurrency.

Responses are grouped by a ``key`` (default: normalized text for a final answer,
or the tool-call signature otherwise); the largest group wins, ties go to the
first sampled.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence

from .model import ModelFn, ToolSchema
from .types import Message, ModelResponse


def _default_key(response: ModelResponse) -> str:
    if response.is_final:
        return "text:" + " ".join(response.text.split()).lower()
    signature = [
        (c.name, json.dumps(c.args, sort_keys=True, default=str))
        for c in response.tool_calls
    ]
    return "tools:" + json.dumps(signature)


class SelfConsistencyModel:
    """Sample the wrapped model N times per turn and return the majority vote."""

    def __init__(
        self,
        model: ModelFn,
        *,
        samples: int = 5,
        key: Callable[[ModelResponse], str] | None = None,
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.model = model
        self.samples = samples
        self.key = key or _default_key

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        responses = await asyncio.gather(
            *(self.model(messages, tools=tools) for _ in range(self.samples))
        )
        groups: dict[str, list[ModelResponse]] = {}
        for response in responses:
            groups.setdefault(self.key(response), []).append(response)

        winner = max(groups.values(), key=len)
        representative = winner[0]
        # Aggregate spend across all samples — you paid for N calls.
        return ModelResponse(
            text=representative.text,
            tool_calls=representative.tool_calls,
            tokens_used=sum(r.tokens_used for r in responses),
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            cost_usd=sum(r.cost_usd for r in responses),
        )
