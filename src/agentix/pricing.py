"""USD cost estimation from token usage.

Provider adapters that know their model's pricing set ``ModelResponse.cost_usd``;
the loop sums it into ``AgentOutcome.cost_usd``. This module holds a small price
table for current Claude models and a helper to compute cost. Prices are USD per
1M tokens as ``(input, output)`` and may drift — override via ``register_price``.
"""

from __future__ import annotations

# USD per 1,000,000 tokens: (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def register_price(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    """Add or override pricing for a model id."""
    PRICES[model] = (input_per_mtok, output_per_mtok)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a call. Returns 0.0 for unknown models."""
    prices = PRICES.get(model)
    if prices is None:
        return 0.0
    input_price, output_price = prices
    return input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price
