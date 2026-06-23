"""Output validation.

An ``output_validator`` is a callable that takes the model's final answer and
either **returns a validated/parsed value** or **raises** on failure. When set on
an ``Agent``, a failure re-prompts the model with the error (bounded by
``max_output_retries``); on success the parsed value is exposed as
``AgentOutcome.parsed``.

The "return parsed / raise on failure" contract is deliberately Pythonic so it
works with anything: ``json.loads`` (malformed JSON raises), a Pydantic model
(``model_validate_json`` raises ``ValidationError``), or your own deterministic
checker ("does this SQL run?", "do the tests pass?"). Validators may be sync or
async.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

#: Returns the parsed value on success; raises on failure. Sync or async.
OutputValidator = Callable[[str], Any]


def json_output(answer: str) -> Any:
    """Validate that the answer is valid JSON; returns the parsed object."""
    return json.loads(answer)


def pydantic_output(model_cls: Any) -> OutputValidator:
    """Validator factory: validate the answer against a Pydantic model.

    Returns the validated model instance; raises ``pydantic.ValidationError`` on
    a mismatch. agentix doesn't import pydantic — you pass your model class::

        agent = Agent(..., output_validator=pydantic_output(MyModel), max_output_retries=2)
        outcome.parsed  # a validated MyModel instance
    """

    def _validate(answer: str) -> Any:
        return model_cls.model_validate_json(answer)

    return _validate


def regex_output(pattern: str) -> OutputValidator:
    """Validator factory: require the answer to match ``pattern``; returns the answer."""
    compiled = re.compile(pattern)

    def _validate(answer: str) -> str:
        if compiled.search(answer) is None:
            raise ValueError(f"answer did not match the required pattern: {pattern}")
        return answer

    return _validate
