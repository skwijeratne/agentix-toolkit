"""Record/replay cassettes for deterministic real-model tests.

Hitting a live model in tests is slow, costly, and flaky. :class:`CassetteModel`
wraps any model: the **first** run records each model call's response to a JSON
file; later runs **replay** from the file with no network. Same idea as VCR —
record once, replay forever (delete the file to re-record).

    model = CassetteModel("tests/cassettes/weather.json", model=AnthropicModel())
    agent = Agent(model=model, system_prompt="...", tools=[...])
    outcome = await agent.run("...")
    model.save()   # writes the cassette in record mode (no-op when replaying)

With ``mode="auto"`` (the default) it records when the file is missing and
replays when it exists. Replay is **sequential**: responses are returned in the
order they were recorded, matching a deterministic loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from .model import ModelFn, ToolSchema
from .serde import tool_call_from_dict, tool_call_to_dict, transcript_to_dicts
from .types import Message, ModelResponse

Mode = Literal["auto", "record", "replay"]


def _response_to_dict(r: ModelResponse) -> dict[str, Any]:
    return {
        "text": r.text,
        "tool_calls": [tool_call_to_dict(c) for c in r.tool_calls],
        "tokens_used": r.tokens_used,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd,
    }


def _response_from_dict(d: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        text=d.get("text", ""),
        tool_calls=[tool_call_from_dict(c) for c in d.get("tool_calls", [])],
        tokens_used=int(d.get("tokens_used", 0)),
        input_tokens=int(d.get("input_tokens", 0)),
        output_tokens=int(d.get("output_tokens", 0)),
        cost_usd=float(d.get("cost_usd", 0.0)),
    )


class CassetteModel:
    """A :class:`~agentix.model.ModelFn` that records/replays model responses.

    ``mode``: ``"auto"`` (record if the file is missing, else replay), ``"record"``
    (wrap ``model`` and capture), or ``"replay"`` (read from the file; no ``model``
    needed). Call :meth:`save` after a recording run to write the file.
    """

    def __init__(
        self,
        path: str,
        model: ModelFn | None = None,
        *,
        mode: Mode = "auto",
    ) -> None:
        self.path = Path(path)
        self.model = model
        if mode == "auto":
            mode = "replay" if self.path.exists() else "record"
        if mode == "record" and model is None:
            raise ValueError("record mode needs a `model` to wrap")
        self.mode: Literal["record", "replay"] = mode
        self._recorded: list[dict[str, Any]] = []
        self._queue: list[ModelResponse] = []
        if mode == "replay":
            data = json.loads(self.path.read_text())
            self._queue = [_response_from_dict(i["response"]) for i in data["interactions"]]

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        if self.mode == "replay":
            if not self._queue:
                raise RuntimeError(
                    "cassette exhausted: the run made more model calls than were "
                    f"recorded in {self.path}. Delete it to re-record."
                )
            return self._queue.pop(0)

        assert self.model is not None  # guaranteed by __init__ in record mode
        response = await self.model(messages, tools=tools)
        self._recorded.append(
            {
                "request": {
                    "messages": transcript_to_dicts(list(messages)),
                    "tools": list(tools),
                },
                "response": _response_to_dict(response),
            }
        )
        return response

    def save(self) -> None:
        """Write the recorded interactions (no-op in replay mode)."""
        if self.mode != "record":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": 1, "interactions": self._recorded}, indent=2)
        )
