"""Ollama model adapter — local models via the native ``ollama`` client.

Run open models (Llama, Qwen, Mistral, …) on your own machine. Install with
``pip install "agentix[ollama]"`` and have an Ollama server running
(``ollama serve``; pull a tool-capable model, e.g. ``ollama pull llama3.1``).

Ollama's chat API is OpenAI-ish but not identical: tool-call ``arguments`` are
JSON **objects** (not strings) and usage fields are named differently, so this
adapter does its own translation rather than reusing the OpenAI-compat helper.
Local inference is free, so ``cost_usd`` is always ``0.0``.

Tip: Ollama also serves an OpenAI-compatible endpoint at ``/v1`` — if you prefer
that surface (or the ``openai`` SDK you already use), point
:class:`~agentix.providers.openai.OpenAIModel` at
``base_url="http://localhost:11434/v1"`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..model import ToolSchema
from ..types import Message, ModelResponse, Role, ToolCall
from . import _openai_compat as oc

DEFAULT_MODEL = "llama3.1"


class OllamaModel:
    """A :class:`~agentix.model.ModelFn` backed by a local Ollama server.

    ``extra`` is forwarded to ``chat`` (e.g. ``options={"temperature": 0}``,
    ``keep_alive``, ``format`` for JSON mode). For tests, inject ``client=`` —
    any object exposing an async ``chat(**kwargs)``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                from ollama import AsyncClient
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OllamaModel requires the 'ollama' package. "
                    'Install it with: pip install "agentix[ollama]"'
                ) from exc
            client = AsyncClient(host=host)
        self._client = client
        self.model = model
        self.extra = extra

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_messages(messages),
            **self.extra,
        }
        if tools:
            kwargs["tools"] = oc.to_tools(tools)  # OpenAI-style function wrappers
        response = await self._client.chat(**kwargs)
        return _parse(response)


def _to_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """agentix messages -> Ollama chat messages (object tool-call arguments)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            out.append({"role": "system", "content": msg.content})
        elif msg.role is Role.USER:
            out.append({"role": "user", "content": msg.content})
        elif msg.role is Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            calls: list[ToolCall] = msg.meta.get("tool_calls", [])
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.args}} for c in calls
                ]
            out.append(entry)
        elif msg.role is Role.TOOL:
            out.append({"role": "tool", "content": msg.content, "name": msg.name})
    return out


def _read(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse(response: Any) -> ModelResponse:
    message = _read(response, "message")
    text = _read(message, "content", "") or ""
    tool_calls: list[ToolCall] = []
    for tc in _read(message, "tool_calls", None) or []:
        fn = _read(tc, "function")
        args = _read(fn, "arguments", {})
        tool_calls.append(
            ToolCall(name=_read(fn, "name"), args=dict(args) if args else {})
        )
    input_tokens = int(_read(response, "prompt_eval_count", 0) or 0)
    output_tokens = int(_read(response, "eval_count", 0) or 0)
    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        tokens_used=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.0,  # local inference
    )
