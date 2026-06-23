"""Google Gemini model adapter (``google-genai`` SDK).

Translates agentix types to Gemini's ``generate_content`` shape — ``contents``
with ``user``/``model`` roles and typed ``parts`` (``text`` / ``function_call`` /
``function_response``). Install with ``pip install "agentix[gemini]"``.

Everything is built as plain ``dict`` (the SDK accepts dict ``contents`` /
``config`` / function declarations), so no ``google.genai.types`` import is
needed and the translation is unit-testable with a bare fake client.

Note: Gemini function calls carry no call id — tool results are matched back by
**name**. Parallel calls to the *same* tool in one turn can't be disambiguated;
that's a Gemini-protocol limitation, not an agentix one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..content import AudioPart, ContentPart, DocumentPart, ImagePart, TextPart
from ..model import ToolSchema
from ..pricing import cost_usd
from ..types import Message, ModelResponse, Role, ToolCall

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiModel:
    """A :class:`~agentix.model.ModelFn` backed by Gemini ``generate_content``.

    ``extra`` is merged into the request ``config`` (e.g.
    ``temperature``, ``max_output_tokens``, ``response_mime_type`` /
    ``response_schema`` for JSON). For tests, inject ``client=`` — any object
    exposing ``client.aio.models.generate_content(...)``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                from google import genai
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "GeminiModel requires the 'google-genai' package. "
                    'Install it with: pip install "agentix[gemini]"'
                ) from exc
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.extra = extra

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        system, contents = _to_contents(messages)
        config: dict[str, Any] = dict(self.extra)
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [{"function_declarations": _to_declarations(tools)}]

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
        )
        return self._parse(response)

    def _parse(self, response: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = _read(response, "candidates", []) or []
        if candidates:
            content = _read(candidates[0], "content")
            for part in _read(content, "parts", []) or []:
                fc = _read(part, "function_call", None)
                if fc is not None:
                    args = _read(fc, "args", {}) or {}
                    tool_calls.append(
                        ToolCall(name=_read(fc, "name"), args=dict(args))
                    )
                    continue
                txt = _read(part, "text", None)
                if txt:
                    text_parts.append(txt)

        usage = _read(response, "usage_metadata", None)
        input_tokens = int(_read(usage, "prompt_token_count", 0) or 0) if usage else 0
        output_tokens = (
            int(_read(usage, "candidates_token_count", 0) or 0) if usage else 0
        )
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(self.model, input_tokens, output_tokens),
        )


def _to_contents(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            system_parts.append(msg.text)
        elif msg.role is Role.USER:
            contents.append({"role": "user", "parts": _to_parts(msg.content)})
        elif msg.role is Role.ASSISTANT:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for call in msg.meta.get("tool_calls", []):
                parts.append(
                    {"function_call": {"name": call.name, "args": call.args}}
                )
            contents.append({"role": "model", "parts": parts})
        elif msg.role is Role.TOOL:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": msg.name,
                                "response": {"result": msg.content},
                            }
                        }
                    ],
                }
            )
    return "\n\n".join(system_parts), contents


def _to_parts(content: str | list[ContentPart]) -> list[dict[str, Any]]:
    """User content -> Gemini parts. Media uses inline_data (base64) or file_data
    (a URI — works for uploaded/`gs://` files; arbitrary http URLs may not)."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
        elif isinstance(part, ImagePart | DocumentPart | AudioPart):
            if part.url is not None:
                parts.append(
                    {"file_data": {"file_uri": part.url, "mime_type": part.media_type}}
                )
            else:
                parts.append(
                    {"inline_data": {"mime_type": part.media_type, "data": part.data}}
                )
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unsupported content part: {part!r}")
    return parts


def _to_declarations(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        decl: dict[str, Any] = {
            "name": t["name"],
            "description": t.get("description", ""),
        }
        params = t.get("parameters") or t.get("input_schema")
        if params and params.get("properties"):
            decl["parameters"] = params  # Gemini rejects empty-property schemas
        out.append(decl)
    return out


def _read(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
