"""Anthropic model adapter.

Translates between agentix's framework-agnostic types and the Anthropic
Messages API. Requires the ``anthropic`` package — install with
``pip install "agentix[anthropic]"``. The import is deferred to construction
time so ``import agentix`` works without the dependency present.

Notes on the translation:
  * agentix's ``system`` prompt becomes the top-level ``system=`` parameter.
  * An assistant turn with tool calls becomes an assistant message whose
    ``content`` is a list of ``text`` + ``tool_use`` blocks. agentix stores the
    structured :class:`ToolCall` objects on the message's ``meta["tool_calls"]``
    so they can be replayed here.
  * A tool result becomes a ``user`` message with a ``tool_result`` block keyed
    by the originating ``tool_use_id``.
  * agentix tool schemas use ``parameters`` (a JSON Schema); Anthropic wants
    ``input_schema`` — translated here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..model import ToolSchema
from ..pricing import cost_usd
from ..streaming import ModelStreamEvent, ResponseComplete, TextDelta
from ..types import Message, ModelResponse, Role, ToolCall

#: Default model. Anthropic's most capable Opus-tier model.
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicModel:
    """A :class:`~agentix.model.ModelFn` backed by the Anthropic Messages API.

    Example::

        from agentix import Agent
        from agentix.providers.anthropic import AnthropicModel

        agent = Agent(model=AnthropicModel(), system_prompt="...")

    ``extra`` keyword arguments are forwarded to ``messages.create`` — use them
    for ``thinking``, ``output_config``, etc.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        api_key: str | None = None,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "AnthropicModel requires the 'anthropic' package. "
                    'Install it with: pip install "agentix[anthropic]"'
                ) from exc
            client = AsyncAnthropic(api_key=api_key)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.extra = extra

    def _build_kwargs(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> dict[str, Any]:
        system, conversation = self._translate_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": conversation,
            **self.extra,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        return kwargs

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        response = await self._client.messages.create(
            **self._build_kwargs(messages, tools)
        )
        return self._translate_response(response)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> AsyncIterator[ModelStreamEvent]:
        # Uses the SDK's streaming helper: forward text deltas, then assemble the
        # full message (carrying any tool_use blocks) for ResponseComplete.
        async with self._client.messages.stream(
            **self._build_kwargs(messages, tools)
        ) as stream:
            async for text in stream.text_stream:
                yield TextDelta(text)
            final = await stream.get_final_message()
        yield ResponseComplete(self._translate_response(final))

    # ── agentix -> Anthropic ──────────────────────────────────────────────

    @staticmethod
    def _translate_messages(
        messages: Sequence[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role is Role.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role is Role.USER:
                conversation.append({"role": "user", "content": msg.content})
            elif msg.role is Role.ASSISTANT:
                conversation.append(
                    {"role": "assistant", "content": _assistant_content(msg)}
                )
            elif msg.role is Role.TOOL:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.meta.get("call_id"),
                                "content": msg.content,
                                "is_error": not msg.meta.get("ok", True),
                            }
                        ],
                    }
                )

        return "\n\n".join(system_parts), conversation

    @staticmethod
    def _translate_tools(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools:
            schema = t.get("parameters") or t.get("input_schema") or {
                "type": "object",
                "properties": {},
            }
            out.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": schema,
                }
            )
        return out

    # ── Anthropic -> agentix ──────────────────────────────────────────────

    def _translate_response(self, response: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(name=block.name, args=dict(block.input), id=block.id)
                )
            # thinking / other block types are ignored for the loop's purposes

        usage = getattr(response, "usage", None)
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        text = "".join(text_parts)
        if getattr(response, "stop_reason", None) == "refusal":
            text = text or "[The request was declined by a safety classifier.]"

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(self.model, input_tokens, output_tokens),
        )


def _assistant_content(msg: Message) -> Any:
    """Build Anthropic assistant content from an agentix assistant message."""
    calls: list[ToolCall] = msg.meta.get("tool_calls", [])
    if not calls:
        return msg.content
    blocks: list[dict[str, Any]] = []
    if msg.content:
        blocks.append({"type": "text", "text": msg.content})
    for call in calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
        )
    return blocks
