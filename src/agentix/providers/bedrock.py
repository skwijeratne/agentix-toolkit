"""AWS Bedrock model adapter (Converse API).

The Bedrock **Converse** API gives one uniform tool-use shape across every model
Bedrock hosts (Claude, Llama, Mistral, Nova, …), so a single adapter covers them
all — pick the model with ``model=<modelId>``. Install with
``pip install "agentix[bedrock]"``; auth/region come from the standard AWS chain
(env vars, ``~/.aws/config``, instance role).

``boto3`` is synchronous, so each call is dispatched to a worker thread via
:func:`asyncio.to_thread` to keep the loop non-blocking. Pricing is account- and
region-specific — register it with :func:`agentix.register_price` if you want
``cost_usd`` populated (otherwise ``0.0``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..model import ToolSchema
from ..pricing import cost_usd
from ..types import Message, ModelResponse, Role, ToolCall


class BedrockModel:
    """A :class:`~agentix.model.ModelFn` backed by ``bedrock-runtime.converse``.

    ``model`` is a Bedrock model id or inference-profile id (e.g.
    ``"anthropic.claude-3-5-sonnet-20241022-v2:0"`` or
    ``"us.anthropic.claude-opus-4-..."``). ``extra`` is forwarded to ``converse``
    (e.g. ``additionalModelRequestFields`` for thinking, ``guardrailConfig``).
    ``max_tokens`` sets ``inferenceConfig.maxTokens``. For tests, inject
    ``client=`` — any object with a sync ``converse(**kwargs)``.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 4096,
        region_name: str | None = None,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "BedrockModel requires the 'boto3' package. "
                    'Install it with: pip install "agentix[bedrock]"'
                ) from exc
            client = boto3.client("bedrock-runtime", region_name=region_name)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.extra = extra

    def _build_kwargs(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> dict[str, Any]:
        system, conversation = _to_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": conversation,
            "inferenceConfig": {"maxTokens": self.max_tokens},
            **self.extra,
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": _to_tool_specs(tools)}
        return kwargs

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        kwargs = self._build_kwargs(messages, tools)
        response = await asyncio.to_thread(self._client.converse, **kwargs)
        return self._parse(response)

    def _parse(self, response: Any) -> ModelResponse:
        message = response["output"]["message"]
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        name=tu["name"],
                        args=dict(tu.get("input", {})),
                        id=tu.get("toolUseId"),
                    )
                )

        usage = response.get("usage", {}) or {}
        input_tokens = int(usage.get("inputTokens", 0) or 0)
        output_tokens = int(usage.get("outputTokens", 0) or 0)
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(self.model, input_tokens, output_tokens),
        )


def _to_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            system_parts.append(msg.content)
        elif msg.role is Role.USER:
            conversation.append({"role": "user", "content": [{"text": msg.content}]})
        elif msg.role is Role.ASSISTANT:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"text": msg.content})
            for call in msg.meta.get("tool_calls", []):
                content.append(
                    {
                        "toolUse": {
                            "toolUseId": call.id or call.name,
                            "name": call.name,
                            "input": call.args,
                        }
                    }
                )
            conversation.append({"role": "assistant", "content": content})
        elif msg.role is Role.TOOL:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": msg.meta.get("call_id") or msg.name,
                                "content": [{"text": msg.content}],
                                "status": "success" if msg.meta.get("ok", True) else "error",
                            }
                        }
                    ],
                }
            )
    return "\n\n".join(system_parts), conversation


def _to_tool_specs(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        schema = t.get("parameters") or t.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        out.append(
            {
                "toolSpec": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "inputSchema": {"json": schema},
                }
            }
        )
    return out
