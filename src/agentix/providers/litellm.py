"""LiteLLM bridge — one adapter, 100+ providers.

`LiteLLM <https://docs.litellm.ai/>`_ exposes a single OpenAI-shaped
``acompletion`` that fans out to OpenAI, Azure, Anthropic, Bedrock, Vertex,
Mistral, Groq, Ollama and many more behind provider-prefixed model ids
(``"gpt-4o"``, ``"anthropic/claude-opus-4-8"``, ``"gemini/gemini-2.0-flash"``,
``"ollama/llama3"``, …). Install with ``pip install "agentix[litellm]"``.

This is the pragmatic "works with whatever the user already has" option; the
dedicated adapters (``OpenAIModel``, ``GeminiModel``, ``BedrockModel``,
``OllamaModel``) exist for when you want first-party control and typed knobs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..model import ToolSchema
from ..types import Message, ModelResponse
from . import _openai_compat as oc


class LiteLLMModel:
    """A :class:`~agentix.model.ModelFn` backed by ``litellm.acompletion``.

    ``model`` uses LiteLLM's provider-prefixed ids. ``extra`` is forwarded to
    ``acompletion`` (e.g. ``api_base``, ``api_key``, ``temperature``,
    ``num_retries``). For tests, inject ``client=`` — any object exposing an
    async ``acompletion(**kwargs)`` (the ``litellm`` module itself by default).
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                import litellm
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "LiteLLMModel requires the 'litellm' package. "
                    'Install it with: pip install "agentix[litellm]"'
                ) from exc
            client = litellm
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
            "messages": oc.to_messages(messages),
            **self.extra,
        }
        if tools:
            kwargs["tools"] = oc.to_tools(tools)
        response = await self._client.acompletion(**kwargs)
        return oc.parse_response(response, self.model)
