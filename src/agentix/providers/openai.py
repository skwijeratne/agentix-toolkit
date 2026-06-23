"""OpenAI model adapter (Chat Completions API).

Translates between agentix's framework-agnostic types and OpenAI's chat API via
the official ``openai`` package — install with ``pip install "agentix[openai]"``.
The import is deferred to construction so ``import agentix`` works without it.

Because the wire format is the OpenAI *Chat Completions* shape, this adapter also
drives any OpenAI-compatible endpoint: pass ``base_url=`` (and an ``api_key`` your
gateway accepts) to point it at vLLM, Together, Groq, a local server, etc. For
Ollama specifically see :class:`~agentix.providers.ollama.OllamaModel`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..model import ToolSchema
from ..pricing import cost_usd
from ..streaming import ModelStreamEvent, ResponseComplete, TextDelta
from ..types import Message, ModelResponse
from . import _openai_compat as oc

#: A capable, current default. Override per call site as needed.
DEFAULT_MODEL = "gpt-4o"


class OpenAIModel:
    """A :class:`~agentix.model.ModelFn` backed by OpenAI Chat Completions.

    Example::

        from agentix import Agent
        from agentix.providers.openai import OpenAIModel

        agent = Agent(model=OpenAIModel(model="gpt-4o"), system_prompt="...")

    ``extra`` keyword arguments are forwarded to ``chat.completions.create`` —
    e.g. ``temperature``, ``response_format`` for JSON mode, or ``reasoning_effort``
    for reasoning models. Tool schemas carrying ``strict`` are forwarded for
    OpenAI strict structured tool calls. Pricing for unknown model ids is ``0.0``
    until registered via :func:`agentix.register_price`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        **extra: Any,
    ) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIModel requires the 'openai' package. "
                    'Install it with: pip install "agentix[openai]"'
                ) from exc
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self.model = model
        self.extra = extra

    def _build_kwargs(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oc.to_messages(messages),
            **self.extra,
        }
        if tools:
            kwargs["tools"] = oc.to_tools(tools)
        return kwargs

    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelResponse:
        response = await self._client.chat.completions.create(
            **self._build_kwargs(messages, tools)
        )
        return oc.parse_response(response, self.model)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
    ) -> AsyncIterator[ModelStreamEvent]:
        # Stream text deltas; ask for usage on the final chunk, then re-issue a
        # single non-streamed call only if tool calls were requested (assembling
        # tool_calls from deltas is brittle and rarely needed for the loop).
        kwargs = self._build_kwargs(messages, tools)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        saw_tool_call = False
        usage: Any = None
        async with _aclosing(
            await self._client.chat.completions.create(**kwargs)
        ) as chunks:
            async for chunk in chunks:
                usage = getattr(chunk, "usage", None) or usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                if getattr(delta, "tool_calls", None):
                    saw_tool_call = True
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    yield TextDelta(piece)

        if saw_tool_call:
            # Re-run without streaming to get well-formed tool_calls.
            yield ResponseComplete(await self(messages, tools=tools))
            return

        text = "".join(text_parts)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        yield ResponseComplete(
            ModelResponse(
                text=text,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd(self.model, input_tokens, output_tokens),
            )
        )


class _aclosing:
    """Minimal ``contextlib.aclosing`` (3.10-compatible) for async iterators."""

    def __init__(self, thing: Any) -> None:
        self._thing = thing

    async def __aenter__(self) -> Any:
        return self._thing

    async def __aexit__(self, *exc: Any) -> None:
        aclose = getattr(self._thing, "aclose", None)
        if aclose is not None:
            await aclose()
