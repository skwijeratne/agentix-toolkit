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
from typing import Any, Literal

from ..model import ToolSchema
from ..pricing import cost_usd
from ..streaming import ModelStreamEvent, ResponseComplete, TextDelta
from ..types import Message, ModelResponse, Role, ToolCall

#: Default model. Anthropic's most capable Opus-tier model.
DEFAULT_MODEL = "claude-opus-4-8"

#: Effort level (cost-vs-quality knob), per `output_config.effort`.
Effort = Literal["low", "medium", "high", "xhigh", "max"]

#: Thinking config: True/"adaptive" -> adaptive, "summarized" -> adaptive w/ a
#: visible summary, False/"disabled" -> off, or a raw dict for full control.
Thinking = bool | Literal["adaptive", "summarized", "disabled"] | dict[str, Any]

#: Beta header required by task budgets.
_TASK_BUDGET_BETA = "task-budgets-2026-03-13"


def _coerce_thinking(value: Thinking | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if value is True or value == "adaptive":
        return {"type": "adaptive"}
    if value == "summarized":
        return {"type": "adaptive", "display": "summarized"}
    if value is False or value == "disabled":
        return {"type": "disabled"}
    if isinstance(value, dict):
        return value
    raise ValueError(f"unsupported thinking value: {value!r}")


class AnthropicModel:
    """A :class:`~agentix.model.ModelFn` backed by the Anthropic Messages API.

    Example::

        from agentix import Agent
        from agentix.providers.anthropic import AnthropicModel

        agent = Agent(model=AnthropicModel(), system_prompt="...")

    Typed knobs (the cost-vs-quality / reasoning controls):

    * ``thinking`` — ``True`` / ``"adaptive"`` (let the model decide how much to
      think), ``"summarized"`` (adaptive + a visible summary), ``"disabled"``, or
      a raw dict. Note: on Opus 4.7+/Fable, extended thinking is *adaptive only*;
      ``"disabled"`` is rejected on Fable — omit it there.
    * ``effort`` — ``"low" | "medium" | "high" | "xhigh" | "max"``; sets
      ``output_config.effort`` (default is ``"high"``). Lower = fewer tokens.
    * ``task_budget`` — an int token budget the model self-moderates against for
      the whole agentic loop (≥ 20000); adds the required beta header. Distinct
      from ``max_tokens`` (a hard per-response cap).

    ``extra`` keyword arguments are forwarded to ``messages.create``. For
    provider-enforced JSON, pass
    ``output_config={"format": {"type": "json_schema", "schema": ...}}`` (merged
    with ``effort``/``task_budget``); pair with the agent's ``output_validator``
    for client-side validation + retry. Tool schemas with a ``strict`` key are
    forwarded for strict tool validation.

    Refusal fallback: a safety *refusal* surfaces as a normal final answer
    (``stop_reason == "refusal"``), **not** an exception — so
    :class:`~agentix.resilience.FallbackModel` (which falls back on errors) does
    **not** catch it. To fall back on a refusal, use the Claude API's server-side
    ``fallbacks`` parameter (pass it via ``extra``) or detect the refusal text in
    your app. ``FallbackModel``/``RetryModel`` remain the right tools for
    *outages and transient errors*.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        api_key: str | None = None,
        client: Any = None,
        thinking: Thinking | None = None,
        effort: Effort | None = None,
        task_budget: int | None = None,
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
        self.thinking = _coerce_thinking(thinking)
        self.effort = effort
        self.task_budget = task_budget
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
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking

        # effort and task_budget both live under output_config (merge with any
        # output_config the caller passed via `extra`, e.g. a structured format).
        output_config: dict[str, Any] = dict(kwargs.get("output_config") or {})
        if self.effort is not None:
            output_config["effort"] = self.effort
        if self.task_budget is not None:
            output_config["task_budget"] = {"type": "tokens", "total": self.task_budget}
        if output_config:
            kwargs["output_config"] = output_config

        # task budgets are beta-gated — add the header (merge with any existing).
        if self.task_budget is not None:
            headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})
            existing = headers.get("anthropic-beta")
            headers["anthropic-beta"] = (
                f"{existing},{_TASK_BUDGET_BETA}" if existing else _TASK_BUDGET_BETA
            )
            kwargs["extra_headers"] = headers
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
            entry: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": schema,
            }
            if "strict" in t:  # forward strict tool-schema enforcement if requested
                entry["strict"] = t["strict"]
            out.append(entry)
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
