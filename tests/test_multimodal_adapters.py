"""Each adapter translates multimodal content into its provider's block shape,
and raises a clear error for media a provider can't accept.

Adapters are called directly (not via the loop) with a single multimodal user
message; we inspect the captured request kwargs.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from agentix import AudioPart, DocumentPart, ImagePart, Message, Role, TextPart

B64 = base64.b64encode(b"hi").decode()  # "aGk="
IMG = ImagePart.from_base64(B64, "image/png")
TXT = TextPart("what is this?")


def _user(*parts: Any) -> list[Message]:
    return [Message(Role.USER, list(parts))]


# ── Anthropic ─────────────────────────────────────────────────────────────


class _AnthropicFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


async def test_anthropic_image_blocks_and_audio_rejected() -> None:
    from agentix.providers.anthropic import AnthropicModel

    fake = _AnthropicFake()
    await AnthropicModel(client=fake)(_user(TXT, IMG))
    content = fake.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": B64},
    }

    aud = AudioPart.from_bytes(b"a", "audio/wav")
    with pytest.raises(ValueError, match="audio"):
        await AnthropicModel(client=_AnthropicFake())(_user(aud))


# ── OpenAI / LiteLLM (shared OpenAI-format translation) ────────────────────


class _OpenAIFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


async def test_openai_image_url_and_audio_blocks() -> None:
    from agentix.providers.openai import OpenAIModel

    fake = _OpenAIFake()
    await OpenAIModel(client=fake)(_user(TXT, IMG, AudioPart.from_base64(B64, "audio/wav")))
    content = fake.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}}
    assert content[2] == {"type": "input_audio", "input_audio": {"data": B64, "format": "wav"}}


async def test_litellm_reuses_openai_image_translation() -> None:
    from agentix.providers.litellm import LiteLLMModel

    class _Lite:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def acompletion(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    fake = _Lite()
    await LiteLLMModel(client=fake, model="gpt-4o")(_user(TXT, IMG))
    content = fake.calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"


# ── Gemini ────────────────────────────────────────────────────────────────


class _GeminiFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._gen))

    async def _gen(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="ok", function_call=None)]
                    )
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )


async def test_gemini_inline_data_for_image_and_document() -> None:
    from agentix.providers.gemini import GeminiModel

    fake = _GeminiFake()
    doc = DocumentPart.from_base64(B64, "application/pdf")
    await GeminiModel(client=fake)(_user(TXT, IMG, doc))
    parts = fake.calls[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "what is this?"}
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": B64}}
    assert parts[2] == {"inline_data": {"mime_type": "application/pdf", "data": B64}}


# ── Bedrock ───────────────────────────────────────────────────────────────


class _BedrockFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:  # sync, like boto3
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }


async def test_bedrock_image_bytes_and_url_rejected() -> None:
    from agentix.providers.bedrock import BedrockModel

    fake = _BedrockFake()
    await BedrockModel(client=fake, model="m")(_user(TXT, IMG))
    content = fake.calls[0]["messages"][0]["content"]
    assert content[0] == {"text": "what is this?"}
    assert content[1] == {"image": {"format": "png", "source": {"bytes": b"hi"}}}

    with pytest.raises(ValueError, match="inline bytes"):
        await BedrockModel(client=_BedrockFake(), model="m")(
            _user(ImagePart.from_url("https://x/y.png"))
        )


# ── Ollama ────────────────────────────────────────────────────────────────


class _OllamaFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "message": {"content": "ok", "tool_calls": None},
            "prompt_eval_count": 1,
            "eval_count": 1,
        }


async def test_ollama_images_at_message_level_and_docs_rejected() -> None:
    from agentix.providers.ollama import OllamaModel

    fake = _OllamaFake()
    await OllamaModel(client=fake)(_user(TXT, IMG))
    user_msg = fake.calls[0]["messages"][0]
    assert user_msg["content"] == "what is this?"
    assert user_msg["images"] == [B64]

    doc = DocumentPart.from_base64(B64, "application/pdf")
    with pytest.raises(ValueError, match="image attachments"):
        await OllamaModel(client=_OllamaFake())(_user(doc))
