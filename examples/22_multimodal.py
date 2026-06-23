"""22 — Multimodal input (vision, documents, audio).

A message's content can be a plain string *or* a list of parts: text interleaved
with images, documents (PDF), and audio. You build the parts the same way for
every provider; each adapter translates them to that vendor's wire format (and
raises a clear error for media a provider can't accept — e.g. audio on Anthropic,
URL images on Bedrock).

This demo is dependency-free: a MockModel stands in for a vision model so you can
see the plumbing without a key. Swap `model=` for any vision-capable adapter
(AnthropicModel, OpenAIModel, GeminiModel, …) and it works unchanged.

Run:
    python examples/22_multimodal.py
"""

from __future__ import annotations

import asyncio

from agentix import (
    Agent,
    DocumentPart,
    ImagePart,
    MockModel,
    ModelResponse,
    TextPart,
)

# A 1x1 transparent PNG — stands in for "an image you loaded".
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001"
)


async def main() -> None:
    # Build a multimodal user turn. Parts can come from bytes, a file, a URL,
    # or raw base64 — pick whichever you have:
    user_turn = [
        TextPart("What's in this image, and how does it relate to the attached doc?"),
        ImagePart.from_bytes(_PNG, "image/png"),          # inline bytes
        # ImagePart.from_path("diagram.png"),             # a local file (mime inferred)
        # ImagePart.from_url("https://example.com/x.jpg"),# a remote URL
        DocumentPart.from_url("https://example.com/spec.pdf", "application/pdf"),
    ]

    agent = Agent(
        model=MockModel([ModelResponse(text="A 1x1 PNG; the doc is a linked PDF spec.")]),
        system_prompt="You are a concise visual assistant.",
    )

    # The loop accepts the parts list anywhere a string request would go.
    outcome = await agent.run(user_turn)
    print("answer:", outcome.answer)

    # `.text` gives a string view of any message (media parts contribute nothing).
    from agentix import Message, Role

    print("text view:", Message(Role.USER, user_turn).text)


# To use a real vision model instead of the mock:
#
#   from agentix.providers.anthropic import AnthropicModel
#   agent = Agent(model=AnthropicModel(), system_prompt="...")
#   await agent.run([TextPart("Describe this"), ImagePart.from_path("cat.png")])


if __name__ == "__main__":
    asyncio.run(main())
