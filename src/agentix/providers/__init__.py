"""Model provider adapters.

Each adapter defers its heavy SDK import to construction, so importing the class
is always safe even when the provider's package isn't installed — you only need
the matching extra (``agentix[openai]``, ``agentix[gemini]``, …) to *use* one.
"""

from __future__ import annotations

from .anthropic import AnthropicModel
from .bedrock import BedrockModel
from .gemini import GeminiModel
from .litellm import LiteLLMModel
from .mock import MockModel
from .ollama import OllamaModel
from .openai import OpenAIModel

__all__ = [
    "AnthropicModel",
    "BedrockModel",
    "GeminiModel",
    "LiteLLMModel",
    "MockModel",
    "OllamaModel",
    "OpenAIModel",
]
