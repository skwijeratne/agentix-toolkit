"""Model provider adapters."""

from __future__ import annotations

from .anthropic import AnthropicModel
from .mock import MockModel

__all__ = ["AnthropicModel", "MockModel"]
