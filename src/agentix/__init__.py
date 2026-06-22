"""agentix — a generic, batteries-included agent toolkit.

Configure the agent loop, tools, guards, and observability instead of
rewriting them for every project. The core is provider-agnostic and async-first.
"""

from __future__ import annotations

from .agent import Agent
from .concurrency import Limiter, bounded_gather
from .confirm import ConfirmFn, always_approve, always_deny, console_confirm
from .errors import AgentError, BudgetExceeded, GuardError, ToolError
from .events import AgentEvents
from .executors import LocalToolExecutor, ToolExecutor
from .guards import (
    Decision,
    Guard,
    GuardContext,
    GuardPipeline,
    InjectionGuard,
    PiiRedactionGuard,
    PiiUrlGuard,
    RecipientTrustGuard,
    TierGuard,
    UntrustedDataGuard,
    secure_defaults,
    wrap_as_untrusted_data,
)
from .mcp import MCPServer
from .model import ModelFn, ToolSchema
from .policy import AgentPolicy, Tier
from .providers import AnthropicModel, MockModel
from .serde import message_from_dict, message_to_dict, outcome_from_dict, outcome_to_dict
from .store import FileStore, MemoryStore, Store
from .streaming import (
    AgentStreamEvent,
    AnswerDelta,
    Done,
    StreamingModelFn,
    ToolFinished,
    ToolStarted,
)
from .tools import Tool, ToolRegistry, tool
from .types import (
    AgentOutcome,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvents",
    "AgentOutcome",
    "AgentPolicy",
    "AgentStreamEvent",
    "AnswerDelta",
    "AnthropicModel",
    "BudgetExceeded",
    "ConfirmFn",
    "Decision",
    "Done",
    "FileStore",
    "Guard",
    "GuardContext",
    "GuardError",
    "GuardPipeline",
    "InjectionGuard",
    "Limiter",
    "LocalToolExecutor",
    "MCPServer",
    "Message",
    "MemoryStore",
    "MockModel",
    "ModelFn",
    "ModelResponse",
    "PiiRedactionGuard",
    "PiiUrlGuard",
    "RecipientTrustGuard",
    "Role",
    "Store",
    "StreamingModelFn",
    "Tier",
    "TierGuard",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolExecutor",
    "ToolFinished",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolStarted",
    "UntrustedDataGuard",
    "__version__",
    "always_approve",
    "always_deny",
    "bounded_gather",
    "console_confirm",
    "message_from_dict",
    "message_to_dict",
    "outcome_from_dict",
    "outcome_to_dict",
    "secure_defaults",
    "tool",
    "wrap_as_untrusted_data",
]
