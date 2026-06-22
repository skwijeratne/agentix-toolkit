"""agentix — a generic, batteries-included agent toolkit.

Configure the agent loop, tools, guards, and observability instead of
rewriting them for every project. The core is provider-agnostic and async-first.
"""

from __future__ import annotations

from .agent import Agent
from .concurrency import Limiter, bounded_gather
from .confirm import ConfirmFn, always_approve, always_deny, console_confirm
from .context import ContextStrategy, TrimRounds, TruncateToolOutputs
from .control import Interrupt
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
from .pricing import cost_usd, register_price
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
from .subagents import subagent_tool
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
    "ContextStrategy",
    "Decision",
    "Done",
    "FileStore",
    "Guard",
    "GuardContext",
    "GuardError",
    "GuardPipeline",
    "InjectionGuard",
    "Interrupt",
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
    "TrimRounds",
    "TruncateToolOutputs",
    "UntrustedDataGuard",
    "__version__",
    "always_approve",
    "always_deny",
    "bounded_gather",
    "console_confirm",
    "cost_usd",
    "message_from_dict",
    "message_to_dict",
    "outcome_from_dict",
    "outcome_to_dict",
    "register_price",
    "secure_defaults",
    "subagent_tool",
    "tool",
    "wrap_as_untrusted_data",
]
