"""agentix — a generic, batteries-included agent toolkit.

Configure the agent loop, tools, guards, and observability instead of
rewriting them for every project. The core is provider-agnostic and async-first.
"""

from __future__ import annotations

from .agent import Agent
from .concurrency import Limiter, bounded_gather
from .confirm import ConfirmFn, always_approve, always_deny, console_confirm
from .consistency import SelfConsistencyModel
from .content import (
    AudioPart,
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
)
from .context import ContextStrategy, TrimRounds, TruncateToolOutputs
from .control import Interrupt
from .errors import AgentError, BudgetExceeded, GuardError, ToolError
from .evals import (
    Case,
    CaseResult,
    EvalReport,
    Score,
    Scorer,
    contains,
    evaluate,
    exact_match,
    llm_judge,
    predicate,
    regex_match,
)
from .events import AgentEvents
from .executors import LocalToolExecutor, ToolExecutor
from .guards import (
    CallbackGuard,
    Decision,
    Guard,
    GuardContext,
    GuardPipeline,
    InjectionGuard,
    JudgeGuard,
    PiiRedactionGuard,
    PiiUrlGuard,
    RecipientTrustGuard,
    TierGuard,
    ToolAllowlistGuard,
    UntrustedDataGuard,
    secure_defaults,
    wrap_as_untrusted_data,
)
from .mcp import MCPServer
from .model import ModelFn, ToolSchema
from .policy import AgentPolicy, Tier
from .pricing import cost_usd, register_price
from .prompts import PromptRegistry, PromptVersion
from .providers import (
    AnthropicModel,
    BedrockModel,
    GeminiModel,
    LiteLLMModel,
    MockModel,
    OllamaModel,
    OpenAIModel,
)
from .resilience import FallbackModel, RetryModel
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
from .tracing import TracingModel, trace_run, tracing_events
from .types import (
    AgentOutcome,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolResult,
)
from .validation import OutputValidator, json_output, pydantic_output, regex_output

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("agentix-toolkit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvents",
    "AgentOutcome",
    "AgentPolicy",
    "AgentStreamEvent",
    "AnswerDelta",
    "AnthropicModel",
    "AudioPart",
    "BedrockModel",
    "BudgetExceeded",
    "CallbackGuard",
    "Case",
    "CaseResult",
    "ConfirmFn",
    "ContentPart",
    "ContextStrategy",
    "Decision",
    "DocumentPart",
    "Done",
    "EvalReport",
    "FallbackModel",
    "FileStore",
    "GeminiModel",
    "Guard",
    "GuardContext",
    "GuardError",
    "GuardPipeline",
    "ImagePart",
    "InjectionGuard",
    "Interrupt",
    "JudgeGuard",
    "Limiter",
    "LiteLLMModel",
    "LocalToolExecutor",
    "MCPServer",
    "Message",
    "MemoryStore",
    "MockModel",
    "ModelFn",
    "ModelResponse",
    "OllamaModel",
    "OpenAIModel",
    "OutputValidator",
    "PiiRedactionGuard",
    "PiiUrlGuard",
    "PromptRegistry",
    "PromptVersion",
    "RecipientTrustGuard",
    "RetryModel",
    "Role",
    "Score",
    "Scorer",
    "SelfConsistencyModel",
    "Store",
    "StreamingModelFn",
    "TextPart",
    "Tier",
    "TierGuard",
    "Tool",
    "ToolAllowlistGuard",
    "ToolCall",
    "ToolError",
    "ToolExecutor",
    "ToolFinished",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolStarted",
    "TracingModel",
    "TrimRounds",
    "TruncateToolOutputs",
    "UntrustedDataGuard",
    "__version__",
    "always_approve",
    "always_deny",
    "bounded_gather",
    "console_confirm",
    "contains",
    "cost_usd",
    "evaluate",
    "exact_match",
    "json_output",
    "llm_judge",
    "message_from_dict",
    "message_to_dict",
    "outcome_from_dict",
    "outcome_to_dict",
    "predicate",
    "pydantic_output",
    "regex_match",
    "regex_output",
    "register_price",
    "secure_defaults",
    "subagent_tool",
    "tool",
    "trace_run",
    "tracing_events",
    "wrap_as_untrusted_data",
]
