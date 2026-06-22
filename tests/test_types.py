from __future__ import annotations

import agentix
from agentix import Message, ModelResponse, Role, ToolCall, ToolResult


def test_public_api_exports() -> None:
    assert agentix.__version__ == "0.1.0"
    assert Message and Role and ToolCall and ToolResult and ModelResponse


def test_message_defaults_to_untrusted() -> None:
    msg = Message(role=Role.TOOL, content="some tool output")
    assert msg.trusted is False
    assert msg.meta == {}


def test_user_message_can_be_trusted() -> None:
    msg = Message(role=Role.USER, content="hello", trusted=True)
    assert msg.trusted is True


def test_model_response_is_final_without_tool_calls() -> None:
    assert ModelResponse(text="done").is_final is True


def test_model_response_not_final_with_tool_calls() -> None:
    resp = ModelResponse(text="", tool_calls=[ToolCall(name="search", args={"q": "x"})])
    assert resp.is_final is False
    assert resp.tool_calls[0].name == "search"


def test_role_is_str_enum() -> None:
    assert Role.USER == "user"
    assert Role.SYSTEM.value == "system"
