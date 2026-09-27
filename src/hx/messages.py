"""Conversation types (messages and tool calls) used across the harness."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """The model asking to run a tool. `id` ties the later tool result back to this call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One entry in the conversation.

    role:
      system     instructions for the model
      user       what the human typed
      assistant  what the model said (text, and/or tool calls)
      tool       the result of running one tool call
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # role == "tool": which call this answers
    name: str | None = None  # role == "tool": which tool produced it
    thinking: str = ""  # reasoning text from "thinking" models; shown, never sent back


def system(text: str) -> Message:
    return Message("system", text)


def user(text: str) -> Message:
    return Message("user", text)


def tool_result(call: ToolCall, content: str) -> Message:
    return Message("tool", content, tool_call_id=call.id, name=call.name)


def to_dict(m: Message) -> dict[str, Any]:
    """JSON-friendly form, used by session files. Empty fields are left out to keep files small."""
    return {k: v for k, v in asdict(m).items() if v not in ("", None, [])}


def from_dict(d: dict[str, Any]) -> Message:
    calls = [ToolCall(**c) for c in d.get("tool_calls", [])]
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=calls,
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
        thinking=d.get("thinking", ""),
    )
