"""Scripted model for tests: replays prepared replies and records the requests it got."""

from typing import Any, Callable

from hx.messages import Message, ToolCall
from hx.models.base import ModelResponse, TextCallback, Usage

Reply = Message | Callable[[list[Message]], Message]


def say(text: str) -> Message:
    return Message("assistant", text)


def call(name: str, call_id: str = "c1", **arguments: Any) -> Message:
    return Message("assistant", tool_calls=[ToolCall(call_id, name, arguments)])


class FakeModel:
    name = "fake"

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.requests: list[list[Message]] = []  # the conversation as sent on each call
        self.tools_seen: list[list[dict] | None] = []

    def chat(self, messages, tools=None, on_text: TextCallback | None = None) -> ModelResponse:
        self.requests.append(list(messages))
        self.tools_seen.append(tools)
        if not self.replies:
            raise AssertionError("FakeModel ran out of scripted replies")
        reply = self.replies.pop(0)
        message = reply(messages) if callable(reply) else reply
        if on_text and message.content:
            on_text(message.content)
        return ModelResponse(message, Usage(prompt_tokens=10, completion_tokens=5), "stop")
