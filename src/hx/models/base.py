"""The interface every model client implements."""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from hx.messages import Message

# Called with each piece of text as the model streams it, so the UI can print as it arrives.
TextCallback = Callable[[str], None]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0


@dataclass
class ModelResponse:
    message: Message  # the assistant message (text and/or tool calls)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""  # "stop", "length" (hit the output limit), ...


class ModelError(RuntimeError):
    """The model call failed (server down, bad model name, HTTP error...)."""


class ModelClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> ModelResponse: ...
