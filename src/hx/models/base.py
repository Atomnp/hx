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
    """The model call failed (server down, bad model name, HTTP error...).

    retryable:   worth trying again (connection refused, timeouts, HTTP 429/500/502/503/504/529)
    retry_after: seconds the server asked us to wait (HTTP Retry-After), if it said
    """

    def __init__(self, message: str, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def http_error(status: int, detail: str, retry_after: str | None = None) -> ModelError:
    wait = None
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            pass
    return ModelError(f"HTTP {status}: {detail[:500]}", retryable=status in RETRYABLE_STATUS, retry_after=wait)


class ModelClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> ModelResponse: ...
