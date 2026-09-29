"""Events emitted by the agent loop. The UI and everything else that reports progress listens to these."""

from dataclasses import dataclass
from typing import Callable, Union

from hx.messages import ToolCall
from hx.models.base import Usage


@dataclass
class TurnStarted:
    """The agent received a user request (tracing opens the turn's root span here)."""

    text: str


@dataclass
class ModelCallStarted:
    """A model call is about to begin (UIs show a spinner until text or a tool call arrives)."""

    step: int


@dataclass
class TextDelta:
    """A piece of streamed assistant text."""

    text: str


@dataclass
class ToolStarted:
    call: ToolCall


@dataclass
class ToolFinished:
    call: ToolCall
    result: str
    is_error: bool = False


@dataclass
class StepFinished:
    """One model call is done (a turn can take many steps when tools are involved)."""

    step: int
    usage: Usage
    context_used: float = 0.0  # share of the context budget the conversation now uses (0..1)


@dataclass
class Notice:
    """Something the harness did on its own that the user should know about (a repair, a warning...)."""

    text: str


@dataclass
class TurnFinished:
    """The agent is done with the user's request."""

    text: str
    steps: int
    reason: str  # "done" | "max_steps" | "interrupted"


Event = Union[TurnStarted, ModelCallStarted, TextDelta, ToolStarted, ToolFinished, StepFinished, Notice, TurnFinished]
EventHandler = Callable[[Event], None]


def ignore(_: Event) -> None:
    pass
