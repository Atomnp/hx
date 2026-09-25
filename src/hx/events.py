"""Events emitted by the agent loop. The UI and everything else that reports progress listens to these."""

from dataclasses import dataclass
from typing import Callable, Union

from hx.messages import ToolCall
from hx.models.base import Usage


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


@dataclass
class TurnFinished:
    """The agent is done with the user's request."""

    text: str
    steps: int
    reason: str  # "done" | "max_steps"


Event = Union[TextDelta, ToolStarted, ToolFinished, StepFinished, TurnFinished]
EventHandler = Callable[[Event], None]


def ignore(_: Event) -> None:
    pass
