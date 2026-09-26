"""The agent loop: call the model, run the tools it asks for, feed the results back, repeat."""

from hx.checkpoints import CheckpointError, Checkpoints
from hx.events import (
    EventHandler,
    Notice,
    StepFinished,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    ignore,
)
from hx.messages import Message, ToolCall, system, tool_result, user
from hx.models.base import ModelClient
from hx.repair import LoopGuard, extract_text_tool_calls
from hx.tools import ToolContext, ToolRegistry

DEFAULT_SYSTEM_PROMPT = "You are hx, a helpful coding assistant running in the user's terminal. Be concise."


class Agent:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        ctx: ToolContext | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 25,
        checkpoints: Checkpoints | None = None,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.ctx = ctx or ToolContext()
        self.max_steps = max_steps  # safety stop so a confused model can't loop forever
        self.messages: list[Message] = [system(system_prompt)]
        self.checkpoints = checkpoints
        # One entry per turn: (snapshot taken before the turn, conversation length before the turn).
        self.turns: list[tuple[str | None, int]] = []

    def run(self, text: str, on_event: EventHandler = ignore) -> str:
        """Handle one user request, calling the model (and tools) as many times as needed."""
        self.turns.append((self.take_snapshot(f"before turn {len(self.turns) + 1}: {text[:60]}", on_event), len(self.messages)))
        self.messages.append(user(text))
        guard = LoopGuard()

        for step in range(1, self.max_steps + 1):
            response = self.model.chat(
                self.messages,
                tools=self.tools.schemas() or None,
                on_text=lambda t: on_event(TextDelta(t)),
            )
            message = response.message
            if not message.tool_calls and self.tools.names():
                # Some models write tool calls as text instead of using the tool-call channel.
                calls, rest = extract_text_tool_calls(message.content, set(self.tools.names()))
                if calls:
                    message.tool_calls, message.content = calls, rest
                    on_event(Notice(f"parsed {len(calls)} tool call(s) the model wrote as text"))
            self.messages.append(message)
            on_event(StepFinished(step, response.usage))

            if not message.tool_calls:
                on_event(TurnFinished(message.content, step, "done"))
                return message.content

            for call in message.tool_calls:
                on_event(ToolStarted(call))
                result, is_error = self.run_tool(call)
                if warning := guard.check(call):
                    result += f"\n[harness: {warning}]"
                    on_event(Notice(f"repeated call detected: {call.name}"))
                self.messages.append(tool_result(call, result))
                on_event(ToolFinished(call, result, is_error))

        on_event(TurnFinished("", self.max_steps, "max_steps"))
        return ""

    def run_tool(self, call: ToolCall) -> tuple[str, bool]:
        result = self.tools.execute(call, self.ctx)
        return result.content, result.is_error

    # ---------------------------------------------------------------- checkpoints

    def take_snapshot(self, label: str, on_event: EventHandler) -> str | None:
        if not self.checkpoints:
            return None
        try:
            return self.checkpoints.snapshot(label)
        except CheckpointError as e:
            on_event(Notice(f"checkpoint failed, undo won't be available for this turn: {e}"))
            return None

    def undo(self) -> str:
        """Undo the last turn: restore the files AND rewind the conversation to before it.
        Rewinding matters: otherwise the model would believe its reverted changes still exist."""
        if not self.turns:
            return "Nothing to undo."
        sha, length = self.turns.pop()
        summary = ""
        if sha and self.checkpoints:
            summary = self.checkpoints.restore(sha)
        del self.messages[length:]
        self.ctx.read_files.clear()  # file contents changed under the model: it must re-read before editing
        return f"Undid the last turn.\n{summary or '(no file changes)'}"
