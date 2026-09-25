"""The agent loop: call the model, run the tools it asks for, feed the results back, repeat."""

from hx.events import (
    EventHandler,
    StepFinished,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    ignore,
)
from hx.messages import Message, ToolCall, system, tool_result, user
from hx.models.base import ModelClient

DEFAULT_SYSTEM_PROMPT = "You are hx, a helpful coding assistant running in the user's terminal. Be concise."


class Agent:
    def __init__(self, model: ModelClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_steps: int = 25):
        self.model = model
        self.max_steps = max_steps  # safety stop so a confused model can't loop forever
        self.messages: list[Message] = [system(system_prompt)]

    def run(self, text: str, on_event: EventHandler = ignore) -> str:
        """Handle one user request, calling the model (and tools) as many times as needed."""
        self.messages.append(user(text))

        for step in range(1, self.max_steps + 1):
            response = self.model.chat(self.messages, on_text=lambda t: on_event(TextDelta(t)))
            self.messages.append(response.message)
            on_event(StepFinished(step, response.usage))

            if not response.message.tool_calls:
                on_event(TurnFinished(response.message.content, step, "done"))
                return response.message.content

            for call in response.message.tool_calls:
                on_event(ToolStarted(call))
                result, is_error = self.run_tool(call)
                self.messages.append(tool_result(call, result))
                on_event(ToolFinished(call, result, is_error))

        on_event(TurnFinished("", self.max_steps, "max_steps"))
        return ""

    def run_tool(self, call: ToolCall) -> tuple[str, bool]:
        # No tools exist yet. Answering with an error, instead of crashing,
        # lets the model see the problem and recover. That's how all tool errors will work.
        return f"Error: unknown tool '{call.name}'. No tools are available.", True
