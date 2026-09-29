"""The agent loop: call the model, run the tools it asks for, feed the results back, repeat."""

from hx.checkpoints import CheckpointError, Checkpoints
from hx.compaction import compact
from hx.context import CHARS_PER_TOKEN, ContextManager
from hx.events import (
    EventHandler,
    ModelCallStarted,
    TurnStarted,
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
from hx.session import Session
from hx.tools.todo import render, unfinished
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
        context: ContextManager | None = None,
        session: Session | None = None,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.ctx = ctx or ToolContext()
        self.max_steps = max_steps  # safety stop so a confused model can't loop forever
        self.messages: list[Message] = [system(system_prompt)]
        self.checkpoints = checkpoints
        self.context = context or ContextManager()
        # One entry per turn: (snapshot taken before the turn, conversation length before the turn).
        # The length becomes None after compaction: those messages no longer exist individually.
        self.turns: list[tuple[str | None, int | None]] = []
        self.session = session  # where the conversation is recorded; None = not recorded
        # UserPromptSubmit/Stop hooks fire for the user's own turns, not for subagents' delegated tasks.
        # (Tool hooks still apply to subagents: they run inside the tool registry.)
        self.lifecycle_hooks = True
        if session:
            session.message(self.messages[0])

    # ---------------------------------------------------------------- recording

    def add(self, m: Message) -> None:
        """Every message enters the conversation through here, so the session log never misses one."""
        self.messages.append(m)
        if self.session:
            self.session.message(m)

    def rewrote_history(self) -> None:
        """Called after compaction or undo: the log records the new full state."""
        if self.session:
            self.session.reset(self.messages, self.turns)

    def resume(self, session: Session, system_prompt: str | None = None) -> int:
        """Load a saved conversation into this agent and keep recording into the same file.
        A fresh system prompt replaces the saved one: the environment (date, git status) may have changed."""
        messages, turns = session.replay()
        if system_prompt and messages and messages[0].role == "system":
            messages[0] = system(system_prompt)
        self.messages, self.turns, self.session = messages, turns, session
        self.ctx.todos = last_todos(messages)
        self.rewrote_history()
        return sum(1 for m in messages if m.role == "user")

    def run(self, text: str, on_event: EventHandler = ignore) -> str:
        """Handle one user request. Ctrl-C (KeyboardInterrupt) ends the turn cleanly instead of crashing hx."""
        try:
            return self._run(text, on_event)
        except KeyboardInterrupt:
            self.repair_after_interrupt()
            on_event(TurnFinished("", 0, "interrupted"))
            return ""

    def repair_after_interrupt(self) -> None:
        """Leave the conversation valid after Ctrl-C: every tool call needs a result, and the model should know
        the user stopped it (otherwise it may resume the abandoned plan on the next turn)."""
        answered = {m.tool_call_id for m in self.messages if m.role == "tool"}
        for m in reversed(self.messages):
            if m.role == "assistant" and m.tool_calls:
                for c in m.tool_calls:
                    if c.id not in answered:
                        self.add(tool_result(c, "Interrupted by the user before this finished."))
                break
            if m.role == "user":
                break
        self.add(user("[The user interrupted this turn. Stop and wait for their next instruction.]"))
        self.add(Message("assistant", "Stopped."))

    def _run(self, text: str, on_event: EventHandler) -> str:
        on_event(TurnStarted(text))
        self.ctx.on_event = on_event
        hooks = self.ctx.hooks if self.lifecycle_hooks else None
        if hooks:
            submit = hooks.run("UserPromptSubmit", {"prompt": text})
            for err in submit.errors:
                on_event(Notice(err))
            if submit.blocked:
                on_event(Notice(f"your message was blocked by a UserPromptSubmit hook: {submit.feedback}"))
                return ""
            if submit.output:
                text += f"\n\n<system-reminder>Context from a UserPromptSubmit hook:\n{submit.output}</system-reminder>"
        turn = (self.take_snapshot(f"before turn {len(self.turns) + 1}: {text[:60]}", on_event), len(self.messages))
        self.turns.append(turn)
        if self.session:
            self.session.turn(*turn)
        self.add(user(text))
        guard = LoopGuard()
        reminded = False
        stop_hook_runs = 0

        self.context.set_tools(self.tools.schemas())
        for step in range(1, self.max_steps + 1):
            self.fit_context(on_event)
            on_event(ModelCallStarted(step))
            response = self.model.chat(
                self.messages,
                tools=self.tools.schemas() or None,
                on_text=lambda t: on_event(TextDelta(t)),
            )
            self.context.calibrate(self.messages, response.usage.prompt_tokens)
            message = response.message
            if not message.tool_calls and self.tools.names():
                # Some models write tool calls as text instead of using the tool-call channel.
                calls, rest = extract_text_tool_calls(message.content, set(self.tools.names()))
                if calls:
                    message.tool_calls, message.content = calls, rest
                    on_event(Notice(f"parsed {len(calls)} tool call(s) the model wrote as text"))
            self.add(message)
            on_event(StepFinished(step, response.usage, self.context.fraction_used(self.messages)))

            if not message.tool_calls:
                open_items = unfinished(self.ctx.todos)
                if open_items and not reminded:
                    # The model wants to stop with plan items still open. Remind it once; a second stop is final.
                    reminded = True
                    on_event(Notice(f"{len(open_items)} todo item(s) still open; reminding the model"))
                    self.add(user(
                        "<system-reminder>Your todo list still has unfinished items:\n"
                        f"{render(open_items)}\nContinue working on them, or update the list (mark items completed, "
                        "or remove ones that no longer apply) and explain why.</system-reminder>"
                    ))
                    continue
                if hooks and stop_hook_runs < 3:
                    # Stop hook: e.g. "run the tests; if they fail, exit 2 with the failures". Capped at 3 so a
                    # hook that always objects can't keep the agent running forever.
                    stop = hooks.run("Stop", {"final_message": message.content, "stop_hook_active": stop_hook_runs > 0})
                    for err in stop.errors:
                        on_event(Notice(err))
                    if stop.blocked:
                        stop_hook_runs += 1
                        on_event(Notice("a Stop hook asked the agent to keep working"))
                        self.add(user(f"<system-reminder>A Stop hook objected:\n{stop.feedback}</system-reminder>"))
                        continue
                on_event(TurnFinished(message.content, step, "done"))
                return message.content

            for call in message.tool_calls:
                on_event(ToolStarted(call))
                result, is_error = self.run_tool(call)
                result = self.context.truncate_tool_output(result)
                if warning := guard.check(call):
                    result += f"\n[harness: {warning}]"
                    on_event(Notice(f"repeated call detected: {call.name}"))
                self.add(tool_result(call, result))
                on_event(ToolFinished(call, result, is_error))

        on_event(TurnFinished("", self.max_steps, "max_steps"))
        return ""

    def run_tool(self, call: ToolCall) -> tuple[str, bool]:
        result = self.tools.execute(call, self.ctx)
        return result.content, result.is_error

    # ---------------------------------------------------------------- context management

    def fit_context(self, on_event: EventHandler) -> None:
        """Escalation ladder, cheapest first, so the prompt never silently overflows the window."""
        cm = self.context
        notes = cm.clear_old_results(self.messages)  # 1. stub old tool results (free)
        if cm.fraction_used(self.messages) >= cm.compact_at:
            report = self.compact()  # 2. summarize older messages (one model call)
            if report:
                notes.append(report)
        if cm.fraction_used(self.messages) >= cm.compact_at:
            notes += cm.clear_old_results(self.messages, keep_recent=1, force=True)  # 3. keep only the newest result
        if cm.fraction_used(self.messages) >= 1.0:
            notes += cm.shrink_last_result(self.messages)  # 4. cut the newest result itself
        if cm.fraction_used(self.messages) >= 1.0:
            notes.append("WARNING: the conversation still exceeds the context window; early messages may be lost")
        for note in notes:
            on_event(Notice(note))

    # ---------------------------------------------------------------- compaction

    def compact(self) -> str:
        """Summarize older messages to free up context. Returns a one-line report."""
        before = self.context.estimate(self.messages)
        budget_chars = int(self.context.budget * 0.6 * CHARS_PER_TOKEN)  # what the summarizer may read
        self.messages, summary = compact(self.model, self.messages, self.context.keep_recent, budget_chars)
        if not summary:
            return ""
        # Earlier turns can no longer be rewound message by message; undo will still restore their files.
        self.turns = [(sha, None) for sha, _ in self.turns]
        self.rewrote_history()
        return f"compacted the conversation: ~{before} → ~{self.context.estimate(self.messages)} tokens"

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
        if length is not None:
            del self.messages[length:]
        else:  # compacted away: we can't cut the conversation, so tell the model instead
            self.messages.append(user("[The user ran /undo: the file changes from the last turn were reverted.]"))
            self.messages.append(Message("assistant", "Noted: the last turn's file changes were reverted."))
        self.rewrote_history()
        self.ctx.read_files.clear()  # file contents changed under the model: it must re-read before editing
        return f"Undid the last turn.\n{summary or '(no file changes)'}"


def last_todos(messages: list[Message]) -> list[dict]:
    """The plan as of the last todo_write call, so a resumed session keeps its todo list."""
    for m in reversed(messages):
        for c in reversed(m.tool_calls):
            if c.name == "todo_write" and isinstance(c.arguments.get("todos"), list):
                return c.arguments["todos"]
    return []


def make_child_agent(model, tools, ctx, instructions, max_steps):
    """Factory for subagents (hx.subagents.Task): a plain Agent, no checkpoints/session of its own."""
    from hx.prompt import environment

    child = Agent(model, tools=tools, ctx=ctx, system_prompt=f"{instructions}\n\n{environment(ctx)}", max_steps=max_steps)
    child.lifecycle_hooks = False
    return child
