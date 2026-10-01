import io

from hx.agent import Agent
from hx.events import ModelCallStarted, Notice, StepFinished, TextDelta, ToolFinished, ToolStarted, TurnFinished
from hx.messages import Message, ToolCall
from hx.models.base import Usage
from hx.models.fake import FakeModel, call, say
from hx.tools import Tool, ToolRegistry, ToolResult
from hx.ui import PlainUI, RichUI


def ctrl_c(messages):
    raise KeyboardInterrupt


class SlowTool(Tool):
    name = "slow"

    def run(self, args, ctx):
        raise KeyboardInterrupt  # the user pressed Ctrl-C while this was running


def assert_valid(messages):
    calls = {c.id for m in messages for c in m.tool_calls}
    results = {m.tool_call_id for m in messages if m.role == "tool"}
    assert calls == results  # every tool call has exactly one result


def test_interrupt_during_model_call():
    agent = Agent(FakeModel([ctrl_c, say("fresh start")]))
    events = []
    assert agent.run("long task", on_event=events.append) == ""
    assert events[-1] == TurnFinished("", 0, "interrupted")
    assert "interrupted" in agent.messages[-2].content
    assert agent.run("something else") == "fresh start"  # still usable


def test_interrupt_during_tool_keeps_conversation_valid():
    two_calls = Message("assistant", tool_calls=[ToolCall("a", "slow", {}), ToolCall("b", "slow", {})])
    agent = Agent(FakeModel([two_calls]), tools=ToolRegistry([SlowTool()]))
    agent.run("go")
    assert_valid(agent.messages)
    assert all("Interrupted" in m.content for m in agent.messages if m.role == "tool")


def test_plain_ui_output(capsys):
    ui = PlainUI()
    c = ToolCall("1", "grep", {"pattern": "x"})
    for e in [ModelCallStarted(1), TextDelta("Hello"), ToolStarted(c), ToolFinished(c, "a.py:1: x"),
              Notice("cleared 2"), StepFinished(1, Usage(100, 5, 1.0), 0.25), TurnFinished("Hello", 1, "done")]:
        ui(e)
    out, err = capsys.readouterr()
    assert "Hello" in out and "→ grep" in out and "✓ a.py:1: x" in out
    assert "[hx] cleared 2" in err and "context 25%" in err


def test_rich_ui_renders_without_errors():
    from rich.console import Console

    ui = RichUI()
    ui.console = Console(file=io.StringIO(), force_terminal=True, width=100)
    c = ToolCall("1", "edit_file", {"path": "a.py"})
    diff = "Edited a.py\n--- a/a.py\n+++ b/a.py\n-old\n+new"
    for e in [ModelCallStarted(1), TextDelta("# Plan\n"), TextDelta("- step one"), ToolStarted(c),
              ToolFinished(c, diff), StepFinished(1, Usage(10, 5, 0.5), 0.1), TurnFinished("done", 1, "done")]:
        ui(e)
    ui.cleanup()
    text = ui.console.file.getvalue()
    assert "edit_file" in text and "Plan" in text and "1 step(s)" in text


def test_edit_calls_are_summarized_not_cut():
    from hx.ui import describe_call

    old = "def add(a, b):\n    result = a + b\n    if result > 500:\n        return 0\n    return result"
    c = ToolCall("1", "edit_file", {"path": "m.py", "old_string": old, "new_string": old + "\n\ndef sub(a, b):\n    return a - b"})
    assert describe_call(c) == "m.py: replace 5 line(s) with 8"
    c = ToolCall("1", "edit_file", {"path": "m.py", "old_string": "100", "new_string": "500", "replace_all": True})
    assert describe_call(c) == "m.py: replace 1 line(s) with 1, every match"
    assert describe_call(ToolCall("1", "write_file", {"path": "t.py", "content": "a\nb\n"})) == "t.py: 2 line(s)"
    assert describe_call(ToolCall("1", "todo_write", {"todos": [{}, {}]})) == "2 item(s)"


def test_long_arguments_show_that_they_were_cut():
    from hx.ui import describe_call

    text = describe_call(ToolCall("1", "bash", {"command": "x" * 100}))
    assert text == "command='" + "x" * 60 + "… (+40 chars)'"
    assert describe_call(ToolCall("1", "grep", {"pattern": "def main"})) == "pattern='def main'"
