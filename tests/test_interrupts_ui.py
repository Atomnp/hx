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


LONG_RESULT = "\n".join(f"line {i}" for i in range(30))


def rich_ui(verbose):
    from rich.console import Console

    ui = RichUI(verbose=verbose)
    ui.console = Console(file=io.StringIO(), force_terminal=False, width=200)
    return ui


def run_events(ui):
    c = ToolCall("1", "write_file", {"path": "a.py", "content": "def f():\n    return 1\n"})
    for e in [ModelCallStarted(1), ToolStarted(c), ToolFinished(c, LONG_RESULT),
              StepFinished(1, Usage(2073, 124, 18.0), 0.13), TurnFinished("ok", 1, "done")]:
        ui(e)
    ui.cleanup()
    return ui.console.file.getvalue()


def test_verbose_rich_ui_shows_everything():
    text = run_events(rich_ui(verbose=True))
    assert "── model call 1 ──" in text
    assert "content:" in text and "    return 1" in text                  # full argument, real newlines
    assert "line 29" in text and "more lines" not in text                # full result
    assert "read 2,073 tokens, wrote 124 · 18.0s · context 13%" in text


def test_normal_rich_ui_stays_short():
    text = run_events(rich_ui(verbose=False))
    assert "model call 1" not in text and "content:" not in text
    assert "line 29" not in text and "18 more lines" in text


def test_show_context_prints_the_system_prompt_and_tools():
    ui = rich_ui(verbose=True)
    ui.show_context("You are hx.\n# Environment\n- Workspace: /w", ["read_file", "bash"])
    text = ui.console.file.getvalue()
    assert "system prompt (41 characters)" in text and "- Workspace: /w" in text
    assert "tools sent with every call (2): read_file, bash" in text


def test_verbose_plain_ui_prints_full_results(capsys):
    ui = PlainUI(verbose=True)
    c = ToolCall("1", "read_file", {"path": "a.py"})
    ui(ToolFinished(c, "x" * 500))
    assert "x" * 500 in capsys.readouterr().out
