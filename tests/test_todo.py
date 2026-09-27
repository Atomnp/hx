from hx.agent import Agent, last_todos
from hx.events import Notice
from hx.messages import Message, ToolCall
from hx.models.fake import FakeModel, say
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def todo_call(items, call_id="t1"):
    todos = [{"content": c, "status": s} for c, s in items]
    return Message("assistant", tool_calls=[ToolCall(call_id, "todo_write", {"todos": todos})])


def run_tool(ctx, items):
    todos = [{"content": c, "status": s} for c, s in items]
    return default_tools().execute(ToolCall("t", "todo_write", {"todos": todos}), ctx)


def test_renders_and_stores_the_list():
    ctx = ToolContext()
    r = run_tool(ctx, [("read code", "completed"), ("fix bug", "in_progress"), ("run tests", "pending")])
    assert not r.is_error and "(1/3 done)" in r.content
    assert "[x] read code\n[~] fix bug\n[ ] run tests" in r.content
    assert len(ctx.todos) == 3


def test_only_one_in_progress():
    r = run_tool(ToolContext(), [("a", "in_progress"), ("b", "in_progress")])
    assert r.is_error and "only one" in r.content


def test_bad_status_rejected_by_schema():
    r = run_tool(ToolContext(), [("a", "doing")])
    assert r.is_error and "must be one of" in r.content


def test_agent_reminds_once_when_stopping_with_open_items():
    model = FakeModel([
        todo_call([("fix bug", "in_progress"), ("run tests", "pending")]),
        say("I fixed it."),                                   # stops early: harness reminds
        todo_call([("fix bug", "completed"), ("run tests", "completed")], "t2"),
        say("Fixed and tested."),
    ])
    agent = Agent(model, tools=default_tools())
    events = []
    assert agent.run("fix the bug and test it", on_event=events.append) == "Fixed and tested."
    reminders = [m for m in agent.messages if "<system-reminder>" in m.content]
    assert len(reminders) == 1 and "run tests" in reminders[0].content
    assert any(isinstance(e, Notice) and "still open" in e.text for e in events)


def test_second_stop_is_final():
    model = FakeModel([todo_call([("a", "pending")]), say("stop"), say("really stop")])
    agent = Agent(model, tools=default_tools())
    assert agent.run("go") == "really stop"


def test_todos_restored_from_history():
    msgs = [todo_call([("old", "pending")]), todo_call([("new", "in_progress")], "t2")]
    assert last_todos(msgs) == [{"content": "new", "status": "in_progress"}]
