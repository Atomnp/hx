from hx.agent import Agent, make_child_agent
from hx.events import Notice
from hx.models.fake import FakeModel, call, say
from hx.subagents import BUILTIN, Task, load_specs, parse_agent_file
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def setup(tmp_path, replies, specs=BUILTIN):
    (tmp_path / "a.py").write_text("SECRET_SAUCE = 42\n")
    model = FakeModel(replies)
    tools = default_tools()
    tools.register(Task(specs, tools, model, make_child_agent))
    return model, Agent(model, tools=tools, ctx=ToolContext(cwd=tmp_path))


def test_child_works_in_its_own_context_and_reports_back(tmp_path):
    model, agent = setup(tmp_path, [
        call("task", agent="explore", task="Where is SECRET_SAUCE defined?"),   # parent delegates
        call("grep", call_id="g1", pattern="SECRET_SAUCE"),                       # child searches
        say("SECRET_SAUCE is defined in a.py:1."),                               # child reports
        say("It's in a.py, line 1."),                                             # parent answers
    ])
    events = []
    assert agent.run("where is SECRET_SAUCE?", on_event=events.append) == "It's in a.py, line 1."

    # The child's grep result never entered the parent's conversation; only the report did.
    parent_text = "\n".join(m.content for m in agent.messages)
    assert "a.py:1: SECRET_SAUCE = 42" not in parent_text
    assert "SECRET_SAUCE is defined in a.py:1." in parent_text and "1 tool call(s)" in parent_text
    # The child started from a fresh conversation: its own system prompt + the task.
    child_first_request = model.requests[1]
    assert [m.role for m in child_first_request] == ["system", "user"]
    assert "exploration subagent" in child_first_request[0].content
    assert any(isinstance(e, Notice) and "[explore] → grep" in e.text for e in events)


def test_explore_cannot_edit_and_nobody_can_recurse(tmp_path):
    model, agent = setup(tmp_path, [
        call("task", agent="explore", task="change a.py"),
        call("edit_file", call_id="e1", path="a.py", old_string="42", new_string="0"),
        call("task", call_id="t2", agent="explore", task="recurse"),
        say("I can't edit."),
        say("ok"),
    ])
    agent.run("go")
    child_results = [m.content for m in model.requests[3] if m.role == "tool"]
    assert "unknown tool 'edit_file'" in child_results[0]
    assert "unknown tool 'task'" in child_results[1]
    assert (tmp_path / "a.py").read_text() == "SECRET_SAUCE = 42\n"


def test_custom_agent_from_markdown(tmp_path):
    d = tmp_path / ".hx" / "agents"
    d.mkdir(parents=True)
    (d / "reviewer.md").write_text("---\nname: reviewer\ndescription: Reviews code\ntools: read_file, grep\n---\nYou review code.\n")
    spec = parse_agent_file(d / "reviewer.md")
    assert spec.name == "reviewer" and spec.tools == ["read_file", "grep"] and spec.instructions == "You review code."
    names = [s.name for s in load_specs(tmp_path)]
    assert names[:2] == ["explore", "general"] and "reviewer" in names


def test_task_schema_lists_agent_types(tmp_path):
    model, agent = setup(tmp_path, [])
    schema = agent.tools.get("task").schema()["function"]
    assert schema["parameters"]["properties"]["agent"]["enum"] == ["explore", "general"]
    assert "explore:" in schema["description"]
