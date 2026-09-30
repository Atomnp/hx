from hx.agent import Agent
from hx.events import Notice
from hx.messages import Message, ToolCall
from hx.models.fake import FakeModel, say
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def call(name, call_id="c1", **args):
    return Message("assistant", tool_calls=[ToolCall(call_id, name, args)])


def write(path="calc.py", call_id="w1"):
    return call("write_file", call_id, path=path, content="def add(a, b):\n    return a + b\n")


def reminders(agent):
    return [m for m in agent.messages if m.role == "user" and "haven't run anything" in m.content]


def make_agent(tmp_path, replies, tools=None):
    return Agent(FakeModel(replies), tools=tools or default_tools(), ctx=ToolContext(cwd=tmp_path))


def test_reminds_once_when_stopping_after_an_unchecked_edit(tmp_path):
    agent = make_agent(tmp_path, [
        write(),
        say("Done, and verified."),                          # claims it, ran nothing: harness reminds
        call("bash", "b1", command="python3 -c 'import calc; assert calc.add(1, 2) == 3'"),
        say("Ran a check: add works."),
    ])
    events = []
    assert agent.run("add an add function", on_event=events.append) == "Ran a check: add works."
    assert len(reminders(agent)) == 1 and "calc.py" in reminders(agent)[0].content
    assert any(isinstance(e, Notice) and "not checked yet" in e.text for e in events)


def test_no_reminder_when_a_command_ran_after_the_edit(tmp_path):
    agent = make_agent(tmp_path, [write(), call("bash", "b1", command="true"), say("Done; the check passed.")])
    assert agent.run("go") == "Done; the check passed."
    assert reminders(agent) == []


def test_an_edit_after_the_last_command_still_counts(tmp_path):
    agent = make_agent(tmp_path, [
        call("bash", "b1", command="true"),
        write(),
        say("Done."),
        say("Nothing to run for this change."),
    ])
    assert agent.run("go") == "Nothing to run for this change."
    assert len(reminders(agent)) == 1


def test_second_stop_is_final(tmp_path):
    agent = make_agent(tmp_path, [write(), say("Done."), say("There is nothing to run here.")])
    assert agent.run("go") == "There is nothing to run here."
    assert len(reminders(agent)) == 1


def test_failed_edits_dont_count(tmp_path):
    # edit_file on a file that doesn't exist fails: nothing changed, nothing to verify
    agent = make_agent(tmp_path, [call("edit_file", path="nope.py", old_string="a", new_string="b"), say("Couldn't.")])
    assert agent.run("go") == "Couldn't."
    assert reminders(agent) == []


def test_lists_each_file_once_relative_to_the_workspace(tmp_path):
    agent = make_agent(tmp_path, [
        write(str(tmp_path / "calc.py")),
        write("calc.py", "w2"),
        write("test_calc.py", "w3"),
        say("Done."),
        say("Not verified."),
    ])
    agent.run("go")
    text = reminders(agent)[0].content
    assert "calc.py, test_calc.py" in text and str(tmp_path) not in text


def test_no_reminder_without_a_bash_tool(tmp_path):
    tools = default_tools()
    tools._tools.pop("bash")
    agent = make_agent(tmp_path, [write(), say("Done.")], tools=tools)
    assert agent.run("go") == "Done."


def test_ablation_switch_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HX_ABLATE", "verify")
    agent = make_agent(tmp_path, [write(), say("Done.")])
    assert agent.run("go") == "Done."
    assert reminders(agent) == []
