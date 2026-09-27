from hx.agent import Agent
from hx.checkpoints import Checkpoints
from hx.context import ContextManager
from hx.messages import Message, ToolCall, from_dict, to_dict
from hx.models.fake import FakeModel, call, say
from hx.session import Session, find_session, list_sessions
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def test_message_roundtrip():
    m = Message("assistant", "hi", tool_calls=[ToolCall("c1", "grep", {"pattern": "x"})], thinking="hmm")
    assert from_dict(to_dict(m)) == m
    assert "tool_call_id" not in to_dict(Message("user", "x"))  # empty fields left out


def make_agent(tmp_path, replies, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    session = Session.create(ws, "fake", root=tmp_path / "sessions")
    agent = Agent(FakeModel(replies), tools=default_tools(), ctx=ToolContext(cwd=ws), session=session, **kw)
    return agent, session, ws


def test_replay_rebuilds_the_conversation(tmp_path):
    agent, session, ws = make_agent(tmp_path, [call("list_dir"), say("there's nothing here")])
    agent.run("what's in this folder?")
    messages, turns = session.replay()
    assert [m.role for m in messages] == [m.role for m in agent.messages]
    assert messages[-1].content == "there's nothing here"
    assert len(turns) == 1


def test_resume_continues_in_the_same_file(tmp_path):
    agent, session, ws = make_agent(tmp_path, [say("first answer")])
    agent.run("first question")

    resumed = Agent(FakeModel([say("second answer")]), ctx=ToolContext(cwd=ws))
    saved = find_session(ws, None, root=tmp_path / "sessions")
    assert saved.id == session.id
    assert resumed.resume(saved, system_prompt="NEW SYSTEM PROMPT") == 1
    assert resumed.messages[0].content == "NEW SYSTEM PROMPT"  # environment refreshed
    resumed.run("second question")

    messages, _ = Session(saved.path).replay()
    assert [m.content for m in messages if m.role == "user"] == ["first question", "second question"]


def test_undo_and_compaction_are_recorded(tmp_path):
    agent, session, ws = make_agent(tmp_path, [say("a"), say("b")],
                                    checkpoints=Checkpoints(tmp_path / "ws", root=tmp_path / "shadow"))
    agent.run("one")
    agent.run("two")
    agent.undo()
    messages, turns = session.replay()
    assert [m.content for m in messages if m.role == "user"] == ["one"]
    assert len(turns) == 1


def test_torn_last_line_is_ignored(tmp_path):
    agent, session, ws = make_agent(tmp_path, [say("ok")])
    agent.run("hello")
    with open(session.path, "a") as f:
        f.write('{"type": "message", "message": {"role": "us')  # crash mid-write
    messages, _ = session.replay()
    assert messages[-1].content == "ok"


def test_list_and_find_by_prefix(tmp_path):
    a1, s1, ws = make_agent(tmp_path, [say("x")])
    a1.run("first session question")
    sessions = list_sessions(ws, root=tmp_path / "sessions")
    assert sessions[0].title == "first session question" and sessions[0].messages == 3
    assert find_session(ws, s1.id[:12], root=tmp_path / "sessions").id == s1.id
    assert find_session(ws, "nope", root=tmp_path / "sessions") is None
