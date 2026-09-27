from hx.agent import Agent
from hx.compaction import SUMMARIZER_PROMPT, SUMMARY_HEADER, compact, render_transcript, split_point
from hx.context import ContextManager
from hx.events import Notice
from hx.messages import Message, ToolCall, system, tool_result, user
from hx.models.base import ModelResponse, Usage
from hx.models.fake import FakeModel, call, say
from hx.tools import Tool, ToolRegistry, ToolResult


class Scripted:
    """Answers summarization requests with a fixed summary; replays a script otherwise."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.summaries = 0
        self.requests = []

    def chat(self, messages, tools=None, on_text=None):
        self.requests.append(list(messages))
        if messages[0].content == SUMMARIZER_PROMPT:
            self.summaries += 1
            return ModelResponse(Message("assistant", "User wants X. Read a.py. Next: finish."), Usage(10, 5))
        return ModelResponse(self.script.pop(0), Usage(10, 5))


def convo():
    msgs = [system("SYS"), user("fix the bug")]
    for i in range(5):
        c = ToolCall(f"c{i}", "read_file", {"path": f"f{i}.py"})
        msgs += [Message("assistant", f"step {i}", tool_calls=[c]), tool_result(c, f"content {i}")]
    return msgs


def test_render_transcript_shortens_results():
    c = ToolCall("1", "read_file", {"path": "a.py"})
    text = render_transcript([user("hi"), Message("assistant", tool_calls=[c]), tool_result(c, "z" * 5000)], budget_chars=1000)
    assert "USER: hi" in text and 'CALLED read_file({"path": "a.py"})' in text and "[5000 chars total]" in text


def test_split_never_orphans_tool_results():
    msgs = convo()
    for keep in range(1, 8):
        i = split_point(msgs, keep)
        assert msgs[i].role != "tool"


def test_compact_keeps_system_summary_and_tail():
    model = Scripted([])
    msgs = convo()
    new, summary = compact(model, msgs, keep_recent=2)
    assert new[0].content == "SYS"
    assert new[1].role == "user" and new[1].content.startswith(SUMMARY_HEADER) and summary in new[1].content
    assert new[-1] is msgs[-1] and new[-2] is msgs[-2]  # the tail is kept verbatim
    assert len(new) < len(msgs)


class Big(Tool):
    name = "big"

    def run(self, args, ctx):
        return ToolResult("y" * 2500)


def test_agent_compacts_automatically_and_keeps_working():
    script = [Message("assistant", tool_calls=[ToolCall(f"c{i}", "big", {"n": i})]) for i in range(8)] + [say("done")]
    model = Scripted(script)
    cm = ContextManager(window=3000, reserve=200, keep_recent=1, clear_at=0.99, compact_at=0.85)  # force compaction
    agent = Agent(model, tools=ToolRegistry([Big()]), context=cm)
    events = []
    assert agent.run("go", on_event=events.append) == "done"
    assert model.summaries >= 1
    assert any(isinstance(e, Notice) and "compacted" in e.text for e in events)
    assert any(SUMMARY_HEADER in m.content for m in agent.messages)


def test_undo_after_compaction_restores_files_note(tmp_path):
    model = FakeModel([say("one"), say("two")])
    agent = Agent(model)
    agent.run("first")
    agent.turns = [(None, None)]  # as if compacted
    agent.undo()
    assert "/undo" in agent.messages[-2].content


def test_single_huge_turn_escalates_instead_of_overflowing():
    # One turn, three big results, tiny window: nothing is "old", so the ladder must escalate.
    script = [Message("assistant", tool_calls=[ToolCall(f"c{i}", "big", {"n": i})]) for i in range(3)] + [say("ok")]
    model = Scripted(script)
    cm = ContextManager(window=1400, reserve=200)  # each capped result is ~350 tokens; three overflow the 1200 budget
    agent = Agent(model, tools=ToolRegistry([Big()]), context=cm)
    events = []
    agent.run("read three big things", on_event=events.append)
    for request in model.requests:
        if request[0].content != SUMMARIZER_PROMPT:
            assert cm.estimate(request) <= cm.budget  # no request exceeded the window
    assert any(isinstance(e, Notice) for e in events)


def test_earlier_summary_is_carried_forward_verbatim():
    c = ToolCall("c1", "read_file", {"path": "b.py"})
    msgs = [system("SYS"), user(f"{SUMMARY_HEADER}\n\npermissions.py defines READ_ONLY_COMMANDS"),
            Message("assistant", "Understood, continuing from the summary."), user("read b.py"),
            Message("assistant", tool_calls=[c]), tool_result(c, "b content"), user("next"), Message("assistant", "ok")]
    model = Scripted([])
    new, summary = compact(model, msgs, keep_recent=2)
    assert "permissions.py defines READ_ONLY_COMMANDS" in new[1].content  # kept by code, not by the model
    assert "User wants X" in new[1].content  # plus the new summary
    sent = model.requests[0][1].content
    assert "READ_ONLY_COMMANDS" not in sent and "Understood, continuing" not in sent  # only fresh messages summarized


def test_results_share_the_render_budget():
    calls = [ToolCall(f"c{i}", "read_file", {"path": f"{i}.py"}) for i in range(2)]
    msgs = []
    for c in calls:
        msgs += [Message("assistant", tool_calls=[c]), tool_result(c, "q" * 10_000)]
    text = render_transcript(msgs, budget_chars=8_000)
    assert 6_000 < len(text) < 9_500  # each result got ~half the budget, not a fixed small slice
