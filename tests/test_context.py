from hx.agent import Agent
from hx.context import CLEARED, ContextManager
from hx.events import Notice
from hx.messages import Message, ToolCall, system, tool_result, user
from hx.models.fake import FakeModel, call, say
from hx.tools import Tool, ToolRegistry, ToolResult


def convo(n_results: int, size: int) -> list[Message]:
    msgs = [system("sys"), user("go")]
    for i in range(n_results):
        c = ToolCall(f"c{i}", "read_file", {"path": f"f{i}.py"})
        msgs += [Message("assistant", tool_calls=[c]), tool_result(c, "x" * size)]
    return msgs


def test_estimate_and_calibration_are_clamped():
    cm = ContextManager()
    msgs = [user("a" * 350)]  # ~100 tokens by the heuristic
    base = cm.estimate(msgs)
    cm.calibrate(msgs, base * 2)  # the model says it's really twice as many
    assert cm.ratio > 1.0 and cm.estimate(msgs) > base
    for _ in range(50):
        cm.calibrate(msgs, 1)  # absurd readings can't push the ratio below the clamp
    assert cm.ratio >= 0.5


def test_tool_schemas_count_toward_the_estimate():
    cm = ContextManager()
    before = cm.estimate([user("hi")])
    cm.set_tools([{"type": "function", "function": {"name": "x", "description": "y" * 700}}])
    assert cm.estimate([user("hi")]) >= before + 200


def test_huge_tool_output_is_cut_in_the_middle():
    cm = ContextManager(window=1000)  # a result may use 250 tokens, about 875 chars
    out = cm.truncate_tool_output("HEAD" + "m" * 5000 + "TAIL")
    assert out.startswith("HEAD") and out.endswith("TAIL") and "characters omitted" in out
    assert len(out) < 1200
    assert cm.truncate_tool_output("small") == "small"


def test_old_results_cleared_but_recent_kept():
    cm = ContextManager(window=4000, reserve=0, keep_recent=2)
    msgs = convo(n_results=8, size=1400)  # ~3200 tokens: over 70% of 4000
    notes = cm.clear_old_results(msgs)
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert notes and "cleared" in notes[0]
    assert tool_msgs[0].content.startswith(CLEARED)
    assert not tool_msgs[-1].content.startswith(CLEARED) and not tool_msgs[-2].content.startswith(CLEARED)
    assert cm.fraction_used(msgs) < 0.5 + 0.05


def test_under_threshold_nothing_is_cleared():
    cm = ContextManager(window=100_000)
    msgs = convo(3, 100)
    assert cm.clear_old_results(msgs) == []


class Big(Tool):
    name = "big"

    def run(self, args, ctx):
        return ToolResult("y" * 3000)


def test_agent_keeps_conversation_in_budget():
    model = FakeModel([call("big", call_id=f"c{i}") for i in range(6)] + [say("done")])
    agent = Agent(model, tools=ToolRegistry([Big()]), context=ContextManager(window=4000, reserve=500, keep_recent=1))
    events = []
    agent.run("go", on_event=events.append)
    assert any(isinstance(e, Notice) and "cleared" in e.text for e in events)
    last_request = model.requests[-1]
    assert any(m.role == "tool" and m.content.startswith(CLEARED) for m in last_request)
