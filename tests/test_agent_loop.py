from hx.agent import Agent
from hx.events import TextDelta, ToolFinished, TurnFinished
from hx.models.fake import FakeModel, call, say


def test_plain_reply_ends_the_turn():
    model = FakeModel([say("hello!")])
    agent = Agent(model)
    events = []

    assert agent.run("hi", on_event=events.append) == "hello!"
    assert [m.role for m in agent.messages] == ["system", "user", "assistant"]
    assert TextDelta("hello!") in events
    assert events[-1] == TurnFinished("hello!", 1, "done")


def test_tool_call_gets_a_result_and_loop_continues():
    model = FakeModel([call("read_file", path="a.py"), say("there is no such tool")])
    agent = Agent(model)
    events = []

    agent.run("read a.py", on_event=events.append)

    # The second model call saw the tool result in the conversation.
    second_request = model.requests[1]
    assert [m.role for m in second_request] == ["system", "user", "assistant", "tool"]
    assert "unknown tool" in second_request[-1].content
    assert any(isinstance(e, ToolFinished) and e.is_error for e in events)


def test_max_steps_stops_a_looping_model():
    model = FakeModel([call("x")] * 3)
    agent = Agent(model, max_steps=3)
    events = []

    assert agent.run("loop forever", on_event=events.append) == ""
    assert events[-1].reason == "max_steps"
