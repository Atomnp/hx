import json

from hx.agent import Agent
from hx.headless import event_to_dict, run_headless
from hx.events import ToolStarted
from hx.messages import ToolCall
from hx.models import ModelError
from hx.models.fake import FakeModel, call, say
from hx.permissions import PermissionPolicy
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def agent_for(tmp_path, replies, mode="ask"):
    return Agent(FakeModel(replies), tools=default_tools(),
                 ctx=ToolContext(cwd=tmp_path, permissions=PermissionPolicy(mode)))  # approve=None: headless


def test_text_output_and_exit_code(tmp_path, capsys):
    code = run_headless(agent_for(tmp_path, [call("list_dir"), say("It's empty.")]), "what's here?")
    out, err = capsys.readouterr()
    assert code == 0 and out.strip() == "It's empty." and "→ list_dir" in err


def test_json_summary(tmp_path, capsys):
    run_headless(agent_for(tmp_path, [call("list_dir"), say("done")]), "go", "json")
    summary = json.loads(capsys.readouterr().out)
    assert summary["result"] == "done" and summary["finish_reason"] == "done" and summary["steps"] == 2
    assert summary["usage"] == {"input_tokens": 20, "output_tokens": 10, "model_calls": 2}
    assert summary["tool_calls"] == [{"name": "list_dir", "arguments": {}, "is_error": False}]


def test_stream_json_is_ndjson(tmp_path, capsys):
    run_headless(agent_for(tmp_path, [call("list_dir"), say("done")]), "go", "stream-json")
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    types = [l["type"] for l in lines]
    assert types[0] == "turn_started" and "tool_started" in types and types[-1] == "result"
    tool = next(l for l in lines if l["type"] == "tool_started")
    assert tool["call"]["name"] == "list_dir"


def test_ask_means_deny_without_a_human(tmp_path, capsys):
    code = run_headless(agent_for(tmp_path, [call("write_file", path="x.txt", content="hi"), say("couldn't write")]),
                        "write x.txt", "json")
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["tool_calls"][0]["is_error"] and not (tmp_path / "x.txt").exists()


def test_failures_exit_nonzero(tmp_path, capsys):
    def broken(_):
        raise ModelError("server down")

    assert run_headless(agent_for(tmp_path, [broken]), "go", "json") == 1
    assert json.loads(capsys.readouterr().out)["finish_reason"] == "error"
    agent = agent_for(tmp_path, [call("list_dir", call_id=f"c{i}") for i in range(30)])
    agent.max_steps = 3
    assert run_headless(agent, "loop", "json") == 1


def test_event_to_dict():
    d = event_to_dict(ToolStarted(ToolCall("c1", "grep", {"pattern": "x"})))
    assert d == {"type": "tool_started", "call": {"id": "c1", "name": "grep", "arguments": {"pattern": "x"}}}
