from hx.agent import Agent
from hx.messages import Message, ToolCall
from hx.models.fake import FakeModel, say
from hx.prompt import build_system_prompt
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def test_repair_switch(monkeypatch, tmp_path):
    text_call = Message("assistant", '{"name": "list_dir", "arguments": {}}')
    model = FakeModel([text_call, say("after")])
    agent = Agent(model, tools=default_tools(), ctx=ToolContext(cwd=tmp_path))
    monkeypatch.setenv("HX_ABLATE", "repair")
    assert agent.run("go") == '{"name": "list_dir", "arguments": {}}'  # not parsed: the turn just ends
    monkeypatch.delenv("HX_ABLATE")
    model = FakeModel([text_call, say("after")])
    assert Agent(model, tools=default_tools(), ctx=ToolContext(cwd=tmp_path)).run("go") == "after"


def test_prompt_and_diagnostics_switches(monkeypatch, tmp_path):
    monkeypatch.setenv("HX_ABLATE", "prompt,diagnostics")
    assert build_system_prompt(ToolContext(cwd=tmp_path)).startswith("You are hx, a helpful coding assistant")
    ctx = ToolContext(cwd=tmp_path)
    r = default_tools().execute(ToolCall("c", "write_file", {"path": "a.py", "content": "def f(:\n"}), ctx)
    assert "syntax error" not in r.content
