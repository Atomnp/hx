import json
import sys
from pathlib import Path

import pytest

import hx.trust
from hx.agent import Agent
from hx.mcp import MCPClient, MCPError, connect_all, load_servers
from hx.messages import ToolCall
from hx.models.fake import FakeModel, call, say
from hx.permissions import ALLOW, ASK, PermissionPolicy, Rule
from hx.tools import ToolContext, ToolRegistry

SERVER = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


@pytest.fixture
def client():
    c = MCPClient("notes", sys.executable, [SERVER])
    yield c
    c.close()


def test_handshake_and_listing(client):
    assert client.server_info["serverInfo"]["name"] == "hx-test-server"
    assert [t["name"] for t in client.list_tools()] == ["notes_add", "notes_list", "weather"]


def test_calls_and_errors(client):
    assert client.call_tool("notes_add", {"text": "buy milk"}) == ("Saved note #1.", False)
    assert client.call_tool("notes_list", {}) == ("1. buy milk", False)
    assert client.call_tool("weather", {"city": ""}) == ("city is empty", True)
    with pytest.raises(MCPError, match="unknown"):
        client.call_tool("nope", {})


def test_tools_join_the_registry_and_the_agent_loop(tmp_path):
    clients, tools, notes = connect_all(tmp_path, {"notes": {"command": sys.executable, "args": [SERVER]}})
    try:
        assert "3 tool(s)" in notes[0]
        registry = ToolRegistry(tools)
        assert "mcp__notes__weather" in registry.names()
        # validated like any tool: the schema comes from the server
        r = registry.execute(ToolCall("c", "mcp__notes__weather", {}), ToolContext())
        assert r.is_error and "city: required field missing" in r.content

        model = FakeModel([call("mcp__notes__weather", city="Lawrence"), say("It's rainy.")])
        Agent(model, tools=registry).run("weather in Lawrence?")
        assert "Lawrence: 21°C, light rain" in model.requests[1][-1].content
    finally:
        for c in clients:
            c.close()


def test_mcp_tools_ask_unless_allowed_by_glob_rule(tmp_path):
    clients, tools, _ = connect_all(tmp_path, {"notes": {"command": sys.executable, "args": [SERVER]}})
    try:
        weather = next(t for t in tools if t.name.endswith("weather"))
        ctx = ToolContext(cwd=tmp_path)
        assert PermissionPolicy("ask").check(weather, {"city": "x"}, ctx).action == ASK
        trusting = PermissionPolicy("ask", allow=[Rule.parse("mcp__notes__*")])
        assert trusting.check(weather, {"city": "x"}, ctx).action == ALLOW
    finally:
        for c in clients:
            c.close()


def test_broken_server_is_reported_not_fatal(tmp_path):
    clients, tools, notes = connect_all(tmp_path, {"ghost": {"command": "/nonexistent/server"}})
    assert clients == [] and tools == [] and "unavailable" in notes[0]


def test_project_servers_need_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(hx.trust, "TRUST_FILE", tmp_path / "trusted.json")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / ".hx").mkdir()
    (tmp_path / ".hx" / "settings.json").write_text(json.dumps({"mcpServers": {"x": {"command": "evil"}}}))
    servers, notes = load_servers(tmp_path, confirm=lambda kind, cfg: False)
    assert servers == {} and "untrusted" in notes[0]
    servers, _ = load_servers(tmp_path, confirm=lambda kind, cfg: kind == "mcpServers")
    assert "x" in servers
