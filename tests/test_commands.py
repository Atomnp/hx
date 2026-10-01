from pathlib import Path

import pytest

from hx.agent import Agent
from hx.commands import Commands, Prompt
from hx.memory import Remember, add_memory, prompt_section
from hx.models.fake import FakeModel, say
from hx.permissions import ASK, PermissionPolicy
from hx.prompt import build_system_prompt
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


class State:
    def __init__(self, agent, cwd):
        self.agent, self.cwd, self.exit = agent, cwd, False


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = Agent(FakeModel([say("hi")]), tools=default_tools(), ctx=ToolContext(cwd=ws, permissions=PermissionPolicy("ask")))
    state = State(agent, ws)
    return state, ws


def test_help_lists_builtins_and_custom(setup):
    state, ws = setup
    (ws / ".hx" / "commands").mkdir(parents=True)
    (ws / ".hx" / "commands" / "review.md").write_text("---\ndescription: Review the diff\n---\nReview git diff. Focus: $ARGUMENTS")
    out = Commands(state).handle("/help")
    assert "/undo" in out and "/mode" in out and "/review" in out and "Review the diff" in out


def test_custom_command_expands_arguments(setup):
    state, ws = setup
    (ws / ".hx" / "commands").mkdir(parents=True)
    (ws / ".hx" / "commands" / "review.md").write_text("---\ndescription: d\n---\nReview git diff. Focus: $ARGUMENTS")
    out = Commands(state).handle("/review error handling")
    assert out == Prompt("Review git diff. Focus: error handling")


def test_mode_clear_exit_unknown(setup):
    state, ws = setup
    c = Commands(state)
    assert "ask" in c.handle("/mode")
    assert "now auto-edit" in c.handle("/mode auto-edit") and state.agent.ctx.permissions.mode == "auto-edit"
    assert "Unknown mode" in c.handle("/mode yolo")
    state.agent.run("hello")
    assert len(state.agent.messages) == 3
    c.handle("/clear")
    assert [m.role for m in state.agent.messages] == ["system"]
    assert "Unknown command" in c.handle("/frobnicate")
    c.handle("/exit")
    assert state.exit


def test_hash_adds_project_memory_and_it_reaches_the_prompt(setup):
    state, ws = setup
    assert "Remembered" in Commands(state).handle("# tests need DB_URL=sqlite:// to run")
    text = (ws / ".hx" / "memory.md").read_text()
    assert "- tests need DB_URL=sqlite:// to run (" in text
    assert "tests need DB_URL" in build_system_prompt(ToolContext(cwd=ws))


def test_user_and_project_memory_sections(setup):
    state, ws = setup
    add_memory(ws, "I prefer short answers", scope="user")
    add_memory(ws, "API client is in src/clients", scope="project")
    section = prompt_section(ws)
    assert section.index("User memory") < section.index("Project memory")


def test_remember_tool_writes_and_needs_edit_permission(setup):
    state, ws = setup
    tool = Remember()
    ctx = ToolContext(cwd=ws)
    assert PermissionPolicy("ask").check(tool, {"fact": "x"}, ctx).action == ASK  # it writes a file
    assert "Remembered" in tool.run({"fact": "use uv run pytest"}, ctx).content
    assert "use uv run pytest" in (ws / ".hx" / "memory.md").read_text()


def test_verbose_command_toggles_and_shows_the_prompt(setup):
    from hx.ui import PlainUI

    state, ws = setup
    state.ui = PlainUI()
    c = Commands(state)
    assert c.handle("/verbose") == "Verbose output is on." and state.ui.verbose
    assert c.handle("/verbose") == "Verbose output is off." and not state.ui.verbose
    state.ui = None
    assert "isn't available" in c.handle("/verbose")
