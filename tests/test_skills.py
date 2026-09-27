from hx.messages import ToolCall
from hx.permissions import ALLOW, ASK, PermissionPolicy
from hx.prompt import build_system_prompt
from hx.skills import SkillTool, discover, parse
from hx.tools import ToolContext
from hx.tools.builtin import default_tools
from pathlib import Path


def make_skill(root, name, description="Does a thing.", body="Step 1. Do it.", extra=None):
    d = root / ".hx" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    for fname, text in (extra or {}).items():
        (d / fname).write_text(text)
    return d


def test_discover_and_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    make_skill(tmp_path, "changelog", "Write changelog entries in Keep a Changelog format.")
    bad = tmp_path / ".hx" / "skills" / "nodesc"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: nodesc\n---\nbody")
    skills = discover(tmp_path)
    assert [s.name for s in skills] == ["changelog"]  # no description: skipped
    assert parse(bad) is None


def test_prompt_has_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    make_skill(tmp_path, "changelog", "Write changelog entries.", body="SECRET DETAILED INSTRUCTIONS " * 50)
    prompt = build_system_prompt(ToolContext(cwd=tmp_path), discover(tmp_path))
    assert "- changelog: Write changelog entries." in prompt
    assert "SECRET DETAILED INSTRUCTIONS" not in prompt  # level 2 isn't loaded until asked for


def test_skill_tool_loads_body_and_lists_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    make_skill(tmp_path, "changelog", body="Use the template.", extra={"template.md": "## [Unreleased]"})
    tool = SkillTool(discover(tmp_path))
    r = tool.run({"name": "changelog"}, ToolContext(cwd=tmp_path))
    assert "# Skill: changelog" in r.content and "Use the template." in r.content
    assert "template.md" in r.content and "description:" not in r.content
    assert tool.run({"name": "nope"}, ToolContext(cwd=tmp_path)).is_error


def test_skill_files_outside_workspace_are_readable_without_asking(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    user_skill = make_skill(home.parent / "home_root", "x")  # stands in for ~/.hx/skills/x
    ws = tmp_path / "ws"
    ws.mkdir()
    policy = PermissionPolicy("ask")
    read = default_tools().get("read_file")
    plain = ToolContext(cwd=ws)
    with_roots = ToolContext(cwd=ws, read_roots=[user_skill])
    args = {"path": str(user_skill / "SKILL.md")}
    assert policy.check(read, args, plain).action == ASK
    assert policy.check(read, args, with_roots).action == ALLOW
