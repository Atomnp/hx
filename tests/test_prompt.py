import subprocess
from pathlib import Path

from hx.permissions import PermissionPolicy
from hx.prompt import MAX_FILE_CHARS, build_system_prompt, find_instruction_files
from hx.tools import ToolContext


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg" / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "AGENTS.md").write_text("Use uv for everything.")
    (repo / "pkg" / "CLAUDE.md").write_text("pkg rule from CLAUDE.md")
    (repo / "pkg" / "sub" / "AGENTS.md").write_text("Tests live in tests/.")
    (repo / "pkg" / "sub" / "CLAUDE.md").write_text("ignored: AGENTS.md wins in the same dir")
    return repo


def test_instruction_files_root_to_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    repo = make_repo(tmp_path)
    files = find_instruction_files(repo / "pkg" / "sub")
    assert [f.relative_to(repo).as_posix() for f in files] == ["AGENTS.md", "pkg/CLAUDE.md", "pkg/sub/AGENTS.md"]


def test_prompt_contains_environment_and_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    repo = make_repo(tmp_path)
    prompt = build_system_prompt(ToolContext(cwd=repo / "pkg" / "sub", permissions=PermissionPolicy("auto-edit")))
    assert "You are hx" in prompt
    assert f"Workspace: {repo / 'pkg' / 'sub'}" in prompt
    assert "Git: branch" in prompt and "Permission mode: auto-edit" in prompt
    assert prompt.index("Use uv for everything.") < prompt.index("Tests live in tests/.")  # closest file last
    assert "ignored: AGENTS.md wins" not in prompt


def test_user_level_instructions_come_first(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".hx").mkdir(parents=True)
    (home / ".hx" / "AGENTS.md").write_text("Always answer in British English.")
    monkeypatch.setattr(Path, "home", lambda: home)
    repo = make_repo(tmp_path)
    prompt = build_system_prompt(ToolContext(cwd=repo))
    assert prompt.index("British English") < prompt.index("Use uv for everything.")


def test_huge_instruction_file_is_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    d = tmp_path / "proj"
    d.mkdir()
    (d / ".git").mkdir()
    (d / "AGENTS.md").write_text("x" * (MAX_FILE_CHARS + 500))
    assert "[truncated; read" in build_system_prompt(ToolContext(cwd=d))


def test_no_git_no_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    d = tmp_path / "plain"
    d.mkdir()
    prompt = build_system_prompt(ToolContext(cwd=d))
    assert "not a git repository" in prompt and "# Project instructions" not in prompt
