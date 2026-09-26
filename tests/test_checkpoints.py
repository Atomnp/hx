import subprocess

import pytest

from hx.agent import Agent
from hx.checkpoints import Checkpoints, checkpoints_for
from hx.models.fake import FakeModel, call, say
from hx.tools import ToolContext
from hx.tools.builtin import default_tools
from pathlib import Path


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "keep.py").write_text("original\n")
    return w


@pytest.fixture
def cps(ws, tmp_path):
    return Checkpoints(ws, root=tmp_path / "shadow")


def test_restore_reverts_edits_creations_and_deletions(ws, cps):
    sha = cps.snapshot("start")
    (ws / "keep.py").write_text("changed\n")
    (ws / "new.py").write_text("created\n")
    (ws / "sub").mkdir()
    (ws / "sub" / "deep.txt").write_text("x")
    summary = cps.restore(sha)
    assert (ws / "keep.py").read_text() == "original\n"
    assert not (ws / "new.py").exists() and not (ws / "sub" / "deep.txt").exists()
    assert "new.py" in summary and "keep.py" in summary


def test_restore_is_itself_undoable(ws, cps):
    first = cps.snapshot("start")
    (ws / "keep.py").write_text("v2\n")
    cps.restore(first)
    before_restore = next(c for c in cps.list() if c.label.startswith("before restoring"))
    cps.restore(before_restore.sha)
    assert (ws / "keep.py").read_text() == "v2\n"


def test_project_git_repo_is_untouched(ws, cps):
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    status_before = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True).stdout
    sha = cps.snapshot("start")
    (ws / "tmp.txt").write_text("x")
    cps.restore(sha)
    status_after = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True).stdout
    assert status_before == status_after  # the user's index/working tree view is unchanged
    assert not (ws / ".git" / "refs" / "heads" / "main").exists()  # we never committed into it


def test_ignored_dirs_survive_restore(ws, cps):
    sha = cps.snapshot("start")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "big.js").write_text("x")
    cps.restore(sha)
    assert (ws / "node_modules" / "big.js").exists()  # excluded: never snapshotted, never deleted


def test_agent_undo_restores_files_and_conversation(ws, cps):
    model = FakeModel([
        call("write_file", path="made_by_agent.py", content="x = 1\n"),
        call("bash", call_id="c2", command="echo hi > from_shell.txt"),
        say("done"),
    ])
    agent = Agent(model, tools=default_tools(), ctx=ToolContext(cwd=ws), checkpoints=cps)
    agent.run("make some files")
    assert (ws / "made_by_agent.py").exists() and (ws / "from_shell.txt").exists()
    assert len(agent.messages) > 2

    msg = agent.undo()
    assert not (ws / "made_by_agent.py").exists()
    assert not (ws / "from_shell.txt").exists()  # shell-made changes are covered too
    assert [m.role for m in agent.messages] == ["system"]
    assert "made_by_agent.py" in msg
    assert agent.undo() == "Nothing to undo."


def test_too_broad_workspaces_get_no_checkpoints():
    assert checkpoints_for(Path.home()) is None
