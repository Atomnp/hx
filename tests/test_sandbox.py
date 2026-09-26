"""The OS sandbox, end to end through the bash tool. macOS + built hx-exec only."""

import sys
import uuid
from pathlib import Path

import pytest

from hx.messages import ToolCall
from hx.permissions import ALLOW, ASK, PermissionPolicy
from hx.sandbox import SandboxConfig
from hx.tools import ToolContext
from hx.tools import shell as shell_mod
from hx.tools.builtin import default_tools

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not shell_mod.exec_binary(), reason="needs macOS and a built hx-exec"
)


def bash(ctx, command, **kw):
    return default_tools().execute(ToolCall("c1", "bash", {"command": command, **kw}), ctx)


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ToolContext(cwd=ws, sandbox=SandboxConfig())


def test_can_write_inside_the_workspace(ctx):
    r = bash(ctx, "echo hi > a.txt && cat a.txt")
    assert not r.is_error and "hi" in r.content
    assert (ctx.cwd / "a.txt").read_text() == "hi\n"


def test_cannot_write_outside(ctx):
    # Not tmp_path's parent: the system temp dir is deliberately writable. Try the home folder.
    outside = Path.home() / f"hx_sandbox_test_{uuid.uuid4().hex}.txt"
    try:
        r = bash(ctx, f"echo x > {outside}")
        assert r.is_error and "Operation not permitted" in r.content
        assert not outside.exists()
        assert "sandbox=false" in r.content  # the model is told how to escalate
    finally:
        outside.unlink(missing_ok=True)


def test_no_network(ctx):
    r = bash(ctx, "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\"")
    assert r.is_error and "Operation not permitted" in r.content


def test_secret_dirs_unreadable(ctx, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("SECRET")
    monkeypatch.setattr(Path, "home", lambda: home)  # SandboxConfig reads Path.home() for the deny list
    r = bash(ctx, f"cat {home}/.ssh/id_rsa")
    assert "SECRET" not in r.content and "Operation not permitted" in r.content


def test_opting_out_runs_unsandboxed(ctx):
    outside = ctx.cwd.parent / "outside.txt"
    r = bash(ctx, f"echo x > {outside}", sandbox=False)
    assert not r.is_error and outside.exists()


def test_sandbox_unlocks_autonomy_in_auto_edit(ctx):
    policy = PermissionPolicy(mode="auto-edit")
    tool = default_tools().get("bash")
    assert policy.check(tool, {"command": "python3 -m pytest"}, ctx).action == ALLOW  # confined, so no prompt
    assert policy.check(tool, {"command": "pip install x", "sandbox": False}, ctx).action == ASK
    no_sandbox = ToolContext(cwd=ctx.cwd)
    assert policy.check(tool, {"command": "python3 -m pytest"}, no_sandbox).action == ASK
