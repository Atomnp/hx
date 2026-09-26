"""bash tool. Most tests run with both the hx-exec runner and the Python fallback."""

import os
import time

import pytest

from hx.messages import ToolCall
from hx.tools import ToolContext
from hx.tools import shell as shell_mod
from hx.tools.builtin import default_tools


@pytest.fixture(params=["native", "python"])
def backend(request, monkeypatch):
    if request.param == "native":
        if not shell_mod.exec_binary():
            pytest.skip("hx-exec not built (run scripts/build_cpp.sh)")
    else:
        monkeypatch.setattr(shell_mod, "exec_binary", lambda: None)
    return request.param


def bash(cwd, command, **kw):
    return default_tools().execute(ToolCall("c1", "bash", {"command": command, **kw}), ToolContext(cwd=cwd))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_runs_in_workspace(tmp_path, backend):
    r = bash(tmp_path, "pwd && echo hi")
    assert not r.is_error
    assert tmp_path.name in r.content  # pwd printed the workspace
    assert "hi" in r.content and "[exit code 0]" in r.content


def test_nonzero_exit_and_stderr(tmp_path, backend):
    r = bash(tmp_path, "echo oops >&2; exit 3")
    assert r.is_error and "oops" in r.content and "exit code 3" in r.content


def test_stdin_is_closed(tmp_path, backend):
    start = time.monotonic()
    r = bash(tmp_path, "read line; echo got:$line", timeout=10)
    assert time.monotonic() - start < 5  # didn't wait for input
    assert "got:" in r.content


def test_timeout_kills_the_whole_group(tmp_path, backend):
    r = bash(tmp_path, "sleep 30 & echo $! > child.pid; sleep 30", timeout=1)
    assert r.is_error and "timed out after 1s" in r.content
    child = int((tmp_path / "child.pid").read_text())
    time.sleep(0.2)
    assert not alive(child)


def test_output_is_truncated_head_and_tail(tmp_path, backend):
    r = bash(tmp_path, "seq 1 200000")
    assert "bytes of output were cut" in r.content
    assert r.content.startswith("1\n2\n3")
    assert "200000" in r.content  # the tail survived


def test_native_returns_promptly_and_cleans_up_background_processes(tmp_path):
    if not shell_mod.exec_binary():
        pytest.skip("hx-exec not built")
    start = time.monotonic()
    r = bash(tmp_path, "(sleep 30 & echo $! > bg.pid); echo done", timeout=20)
    assert time.monotonic() - start < 3  # didn't wait for the background sleep's open pipe
    assert "done" in r.content
    time.sleep(0.2)
    assert not alive(int((tmp_path / "bg.pid").read_text()))


def test_quiet_environment(tmp_path, backend):
    r = bash(tmp_path, "echo $PAGER $GIT_PAGER $TERM")
    assert "cat cat dumb" in r.content
