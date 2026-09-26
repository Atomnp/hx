"""grep/glob tools. Each test runs twice: with the C++ binary (if built) and with the Python fallback."""

import pytest

from hx.messages import ToolCall
from hx.tools import ToolContext
from hx.tools import search as search_mod
from hx.tools.builtin import default_tools


@pytest.fixture(params=["native", "python"])
def backend(request, monkeypatch):
    if request.param == "native":
        if not search_mod.search_binary():
            pytest.skip("hx-search not built (run scripts/build_cpp.sh)")
    else:
        monkeypatch.setattr(search_mod, "search_binary", lambda: None)
    return request.param


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return helper()\n\ndef helper():\n    return 42\n")
    (tmp_path / "src" / "util.ts").write_text("export function helper() {}\n")
    (tmp_path / "README.md").write_text("Call main() to start.\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("function helper() {}\n")
    return tmp_path


def run(cwd, name, **args):
    return default_tools().execute(ToolCall("c1", name, args), ToolContext(cwd=cwd))


def test_grep_finds_lines_with_paths(project, backend):
    r = run(project, "grep", pattern=r"def \w+")
    assert r.content.splitlines() == ["src/app.py:1: def main():", "src/app.py:4: def helper():"]


def test_grep_glob_and_case(project, backend):
    r = run(project, "grep", pattern="HELPER", glob="*.ts", ignore_case=True)
    assert r.content == "src/util.ts:1: export function helper() {}"


def test_grep_skips_node_modules(project, backend):
    assert "node_modules" not in run(project, "grep", pattern="helper").content


def test_grep_in_subdir_keeps_workspace_relative_paths(project, backend):
    r = run(project, "grep", pattern="return 42", path="src")
    assert r.content == "src/app.py:5:     return 42"


def test_grep_truncates(project, backend):
    r = run(project, "grep", pattern="helper", max_results=1)
    assert len(r.content.splitlines()) == 2 and "stopped at 1" in r.content


def test_grep_no_match_and_bad_regex(project, backend):
    assert "No matches" in run(project, "grep", pattern="nothing_here").content
    assert run(project, "grep", pattern="(").is_error


def test_glob(project, backend):
    assert run(project, "glob", pattern="*.py").content == "src/app.py"
    assert run(project, "glob", pattern="src/*").content.splitlines() == ["src/app.py", "src/util.ts"]


def test_native_honors_gitignore(project):
    if not search_mod.search_binary():
        pytest.skip("hx-search not built")
    (project / ".gitignore").write_text("*.ts\n")
    assert "util.ts" not in run(project, "grep", pattern="helper").content
