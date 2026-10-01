import os

import pytest

from hx.messages import ToolCall
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


@pytest.fixture
def env(tmp_path):
    registry, ctx = default_tools(), ToolContext(cwd=tmp_path)

    def run(name, **args):
        return registry.execute(ToolCall("c1", name, args), ctx)

    return tmp_path, run


def test_write_creates_file_and_dirs(env):
    root, run = env
    r = run("write_file", path="pkg/new.py", content="x = 1\n")
    assert not r.is_error and "Created" in r.content
    assert (root / "pkg" / "new.py").read_text() == "x = 1\n"


def test_overwrite_requires_read_first(env):
    root, run = env
    (root / "a.py").write_text("old\n")
    assert "read" in run("write_file", path="a.py", content="new\n").content.lower()
    run("read_file", path="a.py")
    r = run("write_file", path="a.py", content="new\n")
    assert not r.is_error and "-old" in r.content and "+new" in r.content


def test_edit_replaces_unique_text_and_shows_diff(env):
    root, run = env
    (root / "a.py").write_text("def f():\n    return 1\n")
    run("read_file", path="a.py")
    r = run("edit_file", path="a.py", old_string="return 1", new_string="return 2")
    assert not r.is_error and "-    return 1" in r.content and "+    return 2" in r.content
    assert (root / "a.py").read_text() == "def f():\n    return 2\n"


def test_edit_refuses_ambiguous_match_and_lists_lines(env):
    root, run = env
    (root / "a.py").write_text("x = 0\ny = 1\nx = 0\n")
    run("read_file", path="a.py")
    r = run("edit_file", path="a.py", old_string="x = 0", new_string="x = 9")
    assert r.is_error and "2 times (lines 1, 3)" in r.content
    r = run("edit_file", path="a.py", old_string="x = 0", new_string="x = 9", replace_all=True)
    assert not r.is_error and (root / "a.py").read_text() == "x = 9\ny = 1\nx = 9\n"


def test_edit_not_found_shows_closest_text(env):
    root, run = env
    (root / "a.py").write_text("def greet(name):\n    print('hi', name)\n")
    run("read_file", path="a.py")
    r = run("edit_file", path="a.py", old_string="def greet(person):", new_string="def greet(who):")
    assert r.is_error and "closest text is near line 1" in r.content and "def greet(name):" in r.content


def test_edit_tolerates_trailing_whitespace_without_touching_the_rest(env):
    root, run = env
    (root / "a.py").write_text("keep = 1   \nvalue = 1  \nother = 2\n")
    run("read_file", path="a.py")
    r = run("edit_file", path="a.py", old_string="value = 1", new_string="value = 5")
    assert not r.is_error and "ignoring trailing whitespace" not in r.content  # exact substring match works here
    r = run("edit_file", path="a.py", old_string="value = 5  \nother = 2   ", new_string="value = 6\nother = 3")
    assert not r.is_error and "ignoring trailing whitespace" in r.content
    assert (root / "a.py").read_text() == "keep = 1   \nvalue = 6\nother = 3\n"  # line 1's whitespace untouched


def test_edit_refuses_stale_file(env):
    root, run = env
    path = root / "a.py"
    path.write_text("v = 1\n")
    run("read_file", path="a.py")
    path.write_text("v = 1  # user edit\n")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # make sure mtime moves
    r = run("edit_file", path="a.py", old_string="v = 1", new_string="v = 2")
    assert r.is_error and "changed on disk" in r.content


def test_consecutive_edits_are_allowed(env):
    root, run = env
    (root / "a.py").write_text("a\nb\n")
    run("read_file", path="a.py")
    assert not run("edit_file", path="a.py", old_string="a", new_string="A").is_error
    assert not run("edit_file", path="a.py", old_string="b", new_string="B").is_error  # our own write refreshed mtime


def test_edit_input_errors(env):
    root, run = env
    (root / "a.py").write_text("a\n")
    run("read_file", path="a.py")
    assert "identical" in run("edit_file", path="a.py", old_string="a", new_string="a").content
    assert "write_file" in run("edit_file", path="missing.py", old_string="a", new_string="b").content


def test_edit_reports_syntax_errors_immediately(env):
    root, run = env
    (root / "a.py").write_text("def f():\n    return 1\n")
    run("read_file", path="a.py")
    r = run("edit_file", path="a.py", old_string="    return 1", new_string="    return (1")
    assert "syntax error" in r.content and "line 2" in r.content
    r = run("edit_file", path="a.py", old_string="    return (1", new_string="    return 1")
    assert "syntax error" not in r.content


def test_write_reports_bad_json(env):
    root, run = env
    assert "syntax error" in run("write_file", path="c.json", content='{"a": 1,}').content
    assert "syntax error" not in run("write_file", path="d.json", content='{"a": 1}').content


MODULE = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(6))  # 18 lines


def test_whole_file_edit_works_but_gets_a_hint(env):
    root, run = env
    (root / "m.py").write_text(MODULE)
    run("read_file", path="m.py")
    r = run("edit_file", path="m.py", old_string=MODULE, new_string=MODULE.replace("x + 5", "x + 50"))
    assert not r.is_error and "+    return x + 50" in r.content
    assert "[harness: old_string was 18 of the file's 18 lines" in r.content and "write_file" in r.content


def test_focused_edit_gets_no_hint(env):
    root, run = env
    (root / "m.py").write_text(MODULE)
    run("read_file", path="m.py")
    r = run("edit_file", path="m.py", old_string="    return x + 5\n", new_string="    return x + 50\n")
    assert not r.is_error and "[harness:" not in r.content


def test_small_files_get_no_hint(env):
    root, run = env
    (root / "a.py").write_text("X = 1\n")
    run("read_file", path="a.py")
    assert "[harness:" not in run("edit_file", path="a.py", old_string="X = 1\n", new_string="X = 2\n").content
