from hx.messages import ToolCall
from hx.tools import ToolContext
from hx.tools.builtin import default_tools


def run(tmp_path, name, **args):
    return default_tools().execute(ToolCall("c1", name, args), ToolContext(cwd=tmp_path))


def test_read_file_numbers_lines(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    r = run(tmp_path, "read_file", path="a.py")
    assert not r.is_error
    assert r.content.splitlines() == ["     1\tx = 1", "     2\ty = 2"]


def test_read_file_offset_limit_and_continuation_hint(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    r = run(tmp_path, "read_file", path="big.txt", offset=3, limit=2)
    assert "line 3" in r.content and "line 4" in r.content and "line 5" not in r.content
    assert "offset=5" in r.content  # tells the model how to continue


def test_read_file_suggests_close_names(tmp_path):
    (tmp_path / "config.py").write_text("")
    r = run(tmp_path, "read_file", path="confg.py")
    assert r.is_error and "Did you mean" in r.content and "config.py" in r.content


def test_read_file_refuses_binary_and_dirs(tmp_path):
    (tmp_path / "img.bin").write_bytes(b"\x89PNG\0\0data")
    (tmp_path / "sub").mkdir()
    assert "binary" in run(tmp_path, "read_file", path="img.bin").content
    assert "list_dir" in run(tmp_path, "read_file", path="sub").content


def test_list_dir_tree_skips_noise(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)")
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    r = run(tmp_path, "list_dir")
    assert "src/" in r.content and "main.py (8 B)" in r.content
    assert ".git" not in r.content and "node_modules" not in r.content
