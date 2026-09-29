"""The default tool set."""

from hx.tools.fs_edit import EditFile, WriteFile
from hx.tools.fs_read import ListDir, ReadFile
from hx.tools.registry import ToolRegistry
from hx.tools.search import Glob, Grep
from hx.tools.shell import Bash
from hx.tools.todo import TodoWrite
from hx.memory import Remember


def default_tools() -> ToolRegistry:
    return ToolRegistry([ReadFile(), ListDir(), Grep(), Glob(), EditFile(), WriteFile(), Bash(), TodoWrite(), Remember()])
