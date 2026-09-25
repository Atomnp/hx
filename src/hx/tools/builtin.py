"""The default tool set."""

from hx.tools.fs_read import ListDir, ReadFile
from hx.tools.registry import ToolRegistry


def default_tools() -> ToolRegistry:
    return ToolRegistry([ReadFile(), ListDir()])
