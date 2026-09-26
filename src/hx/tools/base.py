"""Tool, ToolResult and ToolContext."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolContext:
    """What a tool may need to know about the session it runs in."""

    cwd: Path = field(default_factory=Path.cwd)  # the workspace root; relative paths resolve against it
    # Files the model has read this session -> their mtime when read. Edit tools require an entry (read
    # before you modify) and an unchanged mtime (nobody changed it since). See tools/fs_edit.py.
    read_files: dict[str, int] = field(default_factory=dict)


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    # True if the tool can't change anything. Permissions let read-only tools run without asking.
    read_only: bool = False

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """The definition sent to the model (OpenAI/Ollama "function" format)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
