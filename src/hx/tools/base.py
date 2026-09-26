"""Tool, ToolResult and ToolContext."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hx.permissions import Approver, PermissionPolicy
    from hx.sandbox import SandboxConfig


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    notes: list[str] = field(default_factory=list)  # repairs the harness applied to the call


@dataclass
class ToolContext:
    """What a tool may need to know about the session it runs in."""

    cwd: Path = field(default_factory=Path.cwd)  # the workspace root; relative paths resolve against it
    # Files the model has read this session -> their mtime when read. Edit tools require an entry (read
    # before you modify) and an unchanged mtime (nobody changed it since). See tools/fs_edit.py.
    read_files: dict[str, int] = field(default_factory=dict)
    # Permission checks. None = no checks (used by unit tests of individual tools).
    permissions: "PermissionPolicy | None" = None
    approve: "Approver | None" = None  # asks the user; None = nobody to ask, so "ask" becomes "deny"
    sandbox: "SandboxConfig | None" = None  # OS sandbox for shell commands; None = off


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    # True if the tool can't change anything. Permissions let read-only tools run without asking.
    read_only: bool = False
    # The argument permission rules match against: a path ("path") or a shell command ("command").
    subject_arg: str | None = None

    def subject(self, args: dict[str, Any]) -> str | None:
        """What this call acts on, for permission rules. Path-like tools default to the workspace."""
        if self.subject_arg is None:
            return None
        return args.get(self.subject_arg, ".")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """The definition sent to the model (OpenAI/Ollama "function" format)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
