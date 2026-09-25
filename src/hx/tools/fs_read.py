"""read_file and list_dir."""

import difflib
from pathlib import Path

from hx.tools.base import Tool, ToolContext, ToolResult

MAX_LINES = 2000  # default lines per read_file call
MAX_LINE_CHARS = 2000  # minified JS and data files have enormous lines
MAX_ENTRIES = 400  # list_dir output cap
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "build", "dist"}


def resolve(ctx: ToolContext, path: str) -> Path:
    """Relative paths are relative to the workspace, like a shell started there."""
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ctx.cwd / p).resolve()


def not_found(path: Path) -> ToolResult:
    """A missing file usually means a typo or a wrong guess at the layout, so offer close matches."""
    parent = path.parent
    hint = ""
    if parent.is_dir():
        close = difflib.get_close_matches(path.name, [c.name for c in parent.iterdir()], n=3)
        if close:
            hint = f" Did you mean: {', '.join(str(parent / c) for c in close)}?"
    else:
        hint = f" (directory {parent} doesn't exist either)"
    return ToolResult(f"Error: {path} does not exist.{hint}", True)


def is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file. Returns lines prefixed with their line numbers (1-based). "
        f"Reads up to {MAX_LINES} lines; for longer files use offset and limit to read the rest. "
        "Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, absolute or relative to the workspace"},
            "offset": {"type": "integer", "description": "First line to read, 1-based (default 1)"},
            "limit": {"type": "integer", "description": f"Max lines to read (default {MAX_LINES})"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    read_only = True

    def run(self, args, ctx):
        path = resolve(ctx, args["path"])
        if not path.exists():
            return not_found(path)
        if path.is_dir():
            return ToolResult(f"Error: {path} is a directory. Use list_dir to see its contents.", True)
        if is_binary(path):
            return ToolResult(f"Error: {path} looks like a binary file ({path.stat().st_size} bytes); not shown.", True)

        offset = max(1, args.get("offset", 1))
        limit = max(1, args.get("limit", MAX_LINES))
        lines = path.read_text(errors="replace").splitlines()
        if not lines:
            return ToolResult(f"({path} is empty)")
        if offset > len(lines):
            return ToolResult(f"Error: offset {offset} is past the end of the file ({len(lines)} lines).", True)

        chunk = lines[offset - 1 : offset - 1 + limit]
        out = []
        for n, line in enumerate(chunk, start=offset):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"... [line truncated, {len(line)} chars]"
            out.append(f"{n:6}\t{line}")
        end = offset + len(chunk) - 1
        if end < len(lines):
            out.append(f"... ({len(lines) - end} more lines; call read_file with offset={end + 1} to continue)")
        return ToolResult("\n".join(out))


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List a directory as an indented tree (directories end with /, files show their size). "
        "Skips .git, node_modules, virtualenvs and caches. Use it to learn a project's layout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: the workspace)"},
            "depth": {"type": "integer", "description": "How many levels deep to go (default 2)"},
        },
        "additionalProperties": False,
    }
    read_only = True

    def run(self, args, ctx):
        root = resolve(ctx, args.get("path", "."))
        if not root.exists():
            return not_found(root)
        if not root.is_dir():
            return ToolResult(f"Error: {root} is a file. Use read_file to read it.", True)

        depth = max(1, args.get("depth", 2))
        lines = [f"{root}/"]
        truncated = False

        def walk(d: Path, level: int):
            nonlocal truncated
            try:
                children = sorted(d.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))
            except PermissionError:
                lines.append("  " * level + "[permission denied]")
                return
            for c in children:
                if len(lines) >= MAX_ENTRIES:
                    truncated = True
                    return
                if c.is_dir():
                    if c.name in IGNORED_DIRS:
                        continue
                    lines.append("  " * level + c.name + "/")
                    if level < depth:
                        walk(c, level + 1)
                else:
                    lines.append("  " * level + f"{c.name} ({c.stat().st_size} B)")

        walk(root, 1)
        if truncated:
            lines.append(f"... (stopped at {MAX_ENTRIES} entries; list a subdirectory or lower depth)")
        return ToolResult("\n".join(lines))
