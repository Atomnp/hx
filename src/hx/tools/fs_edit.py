"""write_file and edit_file.

Both refuse to change a file the model hasn't read, or one that changed on disk since it was read.
"""

import difflib
from pathlib import Path

from hx.tools.base import Tool, ToolContext, ToolResult
from hx.tools.fs_read import resolve

MAX_DIFF_LINES = 60


def mark_read(ctx: ToolContext, path: Path) -> None:
    ctx.read_files[str(path)] = path.stat().st_mtime_ns


def check_fresh(ctx: ToolContext, path: Path) -> ToolResult | None:
    """None if the model may modify `path`; otherwise an error result explaining why not."""
    seen = ctx.read_files.get(str(path))
    if seen is None:
        return ToolResult(f"Error: read {path} with read_file before modifying it.", True)
    if path.stat().st_mtime_ns != seen:
        return ToolResult(f"Error: {path} changed on disk since you read it. Read it again, then retry.", True)
    return None


def diff(before: str, after: str, path: Path) -> str:
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{path.name}", f"b/{path.name}", lineterm="", n=2))
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... ({len(lines) - MAX_DIFF_LINES} more diff lines)"]
    return "\n".join(lines)


class WriteFile(Tool):
    subject_arg = "path"
    name = "write_file"
    description = (
        "Create a new file, or completely replace an existing one, with the given content. "
        "Parent directories are created. To change part of an existing file, use edit_file instead. "
        "An existing file must be read with read_file first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to write, absolute or relative to the workspace"},
            "content": {"type": "string", "description": "The complete new file content"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def run(self, args, ctx):
        path = resolve(ctx, args["path"])
        content = args["content"]
        if path.is_dir():
            return ToolResult(f"Error: {path} is a directory.", True)

        if path.exists():
            if err := check_fresh(ctx, path):
                return err
            before = path.read_text(errors="replace")
            path.write_text(content)
            mark_read(ctx, path)
            return ToolResult(f"Overwrote {path} ({len(content.splitlines())} lines).\n{diff(before, content, path)}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        mark_read(ctx, path)  # we know its exact content, so later edits are allowed
        return ToolResult(f"Created {path} ({len(content.splitlines())} lines).")


def normalize(text: str) -> str:
    """Trailing whitespace is invisible to models and often wrong in their old_string; ignore it for matching."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def find_ignoring_trailing_ws(text: str, old: str) -> tuple[int, int] | None:
    """Find `old` in `text` ignoring trailing whitespace on each line. Returns the (start, end) span in the
    ORIGINAL text, or None if there's no unique match. Only that span is replaced; the rest of the file,
    trailing whitespace included, stays byte-for-byte the same."""
    norm_text, norm_old = normalize(text), normalize(old)
    if not norm_old.strip() or norm_text.count(norm_old) != 1:
        return None
    line_starts, pos = [], 0
    for line in text.split("\n"):
        line_starts.append(pos)
        pos += len(line) + 1

    def to_original(idx: int) -> int:
        # Normalizing only removes characters at line ends, so (line, column) positions inside the
        # stripped content are identical in both texts.
        line = norm_text.count("\n", 0, idx)
        col = idx - (norm_text.rfind("\n", 0, idx) + 1)
        return line_starts[line] + col

    start = norm_text.find(norm_old)
    return to_original(start), to_original(start + len(norm_old))


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


class EditFile(Tool):
    subject_arg = "path"
    name = "edit_file"
    description = (
        "Replace an exact piece of text in a file. old_string must match the file exactly (including indentation) "
        "and be unique; include a few surrounding lines to make it unique, or set replace_all. "
        "Read the file first. Returns a diff of the change."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text (must differ from old_string)"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def run(self, args, ctx):
        path = resolve(ctx, args["path"])
        old, new = args["old_string"], args["new_string"]
        if not path.is_file():
            return ToolResult(f"Error: {path} is not a file. Use write_file to create it.", True)
        if old == new:
            return ToolResult("Error: old_string and new_string are identical; nothing to change.", True)
        if not old:
            return ToolResult("Error: old_string is empty. To create or replace a whole file, use write_file.", True)
        if err := check_fresh(ctx, path):
            return err

        text = path.read_text(errors="replace")
        count = text.count(old)

        if count == 0:
            # Fallback: match ignoring trailing whitespace. Only used when it's unambiguous.
            span = find_ignoring_trailing_ws(text, old)
            if span is None:
                return ToolResult(self.not_found_message(path, text, old), True)
            after = text[: span[0]] + new + text[span[1] :]
            path.write_text(after)
            mark_read(ctx, path)
            return ToolResult(
                f"Edited {path}: 1 replacement (matched after ignoring trailing whitespace).\n{diff(text, after, path)}"
            )

        if count > 1 and not args.get("replace_all"):
            lines = []
            start = 0
            while (i := text.find(old, start)) != -1:
                lines.append(str(line_of(text, i)))
                start = i + 1
            return ToolResult(
                f"Error: old_string appears {count} times (lines {', '.join(lines)}). "
                "Include more surrounding context to make it unique, or set replace_all=true.",
                True,
            )

        after = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        path.write_text(after)
        mark_read(ctx, path)
        return ToolResult(f"Edited {path}: {count} replacement(s).\n{diff(text, after, path)}")

    @staticmethod
    def not_found_message(path: Path, text: str, old: str) -> str:
        """Show the closest region of the file so the model can see what it got wrong."""
        file_lines = text.splitlines()
        first = old.strip().splitlines()[0].strip() if old.strip() else ""
        best = difflib.get_close_matches(first, [l.strip() for l in file_lines], n=1, cutoff=0.5)
        msg = f"Error: old_string not found in {path}."
        if best:
            idx = [l.strip() for l in file_lines].index(best[0])
            lo, hi = max(0, idx - 2), min(len(file_lines), idx + 3)
            snippet = "\n".join(f"{n + 1:6}\t{file_lines[n]}" for n in range(lo, hi))
            msg += f" The closest text is near line {idx + 1}:\n{snippet}\nCopy the exact text (indentation included)."
        else:
            msg += " Read the file again to get the exact current text."
        return msg
