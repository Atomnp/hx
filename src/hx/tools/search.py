"""grep and glob. Uses the C++ hx-search binary when it's built, otherwise a Python fallback."""

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

from hx.native import find_binary
from hx.tools.base import Tool, ToolContext, ToolResult
from hx.tools.fs_read import IGNORED_DIRS

DEFAULT_MAX = 100


def search_binary() -> str | None:
    return find_binary("hx-search", "search")


def run_native(binary: str, args: list[str], cwd: Path) -> tuple[list[dict], dict]:
    """Run hx-search with --json; return (records, summary)."""
    proc = subprocess.run([binary, *args, "--json"], cwd=cwd, capture_output=True, text=True, errors="replace", timeout=120)
    if proc.returncode == 2:
        raise ValueError(proc.stderr.strip().splitlines()[0] if proc.stderr.strip() else "hx-search failed")
    records, summary = [], {}
    for line in proc.stdout.splitlines():
        obj = json.loads(line)
        if "summary" in obj:
            summary = obj["summary"]
        else:
            records.append(obj)
    return records, summary


# ---------------------------------------------------------------- pure-Python fallback

def walk_files(root: Path, glob: str | None):
    """Yield (display_path, full_path). Skips the usual noise dirs; doesn't read .gitignore (the C++ version does)."""
    if root.is_file():
        yield str(root), root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if glob and not fnmatch.fnmatch(rel if "/" in glob else name, glob):
                continue
            yield rel, full


def python_grep(root: Path, pattern: str, glob: str | None, ignore_case: bool, max_results: int):
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    records = []
    for rel, full in walk_files(root, glob):
        try:
            data = full.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for n, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if regex.search(line):
                records.append({"path": rel, "line": n, "text": line[:300]})
                if len(records) >= max_results:
                    return records, {"truncated": True}
    return records, {"truncated": False}


# ---------------------------------------------------------------- tools

def display_root(ctx: ToolContext, path: str) -> tuple[Path, str]:
    """Return (absolute root, the root argument to pass so printed paths are workspace-relative)."""
    full = (ctx.cwd / path).resolve() if not Path(path).is_absolute() else Path(path)
    return full, path


class Grep(Tool):
    subject_arg = "path"
    name = "grep"
    description = (
        "Search file contents with a regular expression (ECMAScript/Python-style syntax). "
        "Returns matching lines as path:line: text. Respects .gitignore. "
        "Use glob to narrow by file name (e.g. '*.py' or 'src/**/*.ts'). "
        "Prefer this over reading many files to find something."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for, e.g. 'def \\w+_tool' or 'TODO'"},
            "path": {"type": "string", "description": "File or directory to search. Omit to search the whole workspace."},
            "glob": {"type": "string", "description": "Only search files matching this glob"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)"},
            "max_results": {"type": "integer", "description": f"Max matching lines (default {DEFAULT_MAX})"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    def run(self, args, ctx):
        path = args.get("path", ".")
        root, root_arg = display_root(ctx, path)
        if not root.exists():
            return ToolResult(f"Error: {path} does not exist.", True)
        max_results = args.get("max_results", DEFAULT_MAX)
        binary = search_binary()
        try:
            if binary:
                cmd = ["grep", args["pattern"], root_arg, "--max", str(max_results)]
                if args.get("glob"):
                    cmd += ["--glob", args["glob"]]
                if args.get("ignore_case"):
                    cmd.append("-i")
                records, summary = run_native(binary, cmd, ctx.cwd)
            else:
                records, summary = python_grep(root, args["pattern"], args.get("glob"), args.get("ignore_case", False), max_results)
                if path != ".":
                    for r in records:
                        r["path"] = r["path"] if r["path"].startswith(path) else f"{path.rstrip('/')}/{r['path']}"
        except (ValueError, re.error) as e:
            return ToolResult(f"Error: bad pattern: {e}", True)

        if not records:
            return ToolResult(f"No matches for {args['pattern']!r}.")
        out = [f"{r['path']}:{r['line']}: {r['text']}" for r in records]
        if summary.get("truncated"):
            out.append(f"... (stopped at {max_results} matches; narrow the pattern, path or glob)")
        return ToolResult("\n".join(out))


class Glob(Tool):
    subject_arg = "path"
    name = "glob"
    description = (
        "Find files by name pattern, e.g. '**/*.py', 'tests/test_*.py', '*.md'. "
        "A pattern without '/' matches file names at any depth. Respects .gitignore. Returns paths, sorted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search. Omit to search the whole workspace."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True
    MAX = 200

    def run(self, args, ctx):
        path = args.get("path", ".")
        root, root_arg = display_root(ctx, path)
        if not root.is_dir():
            return ToolResult(f"Error: {path} is not a directory.", True)
        binary = search_binary()
        if binary:
            records, summary = run_native(binary, ["files", root_arg, "--glob", args["pattern"], "--max", str(self.MAX)], ctx.cwd)
            paths = [r["path"] for r in records]
            total = summary.get("files", len(paths))
        else:
            all_paths = [rel if path == "." else f"{path.rstrip('/')}/{rel}" for rel, _ in walk_files(root, args["pattern"])]
            paths, total = all_paths[: self.MAX], len(all_paths)

        if not paths:
            return ToolResult(f"No files match {args['pattern']!r}.")
        if total > len(paths):
            paths.append(f"... ({total - len(paths)} more; use a narrower pattern)")
        return ToolResult("\n".join(paths))
