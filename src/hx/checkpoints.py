"""Snapshots of the workspace in a separate git repo (~/.hx/checkpoints), so a turn can be undone.

Uses GIT_DIR/GIT_WORK_TREE, so the project's own .git is never touched.
"""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".hx" / "checkpoints"

# Never snapshot these, even without a .gitignore (the project's .gitignore files are honored too).
EXCLUDES = ["node_modules/", ".venv/", "venv/", "__pycache__/", "*.pyc", ".mypy_cache/", ".pytest_cache/",
            "build/", "dist/", ".DS_Store", ".hx/"]


@dataclass
class Checkpoint:
    sha: str
    label: str
    age: str


class CheckpointError(RuntimeError):
    pass


class Checkpoints:
    def __init__(self, workspace: Path, root: Path = DEFAULT_ROOT):
        self.workspace = workspace.resolve()
        key = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:16]
        self.git_dir = root / key
        self.ready = False

    def git(self, *args: str) -> str:
        env = {
            **os.environ,
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.workspace),
            "GIT_AUTHOR_NAME": "hx", "GIT_AUTHOR_EMAIL": "hx@localhost",
            "GIT_COMMITTER_NAME": "hx", "GIT_COMMITTER_EMAIL": "hx@localhost",
        }
        # No hooks, no signing, no pager: this repo is private plumbing, not the user's.
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-c", "core.pager=cat", *args]
        proc = subprocess.run(cmd, cwd=self.workspace, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CheckpointError(f"git {' '.join(args)}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def init(self) -> None:
        if self.ready:
            return
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.mkdir(parents=True, exist_ok=True)
            self.git("init", "--quiet")
            (self.git_dir / "info").mkdir(exist_ok=True)
            (self.git_dir / "info" / "exclude").write_text("\n".join(EXCLUDES) + "\n")
        self.ready = True

    def snapshot(self, label: str) -> str:
        """Record the current state of the workspace. Returns the snapshot id."""
        self.init()
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "--no-verify", "-m", label)
        return self.git("rev-parse", "HEAD")

    def changes_since(self, sha: str) -> str:
        """`git diff --stat` between a snapshot and the workspace as it is now."""
        self.git("add", "--all")
        return self.git("diff", "--cached", "--stat", sha)

    def restore(self, sha: str) -> str:
        """Make the workspace match snapshot `sha`. Returns a summary of what changed.

        The current state is snapshotted first, so a restore can itself be undone.
        `read-tree -u --reset` rewrites the working tree from the snapshot: changed files go back,
        files created since are deleted, deleted files come back. Ignored files are left alone."""
        summary = self.changes_since(sha)
        self.snapshot(f"before restoring {sha[:8]}")
        self.git("read-tree", "-u", "--reset", sha)
        self.git("commit", "--quiet", "--allow-empty", "--no-verify", "-m", f"restored {sha[:8]}")
        return summary

    def list(self, n: int = 20) -> list[Checkpoint]:
        self.init()
        try:
            out = self.git("log", f"-{n}", "--format=%H%x09%s%x09%cr")
        except CheckpointError:  # no commits yet
            return []
        return [Checkpoint(*line.split("\t", 2)) for line in out.splitlines() if line]


def checkpoints_for(workspace: Path) -> Checkpoints | None:
    """None for workspaces too broad to snapshot (your home folder, the filesystem root)."""
    ws = workspace.resolve()
    if ws in (Path.home().resolve(), Path("/")):
        return None
    return Checkpoints(ws)
