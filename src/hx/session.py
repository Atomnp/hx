"""Sessions: each conversation is an append-only JSONL file under ~/.hx/sessions/, so it can be resumed."""

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hx.messages import Message, from_dict, to_dict

DEFAULT_ROOT = Path.home() / ".hx" / "sessions"


def workspace_dir(workspace: Path, root: Path) -> Path:
    return root / hashlib.sha1(str(workspace.resolve()).encode()).hexdigest()[:16]


@dataclass
class SessionInfo:
    id: str
    path: Path
    created: str
    title: str  # the first user message
    messages: int


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.stem

    @classmethod
    def create(cls, workspace: Path, model: str, root: Path = DEFAULT_ROOT) -> "Session":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = workspace_dir(workspace, root) / f"{stamp}-{secrets.token_hex(3)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        s = cls(path)
        s.write({"type": "meta", "id": s.id, "workspace": str(workspace.resolve()), "model": model,
                 "created": datetime.now().isoformat(timespec="seconds")})
        return s

    def write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    # -- what the agent records

    def message(self, m: Message) -> None:
        self.write({"type": "message", "message": to_dict(m)})

    def turn(self, sha: str | None, length: int | None) -> None:
        self.write({"type": "turn", "sha": sha, "length": length})

    def reset(self, messages: list[Message], turns: list[tuple[str | None, int | None]]) -> None:
        self.write({"type": "reset", "messages": [to_dict(m) for m in messages], "turns": [list(t) for t in turns]})

    # -- reading back

    def replay(self) -> tuple[list[Message], list[tuple[str | None, int | None]]]:
        messages: list[Message] = []
        turns: list[tuple[str | None, int | None]] = []
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a crash: skip it
            kind = rec.get("type")
            if kind == "message":
                messages.append(from_dict(rec["message"]))
            elif kind == "turn":
                turns.append((rec["sha"], rec["length"]))
            elif kind == "reset":
                messages = [from_dict(d) for d in rec["messages"]]
                turns = [tuple(t) for t in rec["turns"]]
        return messages, turns


def list_sessions(workspace: Path, root: Path = DEFAULT_ROOT) -> list[SessionInfo]:
    """Newest first."""
    out = []
    for path in sorted(workspace_dir(workspace, root).glob("*.jsonl"), reverse=True):
        created, title, count = "", "", 0
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "meta":
                created = rec.get("created", "")
            elif rec.get("type") == "message":
                count += 1
                m = rec["message"]
                if not title and m.get("role") == "user":
                    title = m.get("content", "")[:70]
        out.append(SessionInfo(path.stem, path, created, title, count))
    return out


def find_session(workspace: Path, session_id: str | None, root: Path = DEFAULT_ROOT) -> Session | None:
    """The given session (by id or unique prefix), or the latest one when session_id is None."""
    sessions = list_sessions(workspace, root)
    if session_id is None:
        return Session(sessions[0].path) if sessions else None
    matches = [s for s in sessions if s.id.startswith(session_id)]
    return Session(matches[0].path) if len(matches) == 1 else None
