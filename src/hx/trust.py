"""One-time trust prompts for project-level hooks and MCP servers, pinned to the config's content."""

import hashlib
import json
from pathlib import Path

TRUST_FILE = Path.home() / ".hx" / "trusted.json"


def fingerprint(cwd: Path, kind: str, config: dict) -> str:
    return hashlib.sha256(f"{cwd.resolve()}|{kind}|{json.dumps(config, sort_keys=True)}".encode()).hexdigest()


def is_trusted(cwd: Path, kind: str, config: dict, trust_file: Path | None = None) -> bool:
    f = trust_file or TRUST_FILE
    return f.is_file() and fingerprint(cwd, kind, config) in json.loads(f.read_text())


def trust(cwd: Path, kind: str, config: dict, trust_file: Path | None = None) -> None:
    f = trust_file or TRUST_FILE
    trusted = json.loads(f.read_text()) if f.is_file() else []
    if (fp := fingerprint(cwd, kind, config)) not in trusted:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(trusted + [fp]))


def gate(cwd: Path, kind: str, config: dict, confirm) -> bool:
    """True if this project config may run: already trusted, or the user confirms now."""
    if not config:
        return False
    if is_trusted(cwd, kind, config) or (confirm is not None and confirm(kind, config)):
        trust(cwd, kind, config)
        return True
    return False
