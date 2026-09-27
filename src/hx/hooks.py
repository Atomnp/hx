"""User-defined shell commands that run at lifecycle events.

Events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop. Each hook gets a JSON payload on stdin.
Exit 0 = ok, exit 2 = block / send feedback, anything else = hook error.

    {"hooks": {"PostToolUse": [{"matcher": "edit_file|write_file", "command": "ruff format --quiet $HX_FILE"}]}}
"""

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")
TRUST_FILE = Path.home() / ".hx" / "trusted_hooks.json"


@dataclass
class Hook:
    event: str
    command: str
    matcher: str | None = None
    timeout: int = 30

    def applies_to(self, tool_name: str | None) -> bool:
        return self.matcher is None or tool_name is None or re.fullmatch(self.matcher, tool_name) is not None


@dataclass
class HookOutcome:
    blocked: bool = False
    feedback: str = ""  # stderr of blocking hooks: why it was blocked / what to do
    output: str = ""  # stdout of successful hooks: extra context
    errors: list[str] = field(default_factory=list)  # hooks that failed (shown to the user, never block)


class Hooks:
    def __init__(self, hooks: list[Hook], cwd: Path):
        self.hooks = hooks
        self.cwd = cwd

    def __bool__(self) -> bool:
        return bool(self.hooks)

    def run(self, event: str, payload: dict, tool_name: str | None = None) -> HookOutcome:
        outcome = HookOutcome()
        data = json.dumps({"event": event, "workspace": str(self.cwd), **payload})
        env = {**os.environ, "HX_EVENT": event, "HX_WORKSPACE": str(self.cwd)}
        if tool_name:
            env["HX_TOOL"] = tool_name
        path = payload.get("arguments", {}).get("path") if isinstance(payload.get("arguments"), dict) else None
        if path:
            env["HX_FILE"] = str((self.cwd / path).resolve())

        for hook in self.hooks:
            if hook.event != event or not hook.applies_to(tool_name):
                continue
            try:
                r = subprocess.run(["/bin/bash", "-c", hook.command], input=data, cwd=self.cwd, env=env,
                                   capture_output=True, text=True, timeout=hook.timeout)
            except subprocess.TimeoutExpired:
                outcome.errors.append(f"{event} hook timed out after {hook.timeout}s: {hook.command}")
                continue
            if r.returncode == 2:
                outcome.blocked = True
                outcome.feedback += r.stderr.strip() + "\n"
            elif r.returncode == 0:
                outcome.output += r.stdout
            else:
                outcome.errors.append(f"{event} hook failed (exit {r.returncode}): {hook.command}: {r.stderr.strip()[:200]}")
        outcome.feedback = outcome.feedback.strip()
        outcome.output = outcome.output.strip()
        return outcome


def parse(config: dict) -> list[Hook]:
    hooks = []
    for event, entries in config.items():
        if event not in EVENTS:
            raise ValueError(f"unknown hook event {event!r}; expected one of {', '.join(EVENTS)}")
        for e in entries:
            hooks.append(Hook(event, e["command"], e.get("matcher"), e.get("timeout", 30)))
    return hooks


def load_hooks(cwd: Path, confirm=None) -> tuple[Hooks, list[str]]:
    """User hooks are always trusted (you wrote them). Project hooks come with the repo, perhaps from someone else,
    so they run only after you confirm them once; the confirmation is tied to their exact content."""
    hooks: list[Hook] = []
    notes: list[str] = []
    user_settings = Path.home() / ".hx" / "settings.json"
    if user_settings.is_file():
        hooks += parse(json.loads(user_settings.read_text()).get("hooks", {}))

    project_settings = cwd / ".hx" / "settings.json"
    if project_settings.is_file():
        config = json.loads(project_settings.read_text()).get("hooks", {})
        if config:
            project_hooks = parse(config)
            if is_trusted(cwd, config) or (confirm and confirm(project_hooks)):
                trust(cwd, config)
                hooks += project_hooks
            else:
                notes.append(f"ignored {len(project_hooks)} untrusted project hook(s) from {project_settings}")
    return Hooks(hooks, cwd), notes


def fingerprint(cwd: Path, config: dict) -> str:
    return hashlib.sha256((str(cwd.resolve()) + json.dumps(config, sort_keys=True)).encode()).hexdigest()


def is_trusted(cwd: Path, config: dict) -> bool:
    return TRUST_FILE.is_file() and fingerprint(cwd, config) in json.loads(TRUST_FILE.read_text())


def trust(cwd: Path, config: dict) -> None:
    trusted = json.loads(TRUST_FILE.read_text()) if TRUST_FILE.is_file() else []
    if (fp := fingerprint(cwd, config)) not in trusted:
        TRUST_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRUST_FILE.write_text(json.dumps(trusted + [fp]))
