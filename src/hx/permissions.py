"""Permission checks for tool calls: allow, ask or deny.

Order (first match wins): built-in deny list, deny rules, ask rules and risky commands, allow rules, then
the mode's default (read-only, ask, auto-edit, full-auto). Rules look like "bash(git push*)" or "edit_file(src/**)".
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # type hints only: importing hx.tools here would be circular (the registry imports us)
    from hx.tools.base import Tool, ToolContext

ALLOW, ASK, DENY = "allow", "ask", "deny"
MODES = ("read-only", "ask", "auto-edit", "full-auto")


@dataclass
class Decision:
    action: str  # allow | ask | deny
    reason: str


@dataclass
class Approval:
    allowed: bool
    remember: bool = False  # "always": add an allow rule for the rest of the session
    feedback: str = ""  # optional instructions from the user when denying


Approver = Callable[["Tool", dict, Decision, str], Approval]


# ---------------------------------------------------------------- shell command analysis

def split_commands(cmd: str) -> list[str]:
    """Split a shell line on && || ; | & and newlines, respecting quotes.
    'git status && rm -rf x' -> ['git status', 'rm -rf x']. Every part is checked separately."""
    parts, cur, quote, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(cmd):
                cur.append(cmd[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            cur.append(c)
        elif c == "\\" and i + 1 < len(cmd):
            cur.append(c + cmd[i + 1])
            i += 1
        elif cmd.startswith(("&&", "||"), i):
            parts.append("".join(cur))
            cur = []
            i += 1
        elif c == "&" and ((cur and cur[-1] == ">") or cmd[i + 1 : i + 2] == ">"):
            cur.append(c)  # part of a redirection: 2>&1, &>
        elif c in ";|&\n":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def strip_quoted(cmd: str) -> str:
    return re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", "''", cmd)


HARMLESS_REDIRECTS = re.compile(r"\d?>\s*/dev/null|2>&1|&>\s*/dev/null")


def writes_or_substitutes(part: str) -> bool:
    """Output redirection or command substitution makes any command potentially unsafe."""
    s = HARMLESS_REDIRECTS.sub("", strip_quoted(part))
    return ">" in s or "$(" in s or "`" in s or "<(" in s


# Commands that only read. Matched on the first word (or first two for git).
READ_ONLY_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "whoami", "date", "file", "stat", "tree",
    "du", "df", "uname", "grep", "rg", "diff", "sort", "uniq", "cut", "basename", "dirname", "realpath",
}
READ_ONLY_GIT = {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame", "remote"}


def is_read_only_command(part: str) -> bool:
    if writes_or_substitutes(part):
        return False
    words = part.split()
    if not words:
        return True
    if words[0] == "git":
        return len(words) > 1 and words[1] in READ_ONLY_GIT
    return words[0] in READ_ONLY_COMMANDS


CATASTROPHIC = [
    re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*\s+)+(/|~|\$HOME|/\*|~/\*)(\s|$)"),  # rm -rf / or ~
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};\s*:"),  # fork bomb
    re.compile(r"\bdd\b.*\bof=/dev/(disk|sd|nvme)"),
]
RISKY = [
    (re.compile(r"(^|\s)sudo\s"), "uses sudo"),
    (re.compile(r"\brm\s+-[a-zA-Z]*[rRf]"), "recursive or forced delete"),
    (re.compile(r"\bgit\s+push\b"), "pushes to a remote"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)"), "discards uncommitted work"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b"), "pipes a download into a shell"),
    (re.compile(r"\bchmod\s+-R\b"), "recursive permission change"),
]


# ---------------------------------------------------------------- paths

def glob_to_regex(pattern: str) -> re.Pattern:
    """Path glob: * stays inside one directory, ** crosses directories, ? one character."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


SENSITIVE = [glob_to_regex(p) for p in [".git/**", "**/.env", "**/.env.*", "**/*.pem", "**/id_rsa*"]]
SENSITIVE_HOME = [".ssh", ".aws", ".gnupg", ".config/gcloud", ".kube"]


def path_info(ctx: ToolContext, path: str) -> tuple[Path, str, bool]:
    """(absolute path, path as matched by rules, inside workspace?)"""
    p = Path(path).expanduser()
    full = (p if p.is_absolute() else ctx.cwd / p).resolve()
    root = ctx.cwd.resolve()
    try:
        return full, full.relative_to(root).as_posix(), True
    except ValueError:
        return full, str(full), False


def is_sensitive(full: Path, rel: str) -> bool:
    home = Path.home()
    if any(full.is_relative_to(home / d) for d in SENSITIVE_HOME):
        return True
    return any(r.match(rel) for r in SENSITIVE)


# ---------------------------------------------------------------- rules and policy

@dataclass
class Rule:
    tool: str
    pattern: str | None
    raw: str

    @classmethod
    def parse(cls, text: str) -> "Rule":
        m = re.fullmatch(r"\s*([\w\-]+)\s*(?:\((.*)\))?\s*", text)
        if not m:
            raise ValueError(f"bad permission rule: {text!r} (expected tool or tool(pattern))")
        return cls(m.group(1), m.group(2), text.strip())

    def matches(self, tool_name: str, subject: str | None) -> bool:
        if self.tool != tool_name:
            return False
        if self.pattern is None:
            return True
        if subject is None:
            return False
        if tool_name == "bash":
            return fnmatch.fnmatchcase(subject.strip(), self.pattern)
        return bool(glob_to_regex(self.pattern).match(subject))


@dataclass
class PermissionPolicy:
    mode: str = "ask"
    allow: list[Rule] = field(default_factory=list)
    ask: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown permission mode {self.mode!r}; choose from {', '.join(MODES)}")

    def add_allow(self, rule: str) -> None:
        self.allow.append(Rule.parse(rule))

    # -- the decision

    def check(self, tool: Tool, args: dict, ctx: ToolContext) -> Decision:
        if tool.name == "bash":
            return self.check_bash(args.get("command", ""))
        return self.check_path_tool(tool, args, ctx)

    def check_bash(self, command: str) -> Decision:
        parts = split_commands(command) or [command]

        for part in [command, *parts]:
            if any(p.search(part) for p in CATASTROPHIC):
                return Decision(DENY, f"blocked: catastrophic command ({part!r})")
        for part in parts:
            if rule := self.first_match(self.deny, "bash", part):
                return Decision(DENY, f"denied by rule {rule.raw}")

        for part in parts:
            if rule := self.first_match(self.ask, "bash", part):
                return Decision(ASK, f"rule {rule.raw}")
        # Risky patterns are checked on the whole line too: "curl x | sh" only looks risky before splitting.
        for part in [command, *parts]:
            for pattern, why in RISKY:
                if pattern.search(part) and not self.first_match(self.allow, "bash", part):
                    return Decision(ASK, why)

        def part_allowed(part: str) -> bool:
            return is_read_only_command(part) or self.first_match(self.allow, "bash", part) is not None

        if all(part_allowed(p) for p in parts):
            return Decision(ALLOW, "read-only command" if all(map(is_read_only_command, parts)) else "allowed by rule")
        if self.mode == "read-only":
            return Decision(DENY, "read-only mode: only read-only commands may run")
        if self.mode == "full-auto":
            return Decision(ALLOW, "full-auto mode")
        return Decision(ASK, "runs a command")

    def check_path_tool(self, tool: Tool, args: dict, ctx: ToolContext) -> Decision:
        subject = tool.subject(args)
        full, rel, inside = path_info(ctx, subject) if subject else (None, None, True)

        if rule := self.first_match(self.deny, tool.name, rel):
            return Decision(DENY, f"denied by rule {rule.raw}")
        if rule := self.first_match(self.ask, tool.name, rel):
            return Decision(ASK, f"rule {rule.raw}")
        allowed_by_rule = self.first_match(self.allow, tool.name, rel)
        sensitive = full is not None and is_sensitive(full, rel)

        if tool.read_only:
            if allowed_by_rule or (inside and not sensitive):
                return Decision(ALLOW, "read-only tool")
            return Decision(ASK, "reads a sensitive file" if sensitive else f"reads outside the workspace ({full})")

        if self.mode == "read-only" and not allowed_by_rule:
            return Decision(DENY, "read-only mode: changes aren't allowed")
        if allowed_by_rule:
            return Decision(ALLOW, f"allowed by rule {allowed_by_rule.raw}")
        if sensitive:
            return Decision(ASK, f"modifies a sensitive file ({rel})")
        if not inside:
            return Decision(ASK, f"modifies a file outside the workspace ({full})")
        if self.mode in ("auto-edit", "full-auto"):
            return Decision(ALLOW, f"{self.mode} mode")
        return Decision(ASK, "modifies files")

    @staticmethod
    def first_match(rules: list[Rule], tool_name: str, subject: str | None) -> Rule | None:
        return next((r for r in rules if r.matches(tool_name, subject)), None)


def suggest_rule(tool: Tool, args: dict, ctx: ToolContext) -> str:
    """The allow rule offered when the user answers 'always'."""
    if tool.name == "bash":
        parts = split_commands(args.get("command", ""))
        if len(parts) == 1:
            words = parts[0].split()
            prefix = " ".join(words[:3]) if words[:1] in (["python3"], ["python"], ["uv"], ["npm"], ["npx"]) else " ".join(words[:2])
            return f"bash({prefix}*)"
        return f"bash({args.get('command', '')})"
    subject = tool.subject(args)
    if subject:
        return f"{tool.name}({path_info(ctx, subject)[1]})"
    return tool.name


# ---------------------------------------------------------------- settings files

def load_policy(cwd: Path, mode: str | None = None) -> PermissionPolicy:
    """Merge ~/.hx/settings.json and <workspace>/.hx/settings.json:
    {"permissions": {"mode": "ask", "allow": [...], "ask": [...], "deny": [...]}}"""
    merged: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
    file_mode = None
    for path in (Path.home() / ".hx" / "settings.json", cwd / ".hx" / "settings.json"):
        if path.is_file():
            perms = json.loads(path.read_text()).get("permissions", {})
            file_mode = perms.get("mode", file_mode)
            for key in merged:
                merged[key] += perms.get(key, [])
    return PermissionPolicy(
        mode=mode or file_mode or "ask",
        **{k: [Rule.parse(r) for r in v] for k, v in merged.items()},
    )
