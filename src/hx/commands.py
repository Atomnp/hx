"""Slash commands. Built-ins are Python functions; custom ones are Markdown prompt templates in
.hx/commands/ that can use $ARGUMENTS.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hx.memory import add_memory, memory_paths
from hx.permissions import MODES
from hx.tools.todo import render


@dataclass
class Command:
    name: str
    help: str
    run: Callable[["object", str], str | None]  # (repl state, args) -> text to print, or None


@dataclass
class Prompt:
    """Returned by a command that wants the agent to handle a (generated) prompt."""

    text: str


def load_custom(cwd: Path) -> dict[str, tuple[str, str]]:
    """name -> (description, template)"""
    found = {}
    for d in (Path.home() / ".hx" / "commands", cwd / ".hx" / "commands"):
        for f in sorted(d.glob("*.md")) if d.is_dir() else []:
            text = f.read_text()
            m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
            desc, body = "", text
            if m:
                meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
                desc, body = meta.get("description", "").strip(), m.group(2)
            found[f.stem] = (desc or f"custom command from {f}", body.strip())
    return found


class Commands:
    def __init__(self, state):
        self.state = state  # the REPL state: .agent, .cwd, .exit
        self.builtin = {c.name: c for c in [
            Command("help", "show commands", self.help),
            Command("undo", "undo the last turn (files and conversation)", lambda s, a: s.agent.undo()),
            Command("checkpoints", "list workspace snapshots", self.checkpoints),
            Command("compact", "summarize the conversation now to free context", lambda s, a: s.agent.compact() or "Nothing to compact yet."),
            Command("clear", "start a fresh conversation (files untouched)", self.clear),
            Command("todos", "show the agent's todo list", lambda s, a: render(s.agent.ctx.todos)),
            Command("mode", "show or change the permission mode: /mode auto-edit", self.mode),
            Command("memory", "show memory files; '# fact' adds to project memory", self.memory),
            Command("stats", "tokens, model time and tool usage for this session", self.stats),
            Command("trace", "the last turn as a span tree with timings", self.trace),
            Command("exit", "quit", self.quit),
        ]}
        self.custom = load_custom(state.cwd)

    def handle(self, line: str) -> str | Prompt | None:
        """Process one REPL line that starts with '/' or '#'. Returns text to print or a Prompt for the agent."""
        if line.startswith("#"):
            path = add_memory(self.state.cwd, line.lstrip("#").strip())
            return f"Remembered in {path}."
        name, _, args = line[1:].partition(" ")
        if name in ("quit", "q"):
            name = "exit"
        if cmd := self.builtin.get(name):
            return cmd.run(self.state, args.strip())
        if name in self.custom:
            return Prompt(self.custom[name][1].replace("$ARGUMENTS", args.strip()))
        return f"Unknown command /{name}. Try /help."

    # -- built-ins

    def help(self, s, a):
        lines = [f"  /{c.name:<12} {c.help}" for c in self.builtin.values()]
        lines += [f"  /{n:<12} {d}" for n, (d, _) in self.custom.items()]
        return "Commands:\n" + "\n".join(lines) + "\n  # <fact>      remember a fact in project memory"

    def checkpoints(self, s, a):
        if not s.agent.checkpoints:
            return "Checkpoints are off."
        return "\n".join(f"  {c.sha[:8]}  {c.age:>16}  {c.label}" for c in s.agent.checkpoints.list()) or "No checkpoints yet."

    def clear(self, s, a):
        s.agent.messages = s.agent.messages[:1]
        s.agent.turns = []
        s.agent.ctx.todos = []
        s.agent.ctx.read_files.clear()
        s.agent.rewrote_history()
        return "Started a fresh conversation."

    def mode(self, s, a):
        policy = s.agent.ctx.permissions
        if not a:
            return f"Permission mode: {policy.mode} (choose from {', '.join(MODES)})"
        if a not in MODES:
            return f"Unknown mode {a!r}; choose from {', '.join(MODES)}"
        policy.mode = a
        return f"Permission mode is now {a}."

    def memory(self, s, a):
        out = []
        for scope, path in memory_paths(s.cwd).items():
            out.append(f"{scope}: {path}\n{path.read_text().strip() if path.is_file() else '  (empty)'}")
        return "\n\n".join(out)

    def stats(self, s, a):
        return s.stats.report() if getattr(s, "stats", None) else "Stats are off."

    def trace(self, s, a):
        return s.tracer.tree() if getattr(s, "tracer", None) else "Tracing is off."

    def quit(self, s, a):
        s.exit = True
        return None
