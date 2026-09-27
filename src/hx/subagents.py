"""Subagents: the task tool runs a child agent with its own context and a limited tool set.

Built-in types are "explore" (read-only) and "general". Custom ones live in .hx/agents/<name>.md:

    ---
    name: reviewer
    description: Reviews a diff for bugs
    tools: read_file, grep, glob
    ---
    You are a code reviewer. ...
"""

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

from hx.events import Notice, ToolStarted
from hx.tools.base import Tool, ToolContext, ToolResult
from hx.tools.registry import ToolRegistry

READ_ONLY_TOOLS = ["read_file", "list_dir", "grep", "glob"]


@dataclass
class AgentSpec:
    name: str
    description: str
    tools: list[str] | None  # None = every tool except `task` (no recursion)
    instructions: str
    max_steps: int = 15


BUILTIN = [
    AgentSpec(
        "explore",
        "Read-only codebase exploration: finding where things are defined or used, how something works, which files matter. "
        "Fast and cheap for your context: only its report comes back.",
        READ_ONLY_TOOLS,
        "You are an exploration subagent. Search and read the codebase to answer the task. You can't change anything. "
        "Be efficient: grep/glob first, read only what you need. Finish with a concise report: the answer, the key "
        "file paths with line numbers, and anything surprising. The report is all the caller will see.",
    ),
    AgentSpec(
        "general",
        "A general-purpose agent with all tools, for a self-contained multi-step job you can describe completely.",
        None,
        "You are a subagent working on one delegated task. Complete it, verify your work, then finish with a concise "
        "report of what you did and found. The report is all the caller will see.",
    ),
]


def parse_agent_file(path: Path) -> AgentSpec | None:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(), re.S)
    if not m:
        return None
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    tools = [t.strip() for t in meta["tools"].split(",")] if meta.get("tools") else None
    return AgentSpec(meta.get("name", path.stem), meta.get("description", ""), tools, m.group(2).strip())


def load_specs(cwd: Path) -> list[AgentSpec]:
    specs = {s.name: s for s in BUILTIN}
    for d in (Path.home() / ".hx" / "agents", cwd / ".hx" / "agents"):
        for f in sorted(d.glob("*.md")) if d.is_dir() else []:
            if spec := parse_agent_file(f):
                specs[spec.name] = spec  # project definitions override built-ins and user ones
    return list(specs.values())


class Task(Tool):
    """The tool the main agent calls. Built per session because its schema lists the available agent types."""

    name = "task"
    # Delegating changes nothing by itself: every tool call the child makes goes through the same
    # permission checks (and approval prompts) as the parent's.
    read_only = True

    def __init__(self, specs: list[AgentSpec], parent_tools: ToolRegistry, model, make_agent):
        self.specs = {s.name: s for s in specs}
        self.parent_tools = parent_tools
        self.model = model
        self.make_agent = make_agent  # builds a child Agent (passed in to avoid a circular import)
        types = "\n".join(f"- {s.name}: {s.description}" for s in specs)
        self.description = (
            "Delegate a self-contained job to a subagent with its own fresh context. Only its final report comes "
            "back, so this keeps your own context small. Use explore for searching/reading large parts of the "
            f"codebase. Give it a complete, specific task: it can't see this conversation.\nAgent types:\n{types}"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": list(self.specs)},
                "task": {"type": "string", "description": "Everything the subagent needs to know to do the job"},
            },
            "required": ["agent", "task"],
            "additionalProperties": False,
        }

    def child_tools(self, spec: AgentSpec) -> ToolRegistry:
        names = spec.tools if spec.tools is not None else [n for n in self.parent_tools.names() if n != "task"]
        return ToolRegistry([self.parent_tools.get(n) for n in names if self.parent_tools.get(n) and n != "task"])

    def run(self, args, ctx: ToolContext):
        spec = self.specs[args["agent"]]
        # Same workspace, permissions, sandbox and approver; but its own read-tracking and todo list.
        child_ctx = dataclasses.replace(ctx, read_files={}, todos=[])
        child = self.make_agent(self.model, self.child_tools(spec), child_ctx, spec.instructions, spec.max_steps)

        emit = ctx.on_event or (lambda e: None)
        calls = 0

        def forward(event):
            nonlocal calls
            if isinstance(event, ToolStarted):
                calls += 1
                emit(Notice(f"[{spec.name}] → {event.call.name}({str(event.call.arguments)[:100]})"))

        report = child.run(args["task"], on_event=forward) or "(the subagent stopped without a report)"
        return ToolResult(f"{report}\n[{spec.name} subagent: {calls} tool call(s), its context discarded]")
