"""Skills: folders with a SKILL.md, from ~/.hx/skills and .hx/skills.

Only names and descriptions go into the system prompt; the full instructions load when the model asks.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from hx.tools.base import Tool, ToolContext, ToolResult


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # the skill's folder

    def body(self) -> str:
        text = (self.path / "SKILL.md").read_text()
        m = re.match(r"^---\n.*?\n---\n?(.*)$", text, re.S)
        return (m.group(1) if m else text).strip()

    def resources(self) -> list[Path]:
        return sorted(p for p in self.path.rglob("*") if p.is_file() and p.name != "SKILL.md")


def parse(skill_dir: Path) -> Skill | None:
    f = skill_dir / "SKILL.md"
    if not f.is_file():
        return None
    m = re.match(r"^---\n(.*?)\n---", f.read_text(), re.S)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    if not meta.get("description"):
        return None  # without a description the model can't know when to use it
    return Skill(meta.get("name", skill_dir.name), meta["description"], skill_dir.resolve())


def discover(cwd: Path) -> list[Skill]:
    found: dict[str, Skill] = {}
    for root in (Path.home() / ".hx" / "skills", cwd / ".hx" / "skills"):
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and (s := parse(d)):
                found[s.name] = s  # project skills override user skills with the same name
    return list(found.values())


def prompt_section(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        "# Skills\nSpecialized instructions you can load with the skill tool. When a task matches a skill's "
        f"description, load it BEFORE starting and follow it.\n{lines}"
    )


class SkillTool(Tool):
    name = "skill"
    read_only = True

    def __init__(self, skills: list[Skill]):
        self.skills = {s.name: s for s in skills}
        self.description = "Load a skill's full instructions (see the Skills section of your instructions for the list)."
        self.parameters = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(self.skills) or [""]}},
            "required": ["name"],
            "additionalProperties": False,
        }

    def run(self, args, ctx: ToolContext):
        skill = self.skills.get(args["name"])
        if not skill:
            return ToolResult(f"Error: no skill named {args['name']!r}. Available: {', '.join(self.skills) or 'none'}", True)
        out = f"# Skill: {skill.name}\n(folder: {skill.path})\n\n{skill.body()}"
        if res := skill.resources():
            out += "\n\nFiles in this skill (read or run them only if the instructions call for it):\n"
            out += "\n".join(f"- {p}" for p in res)
        return ToolResult(out)
