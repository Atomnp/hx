"""Tool registry. execute() is the single path every tool call goes through."""

from typing import Any

from hx.messages import ToolCall
from hx.repair import find_placeholders, repair_arguments
from hx.tools.base import Tool, ToolContext, ToolResult
from hx.tools.schema import validate


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"Error: unknown tool '{call.name}'. Available tools: {', '.join(self._tools) or 'none'}", True)

        args, notes = repair_arguments(call.arguments, tool.parameters)

        if placeholders := find_placeholders(args):
            return ToolResult(
                f"Error: these arguments look like unfilled placeholders: {', '.join(placeholders)}. "
                "Use real values; look them up with the tools first if you don't know them.",
                True,
            )

        problems = validate(args, tool.parameters)
        if problems:
            return ToolResult(f"Error: invalid arguments for {call.name}:\n- " + "\n- ".join(problems), True, notes)

        try:
            result = tool.run(args, ctx)
        except Exception as e:  # a buggy tool must never crash the agent
            result = ToolResult(f"Error: {call.name} failed: {type(e).__name__}: {e}", True)
        if notes:
            # Tell the model what we fixed, so it can send correct arguments next time.
            result.notes = notes
            result.content += f"\n[harness: fixed your arguments: {'; '.join(notes)}]"
        return result
