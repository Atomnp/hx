"""Tool registry. execute() is the single path every tool call goes through."""

from typing import Any

from hx.messages import ToolCall
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

        problems = validate(call.arguments, tool.parameters)
        if problems:
            return ToolResult(f"Error: invalid arguments for {call.name}:\n- " + "\n- ".join(problems), True)

        try:
            return tool.run(call.arguments, ctx)
        except Exception as e:  # a buggy tool must never crash the agent
            return ToolResult(f"Error: {call.name} failed: {type(e).__name__}: {e}", True)
