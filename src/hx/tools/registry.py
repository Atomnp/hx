"""Tool registry. execute() is the single path every tool call goes through."""

from typing import Any

from hx.messages import ToolCall
from hx.repair import find_placeholders, repair_arguments
from hx.tools.base import Tool, ToolContext, ToolResult
from hx.permissions import ALLOW, DENY, suggest_rule
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

        if ctx.hooks:  # PreToolUse: your hooks can veto a call before permissions are even asked
            pre = ctx.hooks.run("PreToolUse", {"tool": call.name, "arguments": args}, call.name)
            self.report_hook_errors(pre, ctx)
            if pre.blocked:
                return ToolResult(f"Blocked by a PreToolUse hook: {pre.feedback or 'no reason given'}", True)

        if refusal := self.check_permission(tool, args, ctx):
            return refusal

        try:
            result = tool.run(args, ctx)
        except Exception as e:  # a buggy tool must never crash the agent
            result = ToolResult(f"Error: {call.name} failed: {type(e).__name__}: {e}", True)

        if ctx.hooks:  # PostToolUse: formatters, linters, extra checks; their output goes back to the model
            post = ctx.hooks.run("PostToolUse", {"tool": call.name, "arguments": args, "result": result.content,
                                                 "is_error": result.is_error}, call.name)
            self.report_hook_errors(post, ctx)
            if post.output:
                result.content += f"\n[PostToolUse hook]\n{post.output}"
            if post.blocked:
                result.content += f"\n[PostToolUse hook feedback]\n{post.feedback}"
        if notes:
            # Tell the model what we fixed, so it can send correct arguments next time.
            result.notes = notes
            result.content += f"\n[harness: fixed your arguments: {'; '.join(notes)}]"
        return result

    @staticmethod
    def check_permission(tool: Tool, args: dict, ctx: ToolContext) -> ToolResult | None:
        """None if the call may run; otherwise the refusal to send back to the model."""
        if ctx.permissions is None:
            return None
        decision = ctx.permissions.check(tool, args, ctx)
        if decision.action == ALLOW:
            return None
        if decision.action == DENY:
            return ToolResult(f"Permission denied: {decision.reason}. Don't retry this; find another way or ask the user.", True)

        # ASK
        if ctx.approve is None:
            return ToolResult(f"Permission required ({decision.reason}) but there's no one to approve it in this session.", True)
        rule = suggest_rule(tool, args, ctx)
        answer = ctx.approve(tool, args, decision, rule)
        if not answer.allowed:
            msg = "The user declined this action."
            msg += f" Their instructions: {answer.feedback}" if answer.feedback else " Ask them how they'd like to proceed."
            return ToolResult(msg, True)
        if answer.remember:
            ctx.permissions.add_allow(rule)
        return None

    @staticmethod
    def report_hook_errors(outcome, ctx: ToolContext) -> None:
        if outcome.errors and ctx.on_event:
            from hx.events import Notice

            for err in outcome.errors:
                ctx.on_event(Notice(err))
