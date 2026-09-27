"""todo_write: the agent's checklist. The model always sends the full list."""

from hx.tools.base import Tool, ToolContext, ToolResult

STATUSES = ["pending", "in_progress", "completed"]
MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render(todos: list[dict]) -> str:
    if not todos:
        return "(todo list is empty)"
    return "\n".join(f"{MARKS[t['status']]} {t['content']}" for t in todos)


def unfinished(todos: list[dict]) -> list[dict]:
    return [t for t in todos if t["status"] != "completed"]


class TodoWrite(Tool):
    name = "todo_write"
    description = (
        "Create or update your task list for the current request. Use it for any task with 3 or more steps: "
        "write the plan first, then keep it current: mark an item in_progress when you start it (only one at a time) "
        "and completed as soon as it's done. Always send the complete list. Skip it for simple one-step requests."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full, current task list",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to do, as a short imperative"},
                        "status": {"type": "string", "enum": STATUSES},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
        "additionalProperties": False,
    }
    read_only = True  # only changes the harness's own state, never the user's files

    def run(self, args, ctx: ToolContext):
        todos = [{"content": t["content"].strip(), "status": t["status"]} for t in args["todos"]]
        if sum(t["status"] == "in_progress" for t in todos) > 1:
            return ToolResult("Error: only one item may be in_progress at a time. Finish one before starting the next.", True)
        if any(not t["content"] for t in todos):
            return ToolResult("Error: every item needs content.", True)
        ctx.todos = todos
        done = len(todos) - len(unfinished(todos))
        return ToolResult(f"Todo list updated ({done}/{len(todos)} done):\n{render(todos)}")
