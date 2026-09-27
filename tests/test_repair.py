from hx.agent import Agent
from hx.events import Notice
from hx.messages import Message, ToolCall
from hx.models.fake import FakeModel, call, say
from hx.repair import LoopGuard, extract_text_tool_calls, find_placeholders, repair_arguments
from hx.tools import Tool, ToolContext, ToolRegistry, ToolResult

KNOWN = {"read_file", "grep"}


# ---------------------------------------------------------------- text tool calls

def test_bare_json_calls_like_qwen_coder():
    text = '{"name": "read_file", "arguments": {"path": "a.py"}}\n{"name": "grep", "arguments": {"pattern": "x"}}'
    calls, rest = extract_text_tool_calls(text, KNOWN)
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"}), ("grep", {"pattern": "x"})]
    assert rest == ""


def test_tagged_and_fenced_calls_keep_surrounding_text():
    text = 'Let me look.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    calls, rest = extract_text_tool_calls(text, KNOWN)
    assert calls[0].arguments == {"path": "a.py"} and rest == "Let me look."
    calls, _ = extract_text_tool_calls('```json\n{"function": {"name": "grep", "arguments": "{\\"pattern\\": \\"y\\"}"}}\n```', KNOWN)
    assert calls[0].arguments == {"pattern": "y"}


def test_ordinary_json_is_left_alone():
    text = 'The config is {"name": "my-app", "version": 2}.'
    calls, rest = extract_text_tool_calls(text, KNOWN)
    assert calls == [] and rest == text


# ---------------------------------------------------------------- arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer"},
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
    },
    "required": ["path"],
}


def test_repairs_common_type_mistakes():
    args, notes = repair_arguments({"path": "a", "offset": "10", "flag": "True", "items": "[1, 2]"}, SCHEMA)
    assert args == {"path": "a", "offset": 10, "flag": True, "items": [1, 2]}
    assert len(notes) == 3


def test_drops_null_optionals_and_unwraps():
    args, _ = repair_arguments({"path": "a", "offset": None}, SCHEMA)
    assert args == {"path": "a"}
    args, notes = repair_arguments({"arguments": {"path": "a"}}, SCHEMA)
    assert args == {"path": "a"} and "unwrapped" in notes[0]


def test_leaves_valid_and_unfixable_values_alone():
    args, notes = repair_arguments({"path": "a", "offset": "ten"}, SCHEMA)
    assert args == {"path": "a", "offset": "ten"} and notes == []


def test_placeholders():
    assert find_placeholders({"person_id": "<person_id>", "path": "{{path}}"}) == ["person_id='<person_id>'", "path='{{path}}'"]
    assert find_placeholders({"old_string": "<br>", "path": "src/a.py"}) == []  # real values pass


# ---------------------------------------------------------------- loop guard

def test_loop_guard_fires_on_third_identical_call():
    g = LoopGuard()
    c = ToolCall("1", "grep", {"pattern": "x"})
    assert g.check(c) is None and g.check(c) is None
    assert "3 times" in g.check(c)
    assert g.check(ToolCall("2", "grep", {"pattern": "y"})) is None


# ---------------------------------------------------------------- end to end through the loop

class Echo(Tool):
    name = "read_file"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}

    def run(self, args, ctx):
        return ToolResult(f"read {args['path']} limit={args.get('limit')}")


def test_agent_runs_text_calls_and_repaired_args():
    model = FakeModel([
        Message("assistant", '{"name": "read_file", "arguments": {"path": "a.py", "limit": "5"}}'),
        say("done"),
    ])
    agent = Agent(model, tools=ToolRegistry([Echo()]))
    events = []
    agent.run("read a.py", on_event=events.append)

    tool_msg = model.requests[1][-1]
    assert tool_msg.role == "tool" and "read a.py limit=5" in tool_msg.content
    assert "fixed your arguments" in tool_msg.content
    assert any(isinstance(e, Notice) and "as text" in e.text for e in events)


def test_agent_rejects_placeholder_and_warns_on_loops():
    model = FakeModel([call("read_file", path="<file_path>")] + [call("read_file", path="a.py")] * 3 + [say("ok")])
    agent = Agent(model, tools=ToolRegistry([Echo()]))
    agent.run("go")
    assert "placeholders" in model.requests[1][-1].content
    assert "3 times" in model.requests[4][-1].content


def test_escaped_newlines_in_code_args_are_fixed():
    schema = {"type": "object", "properties": {"content": {"type": "string"}, "command": {"type": "string"}}}
    args, notes = repair_arguments({"content": 'def f():\\n    """Doc."""\\n    return 1'}, schema)
    assert args["content"] == 'def f():\n    """Doc."""\n    return 1' and "escaped newlines" in notes[0]


def test_legit_backslash_n_is_left_alone():
    schema = {"type": "object", "properties": {"content": {"type": "string"}, "command": {"type": "string"}}}
    one = 'print("a\\nb")'  # a single \n inside a string literal
    real = 'x = 1\nprint("a\\nb\\nc")'  # has real newlines already
    cmd = "printf 'a\\nb\\nc'"  # shell commands are never touched
    args, notes = repair_arguments({"content": one}, schema)
    assert args["content"] == one and notes == []
    args, _ = repair_arguments({"content": real}, schema)
    assert args["content"] == real
    args, _ = repair_arguments({"command": cmd}, schema)
    assert args["command"] == cmd
