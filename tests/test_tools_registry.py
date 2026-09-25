from hx.agent import Agent
from hx.messages import ToolCall
from hx.models.fake import FakeModel, call, say
from hx.tools import Tool, ToolContext, ToolRegistry, ToolResult
from hx.tools.schema import validate


class Add(Tool):
    name = "add"
    description = "Add two integers."
    parameters = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    read_only = True

    def run(self, args, ctx):
        return ToolResult(str(args["a"] + args["b"]))


class Boom(Tool):
    name = "boom"

    def run(self, args, ctx):
        raise RuntimeError("kaboom")


def run(registry, name, **args):
    return registry.execute(ToolCall("c1", name, args), ToolContext())


def test_schema_format():
    s = Add().schema()
    assert s["type"] == "function" and s["function"]["name"] == "add"
    assert s["function"]["parameters"]["required"] == ["a", "b"]


def test_execute_valid_call():
    assert run(ToolRegistry([Add()]), "add", a=2, b=3) == ToolResult("5")


def test_unknown_tool_lists_available_ones():
    r = run(ToolRegistry([Add()]), "subtract")
    assert r.is_error and "Available tools: add" in r.content


def test_invalid_arguments_are_explained():
    r = run(ToolRegistry([Add()]), "add", a="2", c=1)
    assert r.is_error
    assert "a: expected integer" in r.content
    assert "b: required field missing" in r.content
    assert "c: unknown field" in r.content


def test_tool_exception_becomes_error_result():
    r = run(ToolRegistry([Boom()]), "boom")
    assert r.is_error and "RuntimeError: kaboom" in r.content


def test_validator_edge_cases():
    assert validate(True, {"type": "integer"})  # bool is not an int here
    assert validate("x", {"type": "string", "enum": ["a", "b"]})
    assert validate([1, "2"], {"type": "array", "items": {"type": "integer"}}) == [
        "arguments[1]: expected integer, got str ('2')"
    ]


def test_agent_sends_schemas_and_runs_tools():
    model = FakeModel([call("add", a=40, b=2), say("42")])
    agent = Agent(model, tools=ToolRegistry([Add()]))

    assert agent.run("what's 40+2?") == "42"
    assert model.tools_seen[0][0]["function"]["name"] == "add"
    assert model.requests[1][-1].content == "42"  # the tool result the model saw
