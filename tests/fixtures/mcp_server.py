"""A tiny MCP server for tests and demos: JSON-RPC 2.0 over stdio, written without any SDK.

Tools: notes_add(text), notes_list(), weather(city). Notes are kept in memory for the life of the process.
"""

import json
import sys

NOTES: list[str] = []
TOOLS = [
    {"name": "notes_add", "description": "Save a short note.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "notes_list", "description": "List saved notes.", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "weather", "description": "Current weather for a city (fake data, for demos).",
     "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
]


def handle(msg):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                "serverInfo": {"name": "hx-test-server", "version": "1.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name, args = params["name"], params.get("arguments", {})
        if name == "notes_add":
            NOTES.append(args["text"])
            text, err = f"Saved note #{len(NOTES)}.", False
        elif name == "notes_list":
            text, err = "\n".join(f"{i}. {n}" for i, n in enumerate(NOTES, 1)) or "No notes yet.", False
        elif name == "weather":
            text, err = (f"{args['city']}: 21°C, light rain", False) if args["city"] else ("city is empty", True)
        else:
            raise KeyError(name)
        return {"content": [{"type": "text", "text": text}], "isError": err}
    raise LookupError(method)


for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue  # notification (e.g. notifications/initialized)
    print("debug: handling", msg.get("method"), file=sys.stderr)  # servers log to stderr, never stdout
    try:
        reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
    except (LookupError, KeyError) as e:
        reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"unknown: {e}"}}
    print(json.dumps(reply), flush=True)
