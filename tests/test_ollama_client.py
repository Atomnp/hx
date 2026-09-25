"""Tests the Ollama client against a tiny fake HTTP server that streams NDJSON like Ollama does."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hx.config import Settings
from hx.messages import Message, ToolCall, user
from hx.models import ModelError
from hx.models.ollama import OllamaClient, to_wire


def serve(chunks):
    """Start a server on a free port that answers /api/chat with `chunks` as NDJSON. Returns (server, requests)."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for c in chunks:
                self.wfile.write((json.dumps(c) + "\n").encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def client_for(server):
    return OllamaClient(Settings(host=f"http://127.0.0.1:{server.server_port}"))


def test_streams_text_and_reports_usage():
    server, received = serve([
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 12, "eval_count": 2},
    ])
    pieces = []
    r = client_for(server).chat([user("hi")], on_text=pieces.append)
    server.shutdown()

    assert pieces == ["Hel", "lo"]
    assert r.message.content == "Hello"
    assert r.usage.prompt_tokens == 12 and r.usage.completion_tokens == 2
    assert r.stop_reason == "stop"
    assert received[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert received[0]["stream"] is True


def test_parses_tool_calls():
    server, _ = serve([
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ])
    r = client_for(server).chat([user("read a.py")], tools=[{"type": "function", "function": {"name": "read_file"}}])
    server.shutdown()

    [call] = r.message.tool_calls
    assert call.name == "read_file" and call.arguments == {"path": "a.py"}
    assert call.id  # we always assign an id


def test_wire_format_for_tool_messages():
    call = ToolCall("c1", "read_file", {"path": "a.py"})
    assert to_wire(Message("assistant", tool_calls=[call]))["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
    ]
    assert to_wire(Message("tool", "file text", tool_call_id="c1", name="read_file"))["tool_name"] == "read_file"


def test_unreachable_server_gives_helpful_error():
    with pytest.raises(ModelError, match="Is it running"):
        OllamaClient(Settings(host="http://127.0.0.1:9"), timeout=2).chat([user("hi")])
