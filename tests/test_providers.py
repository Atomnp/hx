import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hx.config import Settings
from hx.messages import Message, ToolCall, user
from hx.models import ModelError, make_client
from hx.models.base import ModelResponse, Usage
from hx.models.openai_compat import OpenAICompatClient, to_wire
from hx.models.retry import RetryingClient


# ---------------------------------------------------------------- retries

class Flaky:
    name = "flaky"

    def __init__(self, failures, error, stream_first=False):
        self.failures, self.error, self.stream_first, self.calls = failures, error, stream_first, 0

    def chat(self, messages, tools=None, on_text=None):
        self.calls += 1
        if self.calls <= self.failures:
            if self.stream_first and on_text:
                on_text("partial ")
            raise self.error
        return ModelResponse(Message("assistant", "ok"), Usage())


def retrying(inner, **kw):
    waits, notes = [], []
    return RetryingClient(inner, sleep=waits.append, on_retry=notes.append, **kw), waits, notes


def test_retries_transient_errors_with_backoff():
    client, waits, notes = retrying(Flaky(3, ModelError("503", retryable=True)))
    assert client.chat([user("hi")]).message.content == "ok"
    assert len(waits) == 3 and 0.5 <= waits[0] <= 1.0 and 1.0 <= waits[1] <= 2.0 and 2.0 <= waits[2] <= 4.0
    assert "attempt 2/4" in notes[0]


def test_gives_up_after_max_attempts():
    client, waits, _ = retrying(Flaky(9, ModelError("503", retryable=True)))
    with pytest.raises(ModelError):
        client.chat([user("hi")])
    assert len(waits) == 3


def test_does_not_retry_permanent_errors():
    inner = Flaky(1, ModelError("400 bad request", retryable=False))
    client, waits, _ = retrying(inner)
    with pytest.raises(ModelError):
        client.chat([user("hi")])
    assert inner.calls == 1 and waits == []


def test_never_retries_after_text_was_streamed():
    inner = Flaky(1, ModelError("connection reset", retryable=True), stream_first=True)
    client, _, _ = retrying(inner)
    with pytest.raises(ModelError):
        client.chat([user("hi")], on_text=lambda t: None)
    assert inner.calls == 1


def test_honors_retry_after():
    client, waits, _ = retrying(Flaky(1, ModelError("429", retryable=True, retry_after=7)))
    client.chat([user("hi")])
    assert waits == [7]


# ---------------------------------------------------------------- OpenAI-compatible client over SSE

def sse_server(events, status=200, headers=None):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if status == 200:
                for e in events:
                    self.wfile.write(f"data: {json.dumps(e) if not isinstance(e, str) else e}\n\n".encode())
            else:
                self.wfile.write(b'{"error": "slow down"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, received


def test_streams_text_and_fragmented_tool_calls():
    srv, received = sse_server([
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "look."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "grep", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"pattern":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "TODO"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12}},
        "[DONE]",
    ])
    pieces = []
    r = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "m", api_key="sk-test").chat([user("hi")], on_text=pieces.append)
    srv.shutdown()
    assert pieces == ["Let me ", "look."] and r.message.content == "Let me look."
    assert r.message.tool_calls == [ToolCall("call_9", "grep", {"pattern": "TODO"})]
    assert r.usage.prompt_tokens == 50 and r.stop_reason == "tool_calls"
    auth, body = received[0]
    assert auth == "Bearer sk-test" and body["stream"] is True and body["stream_options"]["include_usage"]


def test_http_429_is_retryable_with_retry_after():
    srv, _ = sse_server([], status=429, headers={"Retry-After": "3"})
    with pytest.raises(ModelError) as e:
        OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "m").chat([user("hi")])
    srv.shutdown()
    assert e.value.retryable and e.value.retry_after == 3


def test_wire_format():
    c = ToolCall("c1", "grep", {"pattern": "x"})
    assert to_wire(Message("assistant", "", tool_calls=[c]))["tool_calls"][0]["function"]["arguments"] == '{"pattern": "x"}'
    assert to_wire(Message("tool", "result", tool_call_id="c1", name="grep")) == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_factory():
    assert make_client(Settings()).name.startswith("ollama:")
    assert make_client(Settings(provider="openai", model="gpt-x")).name == "openai-compat:gpt-x"
    with pytest.raises(ValueError):
        make_client(Settings(provider="nope"))
