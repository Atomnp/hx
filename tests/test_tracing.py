import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from hx.agent import Agent
from hx.models.fake import FakeModel, call, say
from hx.tools import ToolContext
from hx.tools.builtin import default_tools
from hx.tracing import Stats, Tracer, fanout, to_otlp


def run_traced(tmp_path, replies, **tracer_kw):
    tracer = Tracer("qwen3:14b", "ollama", path=tmp_path / "trace.jsonl", **tracer_kw)
    stats = Stats()
    agent = Agent(FakeModel(replies), tools=default_tools(), ctx=ToolContext(cwd=tmp_path))
    agent.run("list files then answer", on_event=fanout(tracer, stats))
    return tracer, stats


def test_span_tree_follows_genai_conventions(tmp_path):
    tracer, _ = run_traced(tmp_path, [call("list_dir"), call("read_file", call_id="c2", path="missing.py"), say("done")])
    names = [s.name for s in tracer.spans]
    assert names == ["invoke_agent hx", "chat qwen3:14b", "execute_tool list_dir", "chat qwen3:14b",
                     "execute_tool read_file", "chat qwen3:14b"]
    root, chat, tool = tracer.spans[0], tracer.spans[1], tracer.spans[2]
    assert root.parent_id is None and all(s.parent_id == root.span_id and s.trace_id == root.trace_id for s in tracer.spans[1:])
    assert chat.attributes["gen_ai.operation.name"] == "chat" and chat.attributes["gen_ai.usage.input_tokens"] == 10
    assert tool.attributes["gen_ai.tool.name"] == "list_dir"
    assert tracer.spans[4].error and tracer.spans[4].attributes["error.type"] == "tool_error"  # missing file
    assert root.attributes["hx.finish_reason"] == "done" and all(s.end_ns >= s.start_ns for s in tracer.spans)
    assert "execute_tool read_file" in tracer.tree() and "✗" in tracer.tree()


def test_spans_written_as_jsonl(tmp_path):
    run_traced(tmp_path, [say("hi")])
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [l["name"] for l in lines] == ["invoke_agent hx", "chat qwen3:14b"]


def test_otlp_export_payload_and_delivery(tmp_path):
    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    run_traced(tmp_path, [say("hi")], otlp_endpoint=f"http://127.0.0.1:{srv.server_port}/v1/traces")
    srv.shutdown()
    path, payload = received[0]
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert path == "/v1/traces" and spans[1]["name"] == "chat qwen3:14b" and spans[1]["kind"] == 3
    attrs = {a["key"]: a["value"] for a in spans[1]["attributes"]}
    assert attrs["gen_ai.usage.output_tokens"] == {"intValue": "5"}


def test_dead_collector_never_breaks_the_agent(tmp_path):
    tracer, _ = run_traced(tmp_path, [say("hi")], otlp_endpoint="http://127.0.0.1:9/v1/traces")
    assert tracer.export_errors == 1


def test_stats_totals(tmp_path):
    _, stats = run_traced(tmp_path, [call("list_dir"), call("list_dir", call_id="c2"), say("done")])
    assert stats.turns == 1 and stats.model_calls == 3
    assert stats.input_tokens == 30 and stats.output_tokens == 15
    assert stats.tool_calls["list_dir"] == 2
    report = stats.report(price_in_per_m=3.0, price_out_per_m=15.0)
    assert "30 in" in report and "est. cost $0.0003" in report and "list_dir" in report
