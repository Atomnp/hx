"""Tracing (OpenTelemetry GenAI spans) and usage stats, built from agent events.

Spans go to ~/.hx/traces/<session>.jsonl, and to $HX_OTLP_ENDPOINT if it's set.
"""

import json
import secrets
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from hx import __version__
from hx.events import Event, ModelCallStarted, StepFinished, ToolFinished, ToolStarted, TurnFinished, TurnStarted

TRACE_DIR = Path.home() / ".hx" / "traces"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int = 0
    attributes: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def to_dict(self) -> dict:
        return {"name": self.name, "trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
                "start_ns": self.start_ns, "end_ns": self.end_ns, "attributes": self.attributes, "error": self.error}


def otlp_value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def to_otlp(spans: list[Span]) -> dict:
    """The OTLP/HTTP JSON payload (resourceSpans → scopeSpans → spans)."""
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "hx"}}]},
        "scopeSpans": [{
            "scope": {"name": "hx", "version": __version__},
            "spans": [{
                "traceId": s.trace_id, "spanId": s.span_id, **({"parentSpanId": s.parent_id} if s.parent_id else {}),
                "name": s.name, "kind": 3 if s.name.startswith("chat") else 1,  # CLIENT for model calls, INTERNAL otherwise
                "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns),
                "attributes": [{"key": k, "value": otlp_value(v)} for k, v in s.attributes.items()],
                "status": {"code": 2, "message": s.error} if s.error else {"code": 1},
            } for s in spans],
        }],
    }]}


class Tracer:
    """An event handler that turns agent events into spans."""

    def __init__(self, model: str, provider: str, path: Path | None = None, otlp_endpoint: str | None = None):
        self.model, self.provider = model, provider
        self.path = path
        self.otlp_endpoint = otlp_endpoint
        self.turn: Span | None = None
        self.open: dict[str, Span] = {}  # "chat" or a tool call id → its open span
        self.spans: list[Span] = []  # spans of the current/last turn
        self.export_errors = 0

    def start(self, name: str, **attributes) -> Span:
        parent = self.turn
        span = Span(name, parent.trace_id if parent else secrets.token_hex(16), secrets.token_hex(8),
                    parent.span_id if parent else None, time.time_ns(), attributes=attributes)
        self.spans.append(span)
        return span

    def __call__(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self.spans = []
            self.turn = None
            self.turn = self.start("invoke_agent hx", **{"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "hx"})
        elif self.turn is None:
            return
        elif isinstance(event, ModelCallStarted):
            self.open["chat"] = self.start(f"chat {self.model}", **{
                "gen_ai.operation.name": "chat", "gen_ai.provider.name": self.provider,
                "gen_ai.request.model": self.model, "hx.step": event.step})
        elif isinstance(event, StepFinished) and (span := self.open.pop("chat", None)):
            span.end_ns = time.time_ns()
            span.attributes.update({"gen_ai.usage.input_tokens": event.usage.prompt_tokens,
                                    "gen_ai.usage.output_tokens": event.usage.completion_tokens,
                                    "hx.context_used": round(event.context_used, 3)})
        elif isinstance(event, ToolStarted):
            self.open[event.call.id] = self.start(f"execute_tool {event.call.name}", **{
                "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": event.call.name,
                "gen_ai.tool.call.id": event.call.id})
        elif isinstance(event, ToolFinished) and (span := self.open.pop(event.call.id, None)):
            span.end_ns = time.time_ns()
            if event.is_error:
                span.error = event.result[:200]
                span.attributes["error.type"] = "tool_error"
        elif isinstance(event, TurnFinished):
            now = time.time_ns()
            for span in self.open.values():  # interrupted mid-call
                span.end_ns, span.error = now, "interrupted"
            self.open.clear()
            self.turn.end_ns = now
            self.turn.attributes["hx.finish_reason"] = event.reason
            self.turn.attributes["hx.steps"] = event.steps
            self.flush()

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                for s in self.spans:
                    f.write(json.dumps(s.to_dict()) + "\n")
        if self.otlp_endpoint:
            req = urllib.request.Request(self.otlp_endpoint, data=json.dumps(to_otlp(self.spans)).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=3).read()
            except OSError:
                self.export_errors += 1  # observability must never break the agent

    def tree(self) -> str:
        """The last turn as an indented tree with durations (for /trace)."""
        if not self.spans:
            return "No trace yet."
        lines = []
        for s in self.spans:
            indent = "" if s.parent_id is None else "  └ "
            extra = ""
            if "gen_ai.usage.input_tokens" in s.attributes:
                extra = f"  {s.attributes['gen_ai.usage.input_tokens']} in / {s.attributes['gen_ai.usage.output_tokens']} out"
            if s.error:
                extra += f"  ✗ {s.error[:60]}"
            lines.append(f"{indent}{s.name:<32} {s.ms:>9.0f} ms{extra}")
        return "\n".join(lines)


class Stats:
    """Running totals for the session (for /stats)."""

    def __init__(self):
        self.turns = 0
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_seconds = 0.0
        self.tool_calls: Counter = Counter()
        self.tool_errors: Counter = Counter()
        self.tool_seconds: dict[str, float] = defaultdict(float)
        self._tool_start: dict[str, float] = {}

    def __call__(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self.turns += 1
        elif isinstance(event, StepFinished):
            self.model_calls += 1
            self.input_tokens += event.usage.prompt_tokens
            self.output_tokens += event.usage.completion_tokens
            self.model_seconds += event.usage.duration_s
        elif isinstance(event, ToolStarted):
            self._tool_start[event.call.id] = time.monotonic()
            self.tool_calls[event.call.name] += 1
        elif isinstance(event, ToolFinished):
            started = self._tool_start.pop(event.call.id, None)
            if started is not None:
                self.tool_seconds[event.call.name] += time.monotonic() - started
            if event.is_error:
                self.tool_errors[event.call.name] += 1

    def report(self, price_in_per_m: float = 0.0, price_out_per_m: float = 0.0) -> str:
        cost = self.input_tokens / 1e6 * price_in_per_m + self.output_tokens / 1e6 * price_out_per_m
        lines = [
            f"turns {self.turns} · model calls {self.model_calls} · model time {self.model_seconds:.1f}s",
            f"tokens: {self.input_tokens:,} in (summed over calls) · {self.output_tokens:,} out"
            + (f" · est. cost ${cost:.4f}" if price_in_per_m or price_out_per_m else " · local model: $0"),
        ]
        if self.input_tokens and self.model_calls:
            lines.append(f"avg prompt {self.input_tokens // self.model_calls:,} tokens/call")
        for name, n in self.tool_calls.most_common():
            errs = f", {self.tool_errors[name]} failed" if self.tool_errors[name] else ""
            lines.append(f"  {name:<20} {n:>3} call(s) {self.tool_seconds[name]:>7.1f}s{errs}")
        return "\n".join(lines)


def fanout(*handlers):
    """One event handler that calls several: UI + tracer + stats."""
    def handle(event):
        for h in handlers:
            h(event)
    return handle
