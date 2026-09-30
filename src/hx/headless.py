"""Headless mode (hx -p): run one request and exit.

Output: text, json (a summary at the end) or stream-json (one event per line). Exit code 0 if the agent
finished, 1 otherwise. Nobody can answer permission prompts, so "ask" becomes "deny".
"""

import json
import sys
import time
from dataclasses import asdict, is_dataclass

from hx.agent import Agent
from hx.events import Event, Notice, StepFinished, TextDelta, ToolFinished, ToolStarted, TurnFinished
from hx.models import ModelError


def event_to_dict(event: Event) -> dict:
    """{"type": "tool_started", ...fields} with nested dataclasses (ToolCall, Usage) expanded."""
    name = type(event).__name__
    kind = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
    fields = {k: (asdict(v) if is_dataclass(v) else v) for k, v in vars(event).items()}
    return {"type": kind, **fields}


def run_headless(agent: Agent, prompt: str, output_format: str = "text") -> int:
    started = time.monotonic()
    tool_calls: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
    finish = {"reason": "error", "steps": 0, "text": ""}

    def on_event(event: Event) -> None:
        if output_format == "stream-json":
            print(json.dumps(event_to_dict(event)), flush=True)
        elif output_format == "text" and isinstance(event, (Notice, ToolStarted)):
            line = f"[hx] {event.text}" if isinstance(event, Notice) else f"→ {event.call.name}({event.call.arguments})"
            print(line, file=sys.stderr, flush=True)

        if isinstance(event, ToolFinished):
            tool_calls.append({"name": event.call.name, "arguments": event.call.arguments, "is_error": event.is_error})
        elif isinstance(event, StepFinished):
            usage["input_tokens"] += event.usage.prompt_tokens
            usage["output_tokens"] += event.usage.completion_tokens
            usage["model_calls"] += 1
        elif isinstance(event, TurnFinished):
            finish.update(reason=event.reason, steps=event.steps, text=event.text)
        elif isinstance(event, TextDelta):
            pass  # the final text arrives with TurnFinished

    error = None
    try:
        agent.run(prompt, on_event=on_event)
    except ModelError as e:
        error = str(e)

    summary = {
        "result": finish["text"],
        "finish_reason": "error" if error else finish["reason"],
        "error": error,
        "steps": finish["steps"],
        "duration_s": round(time.monotonic() - started, 2),
        "usage": usage,
        "tool_calls": tool_calls,
        "session_id": agent.session.id if agent.session else None,
    }
    if output_format == "text":
        if error:
            print(f"[model error] {error}", file=sys.stderr)
        print(finish["text"])
    else:
        print(json.dumps(summary if output_format == "json" else {"type": "result", **summary}), flush=True)
    return 0 if summary["finish_reason"] == "done" else 1
