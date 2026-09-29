"""Client for OpenAI-compatible Chat Completions APIs, streamed as SSE.

Tool call arguments arrive as string fragments keyed by index, so they're joined and parsed at the end.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any

from hx.messages import Message, ToolCall
from hx.models.base import ModelError, ModelResponse, TextCallback, Usage, http_error


def to_wire(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in m.tool_calls
        ]
    return out


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 600,
                 extra_headers: dict | None = None, temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.temperature = temperature
        self.name = f"openai-compat:{model}"

    def chat(self, messages, tools=None, on_text: TextCallback | None = None) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_wire(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers)

        started = time.monotonic()
        content: list[str] = []
        calls: dict[int, dict] = {}  # index -> {"id", "name", "arguments": [fragments]}
        usage: dict = {}
        finish = ""
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue  # blank keep-alive lines, comments, "event:" lines
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise ModelError(f"server error: {chunk['error']}")
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            content.append(delta["content"])
                            if on_text:
                                on_text(delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": []})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function", {})
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"].append(fn.get("arguments") or "")
                        finish = choice.get("finish_reason") or finish
        except urllib.error.HTTPError as e:
            raise http_error(e.code, e.read().decode(errors="replace"), e.headers.get("Retry-After")) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Can't reach {self.base_url} ({e.reason})", retryable=True) from e
        except (TimeoutError, ConnectionError) as e:
            raise ModelError(f"connection to {self.base_url} failed: {e}", retryable=True) from e

        tool_calls = []
        for index in sorted(calls):
            slot = calls[index]
            raw_args = "".join(slot["arguments"]) or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}  # the validator will report it; repair may fix it
            tool_calls.append(ToolCall(slot["id"] or f"call_{index}", slot["name"], args))

        return ModelResponse(
            Message("assistant", "".join(content), tool_calls=tool_calls),
            Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), time.monotonic() - started),
            finish,
        )
