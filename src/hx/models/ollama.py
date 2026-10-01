"""Ollama client (/api/chat, streaming NDJSON)."""

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from hx.config import Settings
from hx.messages import Message, ToolCall
from hx.models.base import ModelError, ModelResponse, TextCallback, Usage, http_error


def to_wire(m: Message) -> dict[str, Any]:
    """Our Message -> Ollama's message JSON."""
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.role == "tool" and m.name:
        out["tool_name"] = m.name
    return out


def parse_tool_calls(raw: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for item in raw:
        fn = item.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):  # some models send arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append(ToolCall(id=item.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name", ""), arguments=args))
    return calls


def keep_alive_value(text: str) -> str | int:
    """Ollama reads keep_alive as a duration string ("30m") or a number of seconds; "-1" alone is not a duration."""
    try:
        return int(text)
    except ValueError:
        return text


class OllamaClient:
    def __init__(self, settings: Settings | None = None, timeout: float = 600):
        self.settings = settings or Settings.from_env()
        self.name = f"ollama:{self.settings.model}"
        self.timeout = timeout

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> ModelResponse:
        s = self.settings
        body: dict[str, Any] = {
            "model": s.model,
            "messages": [to_wire(m) for m in messages],
            "stream": True,
            "think": s.think,
            "options": {"num_ctx": s.num_ctx, "temperature": s.temperature},
            "keep_alive": keep_alive_value(s.keep_alive),
        }
        if tools:
            body["tools"] = tools

        req = urllib.request.Request(
            f"{s.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        content, thinking, calls = [], [], []
        final: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:  # one JSON object per line
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise ModelError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message", {})
                    if msg.get("content"):
                        content.append(msg["content"])
                        if on_text:
                            on_text(msg["content"])
                    if msg.get("thinking"):
                        thinking.append(msg["thinking"])
                    if msg.get("tool_calls"):
                        calls.extend(parse_tool_calls(msg["tool_calls"]))
                    if chunk.get("done"):
                        final = chunk
        except urllib.error.HTTPError as e:
            raise http_error(e.code, "Ollama: " + e.read().decode(errors="replace"), e.headers.get("Retry-After")) from e
        except urllib.error.URLError as e:
            raise ModelError(f"Can't reach Ollama at {s.host} ({e.reason}). Is it running? Try: ollama serve",
                             retryable=True) from e
        except (TimeoutError, ConnectionError) as e:
            raise ModelError(f"Ollama connection failed: {e}", retryable=True) from e

        return ModelResponse(
            message=Message("assistant", "".join(content), tool_calls=calls, thinking="".join(thinking)),
            usage=Usage(
                prompt_tokens=final.get("prompt_eval_count", 0),
                completion_tokens=final.get("eval_count", 0),
                duration_s=time.monotonic() - started,
            ),
            stop_reason=final.get("done_reason", ""),
        )


if __name__ == "__main__":
    # Quick manual check: python -m hx.models.ollama "Say hello in Nepali"
    import sys

    from hx.messages import user

    client = OllamaClient()
    r = client.chat([user(" ".join(sys.argv[1:]) or "Say hello.")], on_text=lambda t: print(t, end="", flush=True))
    print(f"\n[{client.name}] {r.usage.prompt_tokens} prompt + {r.usage.completion_tokens} output tokens, {r.usage.duration_s:.1f}s")
