"""Fixes common tool-call mistakes from smaller models: calls written as text, mistyped arguments,
unfilled placeholders, and the same call repeated over and over.
"""

import json
import re
from collections import deque

from hx.messages import ToolCall

# ---------------------------------------------------------------- 1. tool calls written as text

# Formats models use when they don't call tools natively:
TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)  # Qwen/Hermes chat template
FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)  # markdown code block


def as_tool_call(obj: object, known: set[str], n: int) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name") or fn.get("tool")
    if name not in known:
        return None
    args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCall(f"text_call_{n}", name, args)


def extract_text_tool_calls(content: str, known: set[str]) -> tuple[list[ToolCall], str]:
    """Find tool calls written into `content`. Returns (calls, content with those calls removed)."""
    spans: list[tuple[int, int, str]] = []
    for pattern in (TAGGED, FENCED):
        spans += [(m.start(), m.end(), m.group(1)) for m in pattern.finditer(content)]

    if not spans:  # bare JSON objects anywhere in the text
        decoder = json.JSONDecoder()
        i = 0
        while (i := content.find("{", i)) != -1:
            try:
                _, end = decoder.raw_decode(content, i)
            except json.JSONDecodeError:
                i += 1
                continue
            spans.append((i, end, content[i:end]))
            i = end

    calls, kept = [], []
    for start, end, text in spans:
        try:
            call = as_tool_call(json.loads(text), known, len(calls) + 1)
        except json.JSONDecodeError:
            call = None
        if call:
            calls.append(call)
            kept.append((start, end))

    cleaned = content
    for start, end in sorted(kept, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    return calls, cleaned.strip()


# ---------------------------------------------------------------- 2. argument repair

INT = re.compile(r"^-?\d+$")
FLOAT = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")


def coerce(value, expected: str | None):
    """Convert `value` to the schema type if it's an unambiguous mismatch; otherwise return it unchanged."""
    if isinstance(value, str):
        s = value.strip()
        if expected == "integer" and INT.match(s):
            return int(s)
        if expected == "number" and FLOAT.match(s):
            return float(s)
        if expected == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
        if expected in ("array", "object") and s[:1] in "[{":
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list if expected == "array" else dict):
                return parsed
    if expected == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# Arguments that carry code or file text. Only these get the escaped-newline repair: in a shell command,
# a literal \n is often intentional (printf, sed).
CODE_ARGS = {"content", "old_string", "new_string"}


def unescape_code(value: str) -> str | None:
    """'def f():\\n    return 1' (no real newlines, several literal backslash-n) -> real newlines.
    A single literal \n is left alone: it's probably inside a string literal like print("a\\nb")."""
    if "\n" in value or value.count("\\n") < 2:
        return None
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


def repair_arguments(args: dict, schema: dict) -> tuple[dict, list[str]]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    notes: list[str] = []
    out = dict(args)

    # {"arguments": {...real args...}}: the model wrapped its arguments one level too deep.
    for wrapper in ("arguments", "parameters", "args", "input"):
        if set(out) == {wrapper} and isinstance(out[wrapper], dict) and wrapper not in props:
            out = dict(out[wrapper])
            notes.append(f"unwrapped arguments from '{wrapper}'")
            break

    for key, value in list(out.items()):
        if value is None and key not in required:
            del out[key]
            notes.append(f"dropped null optional argument '{key}'")
            continue
        if key in CODE_ARGS and isinstance(value, str) and (unescaped := unescape_code(value)) is not None:
            out[key] = unescaped
            notes.append(f"'{key}' contained escaped newlines (\\n) instead of real ones; unescaped it")
            continue
        expected = props.get(key, {}).get("type")
        fixed = coerce(value, expected)
        if type(fixed) is not type(value) or fixed != value:
            out[key] = fixed
            notes.append(f"converted '{key}' {value!r} to {expected} {fixed!r}")
    return out, notes


# ---------------------------------------------------------------- 3. placeholders

PLACEHOLDER = [
    re.compile(r"^<[A-Za-z]+[_ \-][A-Za-z0-9_ \-]*>$"),  # <file_path>, <person id>
    re.compile(r"^\{\{\s*[\w.]+\s*\}\}$"),  # {{path}}
    re.compile(r"^(\.\.\.|…)$"),
]


def find_placeholders(args: dict) -> list[str]:
    return [
        f"{k}={v!r}"
        for k, v in args.items()
        if isinstance(v, str) and len(v) <= 60 and any(p.match(v.strip()) for p in PLACEHOLDER)
    ]


# ---------------------------------------------------------------- 4. loop guard


class LoopGuard:
    """Detects the same tool call (name + identical arguments) repeated within the last few calls."""

    def __init__(self, window: int = 6, threshold: int = 3):
        self.recent: deque[str] = deque(maxlen=window)
        self.threshold = threshold

    def check(self, call: ToolCall) -> str | None:
        sig = call.name + json.dumps(call.arguments, sort_keys=True)
        self.recent.append(sig)
        n = self.recent.count(sig)
        if n >= self.threshold:
            return (
                f"You have made this exact {call.name} call {n} times recently and it returns the same result. "
                "Don't repeat it: try a different approach, or tell the user what's blocking you."
            )
        return None
