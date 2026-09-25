"""Minimal JSON Schema validation for tool arguments (type, properties, required, enum, items,
additionalProperties).
"""

from typing import Any

TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Return a list of problems ("" means the whole arguments object). Empty list = valid."""
    where = path or "arguments"
    problems: list[str] = []

    expected = schema.get("type")
    if expected:
        py_type = TYPES[expected]
        # bool is a subclass of int in Python; don't let True pass as an integer.
        if not isinstance(value, py_type) or (expected in ("integer", "number") and isinstance(value, bool)):
            return [f"{where}: expected {expected}, got {type(value).__name__} ({value!r})"]

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{where}: must be one of {schema['enum']}, got {value!r}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{path + '.' if path else ''}{name}: required field missing")
        for name, item in value.items():
            if name in props:
                problems += validate(item, props[name], f"{path + '.' if path else ''}{name}")
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path + '.' if path else ''}{name}: unknown field (allowed: {sorted(props)})")

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problems += validate(item, schema["items"], f"{where}[{i}]")

    return problems
