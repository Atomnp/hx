"""Compaction: replace older messages with a model-written summary, keeping recent ones as they are."""

import json

from hx.messages import Message, system, user
from hx.models.base import ModelClient

SUMMARY_HEADER = "[Summary of the conversation so far; older messages were compacted to save context]"

SUMMARIZER_PROMPT = """You are summarizing part of a coding session so the assistant can continue it with a fresh \
context window.

Keep specific identifiers: file paths, function/class/constant names and what they do, commands, error messages. \
Later questions depend on exactly these details.

Write a concise, factual summary (at most ~500 words) with these sections:

1. User requests: what the user asked for (quote them where the wording matters).
2. Progress: what has been done so far; files read, created or modified (with paths) and why.
3. Key facts: decisions, constraints, project conventions, errors and how they were fixed, commands that work \
(for example how to run the tests).
4. Current state: where the work stands right now.
5. Next steps: what still needs to be done, if anything.

Use only information from the transcript. No preamble."""

ACK = "Understood, continuing from the summary."
MAX_SUMMARY_CHARS = 8000  # once the rolling summary is longer than this, the model merges it into one
MERGE_PROMPT = """Merge these consecutive summaries of one coding session into a single summary with the same five \
sections. Keep every specific identifier (file paths, function/class/constant names, commands, errors) and every user \
request. Drop only repetition."""


def render_transcript(messages: list[Message], budget_chars: int = 16_000) -> str:
    """Messages as plain text for the summarizer. Tool results share whatever budget is left after
    everything else, so the summarizer sees as much of the actual data as the window allows."""
    results = [m for m in messages if m.role == "tool"]
    other_chars = sum(len(m.content) + 80 for m in messages if m.role != "tool")
    per_result = max(300, (budget_chars - other_chars) // max(1, len(results)))
    lines = []
    for m in messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            if m.content and m.content != ACK:
                lines.append(f"ASSISTANT: {m.content}")
            for c in m.tool_calls:
                lines.append(f"ASSISTANT CALLED {c.name}({json.dumps(c.arguments)[:300]})")
        elif m.role == "tool":
            body = m.content if len(m.content) <= per_result else m.content[:per_result] + f" ... [{len(m.content)} chars total]"
            lines.append(f"RESULT of {m.name}: {body}")
    return "\n".join(lines)


def split_point(messages: list[Message], keep_recent: int) -> int:
    """Index where the kept tail starts. Never splits an assistant tool call from its results:
    the tail must start at a user message or at an assistant message."""
    i = max(1, len(messages) - keep_recent)
    while i > 1 and messages[i].role == "tool":
        i -= 1  # step back to the assistant message that made these calls, so call and results stay together
    return i


def is_summary(m: Message) -> bool:
    return m.role == "user" and m.content.startswith(SUMMARY_HEADER)


def compact(
    model: ModelClient, messages: list[Message], keep_recent: int = 6, budget_chars: int = 16_000
) -> tuple[list[Message], str]:
    """Return (new message list, summary). messages[0] must be the system prompt; it's always kept.

    Earlier summaries are carried forward VERBATIM (code guarantees they survive; we don't rely on the
    model to copy them). Only the messages since then are summarized and appended. If the rolling summary
    grows past MAX_SUMMARY_CHARS, the model merges it into one."""
    cut = split_point(messages, keep_recent)
    old, tail = messages[1:cut], messages[cut:]
    earlier = [m.content[len(SUMMARY_HEADER):].strip() for m in old if is_summary(m)]
    fresh = [m for m in old if not is_summary(m) and m.content != ACK]
    if not fresh:
        return messages, ""

    response = model.chat([system(SUMMARIZER_PROMPT), user(render_transcript(fresh, budget_chars))])
    parts = earlier + [response.message.content.strip()]
    summary = "\n\n--- then ---\n\n".join(parts)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = model.chat([system(MERGE_PROMPT), user(summary)]).message.content.strip()

    compacted = [messages[0], user(f"{SUMMARY_HEADER}\n\n{summary}")]
    if tail and tail[0].role == "user":
        # Two user messages in a row confuse some chat templates; acknowledge the summary in between.
        compacted.append(Message("assistant", ACK))
    return compacted + tail, summary
