"""Keeps the conversation inside the model's context window.

Ollama silently drops the start of a prompt that's too long, so we have to stay under the limit ourselves.
"""

import json
from dataclasses import dataclass

from hx.messages import Message

CHARS_PER_TOKEN = 3.5  # a starting guess; code tokenizes denser than English prose
PER_MESSAGE_OVERHEAD = 4  # role markers and separators in the chat template
CLEARED = "[old tool result cleared to save context"


def raw_estimate(m: Message) -> int:
    chars = len(m.content)
    if m.tool_calls:
        chars += sum(len(c.name) + len(json.dumps(c.arguments)) for c in m.tool_calls)
    return int(chars / CHARS_PER_TOKEN) + PER_MESSAGE_OVERHEAD


@dataclass
class ContextManager:
    window: int = 16384  # the model's context size (num_ctx)
    reserve: int = 2048  # room left for the model's reply
    max_tool_result_share: float = 0.25  # one tool result may use at most this share of the window
    clear_at: float = 0.70  # start clearing old tool results when the prompt reaches 70% of the budget...
    clear_to: float = 0.50  # ...and clear until it's back under 50%
    compact_at: float = 0.85  # still this full after clearing? summarize the conversation (compaction.py)
    keep_recent: int = 4  # never clear the most recent N tool results; the model is still using them
    ratio: float = 1.0  # calibration: real tokens / estimated tokens
    overhead: int = 0  # estimated tokens outside the messages, e.g. the tool schemas sent with every request

    def set_tools(self, schemas: list[dict]) -> None:
        self.overhead = int(len(json.dumps(schemas)) / CHARS_PER_TOKEN) if schemas else 0

    @property
    def budget(self) -> int:
        return self.window - self.reserve

    def estimate(self, messages: list[Message]) -> int:
        return int((sum(raw_estimate(m) for m in messages) + self.overhead) * self.ratio)

    def calibrate(self, messages: list[Message], actual_prompt_tokens: int) -> None:
        """Learn the real chars-per-token from the model's own count. Smoothed and clamped, so one odd
        reading can't swing it."""
        estimated = sum(raw_estimate(m) for m in messages) + self.overhead
        if estimated > 0 and actual_prompt_tokens > 0:
            observed = min(2.0, max(0.5, actual_prompt_tokens / estimated))
            self.ratio = 0.7 * self.ratio + 0.3 * observed

    def fraction_used(self, messages: list[Message]) -> float:
        return self.estimate(messages) / self.budget

    # -- 1. per-result cap

    def truncate_tool_output(self, text: str) -> str:
        max_tokens = int(self.window * self.max_tool_result_share)
        max_chars = int(max_tokens * CHARS_PER_TOKEN / self.ratio)
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        omitted = len(text) - 2 * half
        return (
            text[:half]
            + f"\n\n... [{omitted} characters omitted to fit the context window; "
            "narrow the request (offset/limit, a more specific pattern) to see this part] ...\n\n"
            + text[-half:]
        )

    # -- 2. clear old tool results

    def clear_old_results(self, messages: list[Message], keep_recent: int | None = None, force: bool = False) -> list[str]:
        """Stub out the oldest tool results until the prompt fits under clear_to. Returns notes for the user.
        force=True ignores the clear_at threshold (used when escalating, see Agent.fit_context)."""
        if not force and self.fraction_used(messages) < self.clear_at:
            return []
        keep = self.keep_recent if keep_recent is None else keep_recent
        tool_indexes = [i for i, m in enumerate(messages) if m.role == "tool" and not m.content.startswith(CLEARED)]
        candidates = tool_indexes[: max(0, len(tool_indexes) - keep)]
        cleared = []
        for i in candidates:
            if self.fraction_used(messages) < self.clear_to:
                break
            m = messages[i]
            m.content = f"{CLEARED}: {m.name} returned {len(m.content)} characters; run it again if you need it]"
            cleared.append(m.name or "tool")
        if not cleared:
            return []
        return [f"cleared {len(cleared)} old tool result(s) to stay within the context window "
                f"(now ~{self.fraction_used(messages):.0%} used)"]

    def shrink_last_result(self, messages: list[Message]) -> list[str]:
        """Last resort: cut the newest tool result down until the prompt fits the budget."""
        for m in reversed(messages):
            if m.role == "tool" and not m.content.startswith(CLEARED):
                excess_tokens = self.estimate(messages) - int(self.budget * 0.95)
                if excess_tokens <= 0:
                    return []
                keep_chars = max(200, len(m.content) - int(excess_tokens * CHARS_PER_TOKEN * self.ratio) - 200)
                if keep_chars >= len(m.content):
                    return []
                half = keep_chars // 2
                m.content = (m.content[:half] + f"\n... [{len(m.content) - keep_chars} characters cut: context window full; "
                             "ask for less at a time] ...\n" + m.content[-half:])
                return ["context still over budget: shortened the latest tool result"]
        return []
