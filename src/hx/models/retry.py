"""Retries for transient model errors: exponential backoff with jitter, honoring Retry-After.

No retry once text has been streamed, since the user has already seen part of the answer.
"""

import random
import time
from typing import Callable

from hx.models.base import ModelClient, ModelError, ModelResponse


class RetryingClient:
    def __init__(self, inner: ModelClient, attempts: int = 4, base_delay: float = 1.0, max_delay: float = 30.0,
                 on_retry: Callable[[str], None] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.name = inner.name
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.on_retry = on_retry or (lambda msg: None)
        self.sleep = sleep

    def chat(self, messages, tools=None, on_text=None) -> ModelResponse:
        streamed = False

        def track(text: str) -> None:
            nonlocal streamed
            streamed = True
            if on_text:
                on_text(text)

        for attempt in range(1, self.attempts + 1):
            try:
                return self.inner.chat(messages, tools=tools, on_text=track)
            except ModelError as e:
                if not e.retryable or streamed or attempt == self.attempts:
                    raise
                delay = e.retry_after if e.retry_after is not None else min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
                delay *= 1 if e.retry_after is not None else random.uniform(0.5, 1.0)
                self.on_retry(f"model call failed ({e}); retrying in {delay:.1f}s (attempt {attempt + 1}/{self.attempts})")
                self.sleep(delay)
        raise AssertionError("unreachable")
