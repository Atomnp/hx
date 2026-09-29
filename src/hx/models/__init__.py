import os
from typing import Callable

from hx.config import Settings
from hx.models.base import ModelClient, ModelError, ModelResponse, Usage

__all__ = ["ModelClient", "ModelError", "ModelResponse", "Usage", "make_client"]


def make_client(settings: Settings, on_retry: Callable[[str], None] | None = None) -> ModelClient:
    """Build the configured provider's client, wrapped in retries."""
    from hx.models.ollama import OllamaClient
    from hx.models.openai_compat import OpenAICompatClient
    from hx.models.retry import RetryingClient

    if settings.provider == "ollama":
        inner = OllamaClient(settings)
    elif settings.provider == "openai":
        inner = OpenAICompatClient(settings.base_url, settings.model, os.environ.get(settings.api_key_env),
                                   temperature=settings.temperature)
    else:
        raise ValueError(f"unknown provider {settings.provider!r}; use 'ollama' or 'openai'")
    return RetryingClient(inner, on_retry=on_retry)
