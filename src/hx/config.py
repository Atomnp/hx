"""Settings, read from environment variables with sensible defaults."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    provider: str = "ollama"  # "ollama" (native API) or "openai" (any OpenAI-compatible server)
    model: str = "qwen3:14b"
    host: str = "http://localhost:11434"  # Ollama
    base_url: str = "https://api.openai.com/v1"  # OpenAI-compatible servers
    api_key_env: str = "OPENAI_API_KEY"  # the NAME of the env var holding the key; the key itself is never stored
    # Ollama's context window. Bigger fits more conversation but uses more memory (KV cache).
    num_ctx: int = 16384
    # qwen3 can "think" before answering. Off by default: much faster, and tool use still works.
    think: bool = False
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.provider = os.environ.get("HX_PROVIDER", s.provider)
        s.base_url = os.environ.get("HX_BASE_URL", s.base_url)
        s.api_key_env = os.environ.get("HX_API_KEY_ENV", s.api_key_env)
        s.model = os.environ.get("HX_MODEL", s.model)
        s.host = os.environ.get("OLLAMA_HOST", s.host)
        if not s.host.startswith("http"):
            s.host = "http://" + s.host
        s.num_ctx = int(os.environ.get("HX_NUM_CTX", s.num_ctx))
        s.think = os.environ.get("HX_THINK", "0") == "1"
        return s
