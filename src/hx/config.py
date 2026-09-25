"""Settings, read from environment variables with sensible defaults."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model: str = "qwen3:14b"
    host: str = "http://localhost:11434"
    # Ollama's context window. Bigger fits more conversation but uses more memory (KV cache).
    num_ctx: int = 16384
    # qwen3 can "think" before answering. Off by default: much faster, and tool use still works.
    think: bool = False
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.model = os.environ.get("HX_MODEL", s.model)
        s.host = os.environ.get("OLLAMA_HOST", s.host)
        if not s.host.startswith("http"):
            s.host = "http://" + s.host
        s.num_ctx = int(os.environ.get("HX_NUM_CTX", s.num_ctx))
        s.think = os.environ.get("HX_THINK", "0") == "1"
        return s
