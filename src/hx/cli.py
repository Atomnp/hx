"""Command-line entry point: an interactive REPL around the agent."""

import argparse
import sys

from hx import __version__
from hx.agent import Agent
from hx.config import Settings
from hx.events import Event, StepFinished, TextDelta, ToolFinished, ToolStarted
from hx.models import ModelError
from hx.models.ollama import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hx", description="A coding-agent harness.")
    parser.add_argument("--version", action="version", version=f"hx {__version__}")
    parser.add_argument("--model", help="Ollama model name (default: $HX_MODEL or qwen3:14b)")
    return parser


def print_event(event: Event) -> None:
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolStarted):
        print(f"\n  → {event.call.name}({event.call.arguments})", flush=True)
    elif isinstance(event, ToolFinished):
        mark = "✗" if event.is_error else "✓"
        print(f"  {mark} {event.result[:200]}", flush=True)
    elif isinstance(event, StepFinished):
        u = event.usage
        print(f"\n  [{u.prompt_tokens} in / {u.completion_tokens} out, {u.duration_s:.1f}s]", file=sys.stderr, flush=True)


def repl(agent: Agent) -> None:
    print(f"hx {__version__} · {agent.model.name} · /exit to quit")
    while True:
        try:
            text = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        try:
            agent.run(text, on_event=print_event)
        except ModelError as e:
            print(f"\n[model error] {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    repl(Agent(OllamaClient(settings)))
    return 0
