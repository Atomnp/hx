"""Command-line entry point: an interactive REPL around the agent."""

import argparse
import sys

from hx import __version__
from hx.agent import Agent
from hx.config import Settings
from hx.events import Event, Notice, StepFinished, TextDelta, ToolFinished, ToolStarted
from hx.models import ModelError
from hx.models.ollama import OllamaClient
from hx.permissions import MODES, Approval, Decision, load_policy
from hx.sandbox import load_sandbox
from hx.tools import Tool, ToolContext
from hx.tools.builtin import default_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hx", description="A coding-agent harness.")
    parser.add_argument("--version", action="version", version=f"hx {__version__}")
    parser.add_argument("--model", help="Ollama model name (default: $HX_MODEL or qwen3:14b)")
    parser.add_argument("--mode", choices=MODES, help="permission mode (default: from .hx/settings.json, else ask)")
    parser.add_argument("--no-sandbox", action="store_true", help="run shell commands without the OS sandbox")
    return parser


def ask_user(tool: Tool, args: dict, decision: Decision, rule: str) -> Approval:
    """Terminal approval prompt. 'n <text>' denies with instructions the model will see."""
    subject = args.get("command") or args.get("path") or ""
    print(f"\n  ⚠ {tool.name}: {subject}\n    ({decision.reason})")
    try:
        answer = input(f"    allow? [y]es / [n]o (+ instructions) / [a]lways {rule} › ").strip()
    except (EOFError, KeyboardInterrupt):
        return Approval(False)
    if answer.lower() in ("y", "yes"):
        return Approval(True)
    if answer.lower() in ("a", "always"):
        return Approval(True, remember=True)
    feedback = answer[1:].strip(" :-") if answer[:1].lower() == "n" else answer
    return Approval(False, feedback=feedback)


def print_event(event: Event) -> None:
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolStarted):
        print(f"\n  → {event.call.name}({event.call.arguments})", flush=True)
    elif isinstance(event, ToolFinished):
        mark = "✗" if event.is_error else "✓"
        print(f"  {mark} {event.result[:200]}", flush=True)
    elif isinstance(event, Notice):
        print(f"  [hx] {event.text}", file=sys.stderr, flush=True)
    elif isinstance(event, StepFinished):
        u = event.usage
        print(f"\n  [{u.prompt_tokens} in / {u.completion_tokens} out, {u.duration_s:.1f}s]", file=sys.stderr, flush=True)


def repl(agent: Agent) -> None:
    mode = agent.ctx.permissions.mode if agent.ctx.permissions else "none"
    sandbox = "on" if agent.ctx.sandbox and agent.ctx.sandbox.enabled else "off"
    print(f"hx {__version__} · {agent.model.name} · permissions: {mode} · sandbox: {sandbox} · /exit to quit")
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
    ctx = ToolContext()
    ctx.permissions = load_policy(ctx.cwd, args.mode)
    ctx.approve = ask_user
    ctx.sandbox = load_sandbox(ctx.cwd, enabled=False if args.no_sandbox else None)
    repl(Agent(OllamaClient(settings), tools=default_tools(), ctx=ctx))
    return 0
