"""Command-line entry point: an interactive REPL around the agent."""

import argparse
import sys
from pathlib import Path

from hx import __version__
from hx.agent import Agent, make_child_agent
from hx.checkpoints import CheckpointError, checkpoints_for
from hx.context import ContextManager
from hx.config import Settings
from hx.hooks import load_hooks
from hx.events import Event, Notice, StepFinished, TextDelta, ToolFinished, ToolStarted, TurnFinished
from hx.models import ModelError
from hx.models.ollama import OllamaClient
from hx.permissions import MODES, Approval, Decision, load_policy
from hx.prompt import build_system_prompt
from hx.session import Session, find_session, list_sessions
from hx.skills import SkillTool, discover
from hx.subagents import Task, load_specs
from hx.sandbox import load_sandbox
from hx.tools import Tool, ToolContext
from hx.tools.builtin import default_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hx", description="A coding-agent harness.")
    parser.add_argument("--version", action="version", version=f"hx {__version__}")
    parser.add_argument("--model", help="Ollama model name (default: $HX_MODEL or qwen3:14b)")
    parser.add_argument("--mode", choices=MODES, help="permission mode (default: from .hx/settings.json, else ask)")
    parser.add_argument("--no-sandbox", action="store_true", help="run shell commands without the OS sandbox")
    parser.add_argument("--no-checkpoints", action="store_true", help="don't snapshot the workspace before each turn")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="ID",
                        help="continue a saved session (the latest one, or the given id / id prefix)")
    parser.add_argument("--sessions", action="store_true", help="list saved sessions for this workspace and exit")
    return parser


def confirm_project_hooks(hooks) -> bool:
    """Project hooks run with your permissions and came with the repo: show them and ask once."""
    print("\n  ⚠ This project defines hooks (shell commands hx would run automatically):")
    for h in hooks:
        print(f"    {h.event}{f' [{h.matcher}]' if h.matcher else ''}: {h.command}")
    try:
        return input("    trust and run them? [y/N] › ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


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
        warnings = [line for line in event.result[200:].splitlines() if line.startswith("⚠")]
        for line in warnings:  # don't let a preview cut hide a warning the model received
            print(f"  {line}", flush=True)
    elif isinstance(event, TurnFinished) and event.reason == "max_steps":
        print(f"\n  [hx] stopped after {event.steps} steps without finishing. Say 'continue' to keep going.", file=sys.stderr)
    elif isinstance(event, Notice):
        print(f"  [hx] {event.text}", file=sys.stderr, flush=True)
    elif isinstance(event, StepFinished):
        u = event.usage
        print(f"\n  [{u.prompt_tokens} in / {u.completion_tokens} out, {u.duration_s:.1f}s, context {event.context_used:.0%}]",
              file=sys.stderr, flush=True)


def repl(agent: Agent) -> None:
    mode = agent.ctx.permissions.mode if agent.ctx.permissions else "none"
    sandbox = "on" if agent.ctx.sandbox and agent.ctx.sandbox.enabled else "off"
    print(f"hx {__version__} · {agent.model.name} · permissions: {mode} · sandbox: {sandbox}")
    print("commands: /undo  /checkpoints  /compact  /exit")
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
        if text == "/undo":
            try:
                print(agent.undo())
            except CheckpointError as e:
                print(f"[undo failed] {e}", file=sys.stderr)
            continue
        if text == "/compact":
            print(agent.compact() or "Nothing to compact yet.")
            continue
        if text == "/checkpoints":
            if not agent.checkpoints:
                print("Checkpoints are off.")
            for c in agent.checkpoints.list() if agent.checkpoints else []:
                print(f"  {c.sha[:8]}  {c.age:>16}  {c.label}")
            continue
        try:
            agent.run(text, on_event=print_event)
        except ModelError as e:
            print(f"\n[model error] {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sessions:
        for info in list_sessions(Path.cwd()):
            print(f"  {info.id}  {info.created}  {info.messages:3} msgs  {info.title}")
        return 0
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    ctx = ToolContext()
    ctx.permissions = load_policy(ctx.cwd, args.mode)
    ctx.approve = ask_user
    ctx.sandbox = load_sandbox(ctx.cwd, enabled=False if args.no_sandbox else None)
    ctx.hooks, hook_notes = load_hooks(ctx.cwd, confirm=confirm_project_hooks)
    for note in hook_notes:
        print(f"[hx] {note}", file=sys.stderr)
    checkpoints = None if args.no_checkpoints else checkpoints_for(ctx.cwd)
    skills = discover(ctx.cwd)
    ctx.read_roots = [s.path for s in skills]
    system_prompt = build_system_prompt(ctx, skills)
    if ctx.hooks:
        start = ctx.hooks.run("SessionStart", {})
        if start.output:
            system_prompt += f"\n\n# Context from SessionStart hooks\n{start.output}"
    model = OllamaClient(settings)
    tools = default_tools()
    if skills:
        tools.register(SkillTool(skills))
    tools.register(Task(load_specs(ctx.cwd), tools, model, make_child_agent))
    agent = Agent(
        model,
        tools=tools,
        ctx=ctx,
        system_prompt=system_prompt,
        checkpoints=checkpoints,
        context=ContextManager(window=settings.num_ctx),
    )
    if args.resume:
        saved = find_session(ctx.cwd, None if args.resume == "latest" else args.resume)
        if saved is None:
            print(f"No saved session matches {args.resume!r} here. Try: hx --sessions", file=sys.stderr)
            return 1
        turns = agent.resume(saved, system_prompt)
        print(f"Resumed session {saved.id} ({turns} earlier request(s)).")
        last_user = next((m.content for m in reversed(agent.messages) if m.role == "user"), "")
        last_reply = next((m.content for m in reversed(agent.messages) if m.role == "assistant" and m.content), "")
        if last_user:
            print(f"  last request: {last_user[:120]}\n  last reply:   {last_reply[:120]}")
    else:
        agent.session = Session.create(ctx.cwd, settings.model)
        agent.session.message(agent.messages[0])
    repl(agent)
    return 0
