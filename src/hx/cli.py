"""Command-line entry point: an interactive REPL around the agent."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from hx import __version__
from hx.agent import Agent, make_child_agent
from hx.checkpoints import CheckpointError, checkpoints_for
from hx.commands import Commands, Prompt
from hx.context import ContextManager
from hx.config import Settings
from hx.hooks import load_hooks
from hx.ui import make_ui
from hx.models import ModelError
from hx.models.ollama import OllamaClient
from hx.permissions import MODES, Approval, Decision, load_policy
from hx.prompt import build_system_prompt
from hx.session import Session, find_session, list_sessions
from hx.mcp import connect_all, load_servers
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


def confirm_project_config(kind: str, config: dict) -> bool:
    """Project hooks / MCP servers run commands with your permissions and came with the repo: show them, ask once."""
    if kind == "hooks":
        print("\n  ⚠ This project defines hooks (shell commands hx would run automatically):")
        for event, entries in config.items():
            for e in entries:
                matcher = f" [{e['matcher']}]" if e.get("matcher") else ""
                print(f"    {event}{matcher}: {e['command']}")
    else:
        print("\n  ⚠ This project defines MCP servers (programs hx would start):")
        for name, cfg in config.items():
            print(f"    {name}: {cfg.get('command')} {' '.join(cfg.get('args', []))}")
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


def enable_history() -> None:
    """Arrow-key history across sessions, via the standard library's readline."""
    try:
        import atexit
        import readline
    except ImportError:
        return
    path = Path.home() / ".hx" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(path)
    except OSError:
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: readline.write_history_file(path))


@dataclass
class ReplState:
    agent: Agent
    cwd: Path
    exit: bool = False


def repl(agent: Agent) -> None:
    mode = agent.ctx.permissions.mode if agent.ctx.permissions else "none"
    sandbox = "on" if agent.ctx.sandbox and agent.ctx.sandbox.enabled else "off"
    print(f"hx {__version__} · {agent.model.name} · permissions: {mode} · sandbox: {sandbox}")
    print("/help for commands · # <fact> to remember something")
    state = ReplState(agent, agent.ctx.cwd)
    commands = Commands(state)
    ui = make_ui()
    enable_history()
    while not state.exit:
        try:
            text = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.startswith(("/", "#")):
            try:
                out = commands.handle(text)
            except CheckpointError as e:
                print(f"[failed] {e}", file=sys.stderr)
                continue
            if not isinstance(out, Prompt):
                if out:
                    print(out)
                continue
            text = out.text  # a custom command expanded into a prompt for the agent
        try:
            agent.run(text, on_event=ui)
        except ModelError as e:
            print(f"\n[model error] {e}", file=sys.stderr)
        finally:
            if hasattr(ui, "cleanup"):
                ui.cleanup()


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
    ctx.hooks, hook_notes = load_hooks(ctx.cwd, confirm=confirm_project_config)
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
    servers, mcp_notes = load_servers(ctx.cwd, confirm=confirm_project_config)
    mcp_clients, mcp_tools, more_notes = connect_all(ctx.cwd, servers)
    for t in mcp_tools:
        tools.register(t)
    for note in mcp_notes + more_notes:
        print(f"[hx] {note}", file=sys.stderr)
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
    try:
        repl(agent)
    finally:
        for client in mcp_clients:
            client.close()
    return 0
