"""The bash tool. Uses the C++ hx-exec runner when it's built, otherwise a Python fallback."""

import json
import os
import signal
import subprocess
import time

from hx.native import find_binary
from hx.sandbox import looks_blocked
from hx.tools.base import Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT = 30_000  # bytes of output the model sees (head + tail)
MAX_FILE_MB = 1024  # RLIMIT_FSIZE: no single file written bigger than this

# Make tools behave non-interactively: no pagers, no color codes, no credential prompts.
QUIET_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONUNBUFFERED": "1",
}


def exec_binary() -> str | None:
    return find_binary("hx-exec", "exec")


def run_native(binary: str, command: str, cwd: str, timeout: int, sandbox_args: list[str] | None = None) -> dict:
    argv = [
        binary,
        "--timeout", str(timeout),
        "--max-output", str(MAX_OUTPUT),
        "--cwd", cwd,
        "--fsize", str(MAX_FILE_MB),
        *(sandbox_args or []),
        "--", "/bin/bash", "-c", command,
    ]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                            env={**os.environ, **QUIET_ENV})
    try:
        out, err = proc.communicate(timeout=timeout + 15)
    except KeyboardInterrupt:
        # Ctrl-C reached hx-exec too (same terminal process group). It forwards the signal to the command's group,
        # waits up to 2s, then SIGKILLs it. Give it time to do that cleanup; killing hx-exec now would orphan
        # the command and everything it started.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"hx-exec failed: {err.strip()}")
    return json.loads(out)


def run_python(command: str, cwd: str, timeout: int) -> dict:
    """Fallback without hx-exec. Weaker: a process left in the background keeps the pipe open,
    so we wait for the timeout; and there are no resource limits."""
    started = time.monotonic()
    proc = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # new session = new process group, so we can kill the whole tree
        env={**os.environ, **QUIET_ENV},
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            out, _ = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # leftovers
    except ProcessLookupError:
        pass

    total = len(out)
    truncated = 0
    if total > MAX_OUTPUT:
        half = MAX_OUTPUT // 2
        truncated = total - MAX_OUTPUT
        out = out[:half] + f"\n\n... [{truncated} bytes of output truncated] ...\n\n".encode() + out[-half:]
    code = proc.returncode
    return {
        "exit_code": 128 - code if code < 0 else code,  # Popen reports -SIG for signals; use the shell's 128+SIG
        "signal": signal.Signals(-code).name if code < 0 else None,
        "timed_out": timed_out,
        "interrupted": False,
        "sandboxed": False,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "output_bytes": total,
        "truncated_bytes": truncated,
        "output": out.decode(errors="replace"),
    }


def format_result(r: dict, timeout: int) -> ToolResult:
    out = r["output"].rstrip("\n") or "(no output)"
    notes = []
    if r.get("sandboxed") and r["exit_code"] != 0 and looks_blocked(r["output"]):
        notes.append(
            "ran in the sandbox, which blocks network access and writes outside the workspace; "
            "if the command genuinely needs those, retry with sandbox=false (the user must approve)"
        )
    if r["timed_out"]:
        notes.append(f"timed out after {timeout}s; the command and its child processes were killed")
    elif r["signal"]:
        notes.append(f"killed by signal: {r['signal']}")
    notes.append(f"exit code {r['exit_code']}")
    if r["truncated_bytes"]:
        notes.append(f"{r['truncated_bytes']} bytes of output were cut from the middle")
    return ToolResult(f"{out}\n[{'; '.join(notes)}]", is_error=r["exit_code"] != 0 or r["timed_out"])


class Bash(Tool):
    name = "bash"
    subject_arg = "command"
    description = (
        "Run a bash command in the workspace directory and return its combined stdout/stderr and exit code. "
        "Use it to run tests, builds, git, and other CLI tools. Each call starts a fresh shell in the workspace "
        "(cd doesn't persist; use `cd dir && cmd`). stdin is closed, so interactive commands fail instead of "
        f"hanging. Default timeout {DEFAULT_TIMEOUT}s (max {MAX_TIMEOUT}s). Long output is truncated in the middle. "
        "Prefer read_file/grep/glob over cat/grep/find."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run"},
            "timeout": {"type": "integer", "description": f"Seconds before the command is killed (default {DEFAULT_TIMEOUT})"},
            "sandbox": {
                "type": "boolean",
                "description": "Default true: the command can't use the network or write outside the workspace. "
                "Set false only when it must (e.g. installing packages); that always needs the user's approval.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, args, ctx: ToolContext):
        timeout = min(max(1, args.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
        binary = exec_binary()
        if binary:
            sandbox_args = ctx.sandbox.hx_exec_args(ctx.cwd) if sandboxed(ctx, args) else None
            r = run_native(binary, args["command"], str(ctx.cwd), timeout, sandbox_args)
        else:
            r = run_python(args["command"], str(ctx.cwd), timeout)  # no sandbox without hx-exec
        return format_result(r, timeout)


def sandboxed(ctx: ToolContext, args: dict) -> bool:
    """Will this bash call run inside the OS sandbox? Permissions use this too."""
    return bool(ctx.sandbox and ctx.sandbox.enabled and exec_binary() and args.get("sandbox", True))
