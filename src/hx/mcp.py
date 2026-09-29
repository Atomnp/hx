"""Minimal MCP client over stdio (JSON-RPC 2.0, one message per line).

Each server tool is registered as mcp__<server>__<tool>. Servers are configured under "mcpServers" in settings.
"""

import itertools
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from hx import __version__, trust
from hx.tools.base import Tool, ToolContext, ToolResult

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict | None = None,
                 cwd: Path | None = None, timeout: float = 60):
        self.name = name
        self.timeout = timeout
        self.ids = itertools.count(1)
        self.pending: dict[int, queue.Queue] = {}
        self.lock = threading.Lock()
        log = Path.home() / ".hx" / "logs"
        log.mkdir(parents=True, exist_ok=True)
        self.stderr_log = open(log / f"mcp-{name}.log", "a")  # servers log to stderr; keep it out of the terminal
        try:
            self.proc = subprocess.Popen(
                [command, *(args or [])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr_log,
                text=True, bufsize=1, cwd=cwd, env={**os.environ, **(env or {})},
            )
        except OSError as e:
            raise MCPError(f"can't start MCP server {name!r} ({command}): {e}") from e
        threading.Thread(target=self.read_loop, daemon=True).start()
        self.server_info = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "hx", "version": __version__},
        })
        self.notify("notifications/initialized")

    # -- JSON-RPC plumbing

    def send(self, message: dict) -> None:
        with self.lock:
            if self.proc.poll() is not None:
                raise MCPError(f"MCP server {self.name!r} has exited")
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = next(self.ids)
        box: queue.Queue = queue.Queue(maxsize=1)
        self.pending[rid] = box
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params else {})})
        try:
            reply = box.get(timeout=self.timeout)
        except queue.Empty:
            raise MCPError(f"MCP server {self.name!r} didn't answer {method} within {self.timeout}s") from None
        finally:
            self.pending.pop(rid, None)
        if "error" in reply:
            err = reply["error"]
            raise MCPError(f"{self.name}: {method} failed: {err.get('message')} (code {err.get('code')})")
        return reply.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        self.send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def read_loop(self) -> None:
        """Background thread: route each response to the request waiting for it."""
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # not JSON-RPC (a stray print from the server): ignore
            if "id" in msg and ("result" in msg or "error" in msg):
                if box := self.pending.get(msg["id"]):
                    box.put(msg)
            elif "id" in msg and "method" in msg:
                # A request FROM the server (ping, roots/list...). Answer ping; decline the rest politely.
                if msg["method"] == "ping":
                    self.send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
                else:
                    self.send({"jsonrpc": "2.0", "id": msg["id"],
                               "error": {"code": -32601, "message": f"hx doesn't support {msg['method']}"}})
            # notifications from the server (logging, progress) are ignored

    # -- MCP methods

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:  # tools/list is paginated
            result = self.request("tools/list", {"cursor": cursor} if cursor else None)
            tools += result.get("tools", [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item["text"])
            else:  # images, audio, resources: describe rather than dump binary into the context
                parts.append(f"[{item.get('type')} content: {item.get('mimeType', '')}]")
        if "structuredContent" in result and not parts:
            parts.append(json.dumps(result["structuredContent"]))
        return "\n".join(parts) or "(no content)", bool(result.get("isError"))

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.stderr_log.close()


class MCPTool(Tool):
    """One server tool, adapted to hx's Tool interface."""

    def __init__(self, client: MCPClient, spec: dict):
        self.client = client
        self.remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description", "") or f"Tool {spec['name']} from MCP server {client.name}"
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}
        # Annotations are hints from the server. They're shown, but they don't bypass permissions:
        # a server you didn't write could claim anything. Allow rules like "mcp__github__*" are how you trust it.
        self.annotations = spec.get("annotations", {})
        self.read_only = False

    def run(self, args, ctx: ToolContext):
        try:
            text, is_error = self.client.call_tool(self.remote_name, args)
        except MCPError as e:
            return ToolResult(f"Error: {e}", True)
        return ToolResult(text, is_error)


def load_servers(cwd: Path, confirm=None) -> tuple[dict[str, dict], list[str]]:
    """User servers always load; project servers start commands, so they need trust like project hooks."""
    servers: dict[str, dict] = {}
    notes: list[str] = []
    user = Path.home() / ".hx" / "settings.json"
    if user.is_file():
        servers.update(json.loads(user.read_text()).get("mcpServers", {}))
    project = cwd / ".hx" / "settings.json"
    if project.is_file():
        config = json.loads(project.read_text()).get("mcpServers", {})
        if config and trust.gate(cwd, "mcpServers", config, confirm):
            servers.update(config)
        elif config:
            notes.append(f"ignored {len(config)} untrusted project MCP server(s) from {project}")
    return servers, notes


def connect_all(cwd: Path, servers: dict[str, dict]) -> tuple[list[MCPClient], list[MCPTool], list[str]]:
    """Start every configured server and collect its tools. A broken server is reported, never fatal."""
    clients, tools, notes = [], [], []
    for name, cfg in servers.items():
        try:
            client = MCPClient(name, cfg["command"], cfg.get("args"), cfg.get("env"), cwd=cwd)
            specs = client.list_tools()
        except (MCPError, KeyError) as e:
            notes.append(f"MCP server {name!r} unavailable: {e}")
            continue
        clients.append(client)
        tools += [MCPTool(client, s) for s in specs]
        info = client.server_info.get("serverInfo", {})
        notes.append(f"MCP server {name!r} ({info.get('name', '?')} {info.get('version', '')}): {len(specs)} tool(s)")
    return clients, tools, notes
