"""Sandbox settings for shell commands, applied by hx-exec with macOS Seatbelt: writes only inside the
workspace and temp dirs, no network, no access to secret directories like ~/.ssh.
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SECRET_DIRS = [".ssh", ".aws", ".gnupg", ".config/gcloud", ".kube", ".docker"]

# Signs in a command's output that the sandbox blocked something.
BLOCKED_SIGNS = ("Operation not permitted", "Could not resolve host", "Couldn't connect to server",
                 "Temporary failure in name resolution", "nodename nor servname provided")


@dataclass
class SandboxConfig:
    enabled: bool = True
    allow_network: bool = False
    writable: list[str] = field(default_factory=list)  # extra writable dirs from settings

    @staticmethod
    def supported() -> bool:
        return sys.platform == "darwin"

    def hx_exec_args(self, workspace: Path) -> list[str]:
        """Flags for hx-exec. Paths are realpaths: Seatbelt matches resolved paths (/tmp is /private/tmp)."""
        writable = {os.path.realpath(workspace), os.path.realpath(tempfile.gettempdir()), "/private/tmp"}
        writable |= {os.path.realpath(os.path.expanduser(w)) for w in self.writable}
        args = ["--sandbox"]
        for w in sorted(writable):
            args += ["--allow-write", w]
        home = Path.home()
        for d in SECRET_DIRS:
            if (home / d).exists():
                args += ["--deny-read", os.path.realpath(home / d)]
        if self.allow_network:
            args.append("--allow-network")
        return args


def load_sandbox(cwd: Path, enabled: bool | None = None) -> SandboxConfig:
    """Settings: {"sandbox": {"enabled": true, "network": false, "writable": ["~/.cache/uv"]}}"""
    cfg = SandboxConfig()
    for path in (Path.home() / ".hx" / "settings.json", cwd / ".hx" / "settings.json"):
        if path.is_file():
            s = json.loads(path.read_text()).get("sandbox", {})
            cfg.enabled = s.get("enabled", cfg.enabled)
            cfg.allow_network = s.get("network", cfg.allow_network)
            cfg.writable += s.get("writable", [])
    if enabled is not None:
        cfg.enabled = enabled
    cfg.enabled = cfg.enabled and SandboxConfig.supported()
    return cfg


def looks_blocked(output: str) -> bool:
    return any(sign in output for sign in BLOCKED_SIGNS)
