"""Locates the C++ helper binaries: env var first, then cpp/build, then $PATH."""

import os
import shutil
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "cpp" / "build"


@cache
def find_binary(name: str, subdir: str) -> str | None:
    env = os.environ.get(f"HX_{name.split('-')[1].upper()}_BIN")
    if env:
        return env if Path(env).is_file() else None
    built = BUILD_DIR / subdir / name
    if built.is_file() and os.access(built, os.X_OK):
        return str(built)
    return shutil.which(name)
