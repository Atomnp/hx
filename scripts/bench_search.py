"""Benchmark hx-search (C++) against the pure-Python fallback on a generated codebase.

    uv run python scripts/bench_search.py [--files 4000] [--lines 300]
"""

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path

from hx.tools import ToolContext
from hx.tools import search as search_mod
from hx.tools.search import Grep

WORDS = "def class return import self value result data config error handler request user cache".split()


def make_tree(root: Path, files: int, lines: int) -> int:
    rng = random.Random(0)
    total = 0
    for i in range(files):
        d = root / f"pkg{i % 40}" / f"mod{i % 7}"
        d.mkdir(parents=True, exist_ok=True)
        body = []
        for n in range(lines):
            body.append("    " + " ".join(rng.choice(WORDS) for _ in range(8)) + f"  # {i}:{n}")
        if i % 97 == 0:
            body.append("    raise ConnectionTimeoutError('upstream')")
        text = "\n".join(body) + "\n"
        (d / f"file{i}.py").write_text(text)
        total += len(text)
    return total


def time_grep(cwd: Path, pattern: str, native: bool, repeat: int = 3) -> tuple[float, int]:
    original = search_mod.search_binary
    if not native:
        search_mod.search_binary = lambda: None
    try:
        times, hits = [], 0
        for _ in range(repeat):
            t = time.perf_counter()
            r = Grep().run({"pattern": pattern, "max_results": 100000}, ToolContext(cwd=cwd))
            times.append(time.perf_counter() - t)
            hits = 0 if r.content.startswith("No matches") else len(r.content.splitlines())
        return statistics.median(times), hits
    finally:
        search_mod.search_binary = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=4000)
    ap.add_argument("--lines", type=int, default=300)
    args = ap.parse_args()
    if not search_mod.search_binary():
        raise SystemExit("hx-search isn't built: run scripts/build_cpp.sh first")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        size = make_tree(root, args.files, args.lines)
        print(f"corpus: {args.files} files, {args.files * args.lines:,} lines, {size / 1e6:.0f} MB\n")
        print(f"{'pattern':<36}{'C++ (s)':>10}{'Python (s)':>12}{'speedup':>10}{'hits':>8}")
        for pattern in ["ConnectionTimeoutError", r"raise \w+Error\(", r"(?i)HANDLER request"]:
            if pattern.startswith("(?i)"):
                continue  # inline flags aren't ECMAScript; covered by ignore_case instead
            native_t, hits = time_grep(root, pattern, native=True)
            py_t, py_hits = time_grep(root, pattern, native=False)
            assert hits == py_hits, (hits, py_hits)
            print(f"{pattern:<36}{native_t:>10.3f}{py_t:>12.3f}{py_t / native_t:>9.1f}x{hits:>8}")


if __name__ == "__main__":
    main()
