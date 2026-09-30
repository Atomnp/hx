"""Compare eval variants side by side.

    uv run python evals/compare.py evals/results/abl-*.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = defaultdict(list)
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            rows[r["variant"]].append(r)
    return rows


def main(paths):
    rows = load(paths)
    tasks = sorted({r["task"] for rs in rows.values() for r in rs})
    print("| variant | pass rate | runs | mean tokens in | median time | tool errors/run | text-call/loop notes |")
    print("|---|---|---|---|---|---|---|")
    for v, rs in rows.items():
        n = len(rs)
        rate = sum(r["passed"] for r in rs) / n
        se = (rate * (1 - rate) / n) ** 0.5
        print(f"| {v} | **{rate:.0%}** ± {se:.0%} | {n} | {statistics.mean(r['input_tokens'] for r in rs):,.0f} | "
              f"{statistics.median(r['duration_s'] for r in rs):.1f}s | {statistics.mean(r['tool_errors'] for r in rs):.1f} | "
              f"{sum(r['finish_reason'] != 'done' for r in rs)} unfinished |")
    print()
    print("| task | " + " | ".join(rows) + " |")
    print("|---|" + "---|" * len(rows))
    for t in tasks:
        cells = []
        for rs in rows.values():
            runs = [r for r in rs if r["task"] == t]
            cells.append(f"{sum(r['passed'] for r in runs)}/{len(runs)}" if runs else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
