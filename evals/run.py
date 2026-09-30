"""Run the eval suite: each task in a fresh copy of its repo, judged by an independent check.

    uv run python evals/run.py                         # all tasks, default variant
    uv run python evals/run.py --tasks 01,04 --repeat 3
    uv run python evals/run.py --variant think         # see VARIANTS

A task passes when (1) its check command exits 0 in the workspace afterwards, (2) its protected files are
byte-for-byte unchanged (no "fixing" the tests), (3) the final answer contains the expected text, if the task has one,
and (4) the agent finished on its own (no step limit, no crash).
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from hx.cli import build_agent, build_parser
from hx.events import TurnFinished
from hx.models import ModelError
from hx.tracing import Stats

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"

# A variant = CLI flags + environment overrides.
VARIANTS: dict[str, dict] = {
    "default": {"flags": [], "env": {}},
    "think": {"flags": [], "env": {"HX_THINK": "1"}},
    # ablations: the same model with harness features switched off
    "no-repair": {"flags": [], "env": {"HX_ABLATE": "repair"}},
    "minimal": {"flags": [], "env": {"HX_ABLATE": "repair,diagnostics,prompt,reminders"}},
    # a different model, with and without the repair layer
    "coder": {"flags": [], "env": {"HX_MODEL": "qwen2.5-coder:14b"}},
    "coder-no-repair": {"flags": [], "env": {"HX_MODEL": "qwen2.5-coder:14b", "HX_ABLATE": "repair"}},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


@contextlib.contextmanager
def inside(workdir: Path, env: dict):
    """Run with cwd = the task copy and the variant's environment, then restore both."""
    old_cwd, old_env = Path.cwd(), {k: os.environ.get(k) for k in env}
    os.chdir(workdir)
    os.environ.update(env)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_task(task_dir: Path, variant: dict, max_steps: int) -> dict:
    spec = json.loads((task_dir / "task.json").read_text())
    with tempfile.TemporaryDirectory(prefix=f"hx-eval-{task_dir.name}-") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(task_dir / "repo", work)
        protected = {p: digest(work / p) for p in spec.get("protect", [])}

        stats, finish = Stats(), {"reason": "error", "text": ""}

        def on_event(e):
            stats(e)
            if isinstance(e, TurnFinished):
                finish.update(reason=e.reason, text=e.text)

        started = time.monotonic()
        error = None
        with inside(work, variant["env"]):
            args = build_parser().parse_args(["--mode", "auto-edit", "--ephemeral", *variant["flags"]])
            agent, cleanup = build_agent(args, interactive=False)
            agent.max_steps = max_steps
            try:
                agent.run(spec["prompt"], on_event=on_event)
            except ModelError as e:
                error = str(e)
            finally:
                cleanup()
        duration = time.monotonic() - started

        check = subprocess.run(spec["check"], shell=True, cwd=work, capture_output=True, text=True, timeout=180)
        changed = [p for p, h in protected.items() if digest(work / p) != h]
        answer_ok = spec.get("answer_contains", "") in finish["text"]
        passed = check.returncode == 0 and not changed and answer_ok and finish["reason"] == "done" and not error

    return {
        "task": task_dir.name,
        "tags": spec.get("tags", []),
        "passed": passed,
        "check_ok": check.returncode == 0,
        "protected_changed": changed,
        "answer_ok": answer_ok,
        "finish_reason": "error" if error else finish["reason"],
        "error": error,
        "duration_s": round(duration, 1),
        "model_calls": stats.model_calls,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "tool_calls": sum(stats.tool_calls.values()),
        "tool_errors": sum(stats.tool_errors.values()),
        "check_output": (check.stdout + check.stderr)[-400:],
        "final_text": finish["text"][:500],  # what the agent said it did (for diagnosing failures)
    }


def summarize(rows: list[dict], variant: str) -> str:
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    rate = passed / n if n else 0
    stderr = (rate * (1 - rate) / n) ** 0.5 if n else 0
    lines = [
        f"## Variant `{variant}`: {passed}/{n} passed ({rate:.0%} ± {stderr:.0%} s.e.)",
        "",
        "| task | result | finish | calls | tokens in/out | tool errs | time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        why = "" if r["passed"] else (" (edited protected files!)" if r["protected_changed"]
                                      else " (wrong answer)" if not r["answer_ok"] else "")
        lines.append(f"| {r['task']} | {'✅' if r['passed'] else '❌'}{why} | {r['finish_reason']} | {r['model_calls']} | "
                     f"{r['input_tokens']:,}/{r['output_tokens']:,} | {r['tool_errors']} | {r['duration_s']}s |")
    lines += ["", f"median time {statistics.median(r['duration_s'] for r in rows):.1f}s · "
                  f"mean input tokens {statistics.mean(r['input_tokens'] for r in rows):,.0f} · "
                  f"cheating attempts {sum(bool(r['protected_changed']) for r in rows)}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", help="comma-separated task prefixes (default: all)")
    ap.add_argument("--variant", default="default", choices=list(VARIANTS))
    ap.add_argument("--repeat", type=int, default=1, help="runs per task (models are nondeterministic)")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--out", type=Path, help="results file (JSONL); default evals/results/<variant>-<time>.jsonl")
    args = ap.parse_args(argv)

    tasks = sorted(d for d in TASKS.iterdir() if (d / "task.json").exists())
    if args.tasks:
        prefixes = args.tasks.split(",")
        tasks = [t for t in tasks if any(t.name.startswith(p) for p in prefixes)]
    out = args.out or ROOT / "results" / f"{args.variant}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in tasks:
        for i in range(args.repeat):
            row = run_task(task, VARIANTS[args.variant], args.max_steps)
            row.update(variant=args.variant, run=i + 1)
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{'PASS' if row['passed'] else 'FAIL'}  {task.name}  ({row['duration_s']}s, {row['model_calls']} calls)",
                  file=sys.stderr, flush=True)
    report = summarize(rows, args.variant)
    out.with_suffix(".md").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
