"""Terminal output: RichUI for interactive use, PlainUI when output is piped."""

import sys

from hx.events import Event, ModelCallStarted, Notice, StepFinished, TextDelta, ToolFinished, ToolStarted, TurnFinished

PREVIEW = 200


def warning_lines(result: str) -> list[str]:
    return [line for line in result[PREVIEW:].splitlines() if line.startswith("⚠")]


class PlainUI:
    def __call__(self, event: Event) -> None:
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolStarted):
            print(f"\n  → {event.call.name}({event.call.arguments})", flush=True)
        elif isinstance(event, ToolFinished):
            mark = "✗" if event.is_error else "✓"
            print(f"  {mark} {event.result[:PREVIEW]}", flush=True)
            for line in warning_lines(event.result):  # don't let the preview cut hide a warning the model received
                print(f"  {line}", flush=True)
        elif isinstance(event, TurnFinished) and event.reason == "max_steps":
            print(f"\n  [hx] stopped after {event.steps} steps without finishing. Say 'continue' to keep going.", file=sys.stderr)
        elif isinstance(event, TurnFinished) and event.reason == "interrupted":
            print("\n  [hx] interrupted.", file=sys.stderr)
        elif isinstance(event, Notice):
            print(f"  [hx] {event.text}", file=sys.stderr, flush=True)
        elif isinstance(event, StepFinished):
            u = event.usage
            print(f"\n  [{u.prompt_tokens} in / {u.completion_tokens} out, {u.duration_s:.1f}s, context {event.context_used:.0%}]",
                  file=sys.stderr, flush=True)


class RichUI:
    def __init__(self):
        from rich.console import Console

        self.console = Console(highlight=False)
        self.live = None  # rich.live.Live while assistant text is streaming
        self.status = None  # spinner while waiting for the model
        self.buffer = ""
        self.tokens_in = self.tokens_out = 0
        self.seconds = 0.0
        self.context = 0.0

    # -- helpers

    def stop_spinner(self):
        if self.status:
            self.status.stop()
            self.status = None

    def end_text(self):
        """Freeze the streamed Markdown so tool lines print below it."""
        if self.live:
            self.live.stop()
            self.live = None
        self.buffer = ""

    def render_result(self, event: ToolFinished):
        from rich.text import Text

        style = "red" if event.is_error else "dim"
        lines = event.result.splitlines()
        shown = lines[:12]
        for line in shown:
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(Text("    " + line, style="green"))
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(Text("    " + line, style="red"))
            else:
                self.console.print(Text("    " + line[:160], style=style))
        if len(lines) > len(shown):
            self.console.print(Text(f"    … {len(lines) - len(shown)} more lines", style="dim italic"))
        for line in warning_lines(event.result):
            self.console.print(Text("    " + line, style="bold yellow"))

    # -- the handler

    def __call__(self, event: Event) -> None:
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.text import Text

        if isinstance(event, ModelCallStarted):
            self.end_text()
            self.status = self.console.status("[dim]thinking…[/dim]", spinner="dots")
            self.status.start()
        elif isinstance(event, TextDelta):
            self.stop_spinner()
            self.buffer += event.text
            if self.live is None:
                self.live = Live(Markdown(self.buffer), console=self.console, refresh_per_second=12, vertical_overflow="visible")
                self.live.start()
            else:
                self.live.update(Markdown(self.buffer))
        elif isinstance(event, ToolStarted):
            self.stop_spinner()
            self.end_text()
            args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in event.call.arguments.items())
            self.console.print(Text.assemble(("  ⏺ ", "cyan"), (event.call.name, "bold cyan"), (f"({args})", "cyan")))
        elif isinstance(event, ToolFinished):
            self.render_result(event)
        elif isinstance(event, Notice):
            self.stop_spinner()
            self.end_text()
            self.console.print(Text(f"  [hx] {event.text}", style="yellow"))
        elif isinstance(event, StepFinished):
            self.stop_spinner()
            self.tokens_in = event.usage.prompt_tokens
            self.tokens_out += event.usage.completion_tokens
            self.seconds += event.usage.duration_s
            self.context = event.context_used
        elif isinstance(event, TurnFinished):
            self.stop_spinner()
            self.end_text()
            if event.reason == "max_steps":
                self.console.print(Text(f"  stopped after {event.steps} steps without finishing. Say 'continue' to keep going.", style="yellow"))
            elif event.reason == "interrupted":
                self.console.print(Text("  interrupted.", style="yellow"))
            self.console.print(Text(
                f"  {event.steps} step(s) · {self.tokens_out} tokens out · {self.seconds:.1f}s · context {self.context:.0%}",
                style="dim"))
            self.tokens_out, self.seconds = 0, 0.0

    def cleanup(self):
        """After an interrupt: make sure no spinner or live region is left running."""
        self.stop_spinner()
        self.end_text()


def make_ui():
    if sys.stdout.isatty():
        try:
            return RichUI()
        except ImportError:
            pass
    return PlainUI()
