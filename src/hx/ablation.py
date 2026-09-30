"""Switches that turn harness features off for evals, e.g. HX_ABLATE=repair."""

import os

KNOWN = {"repair", "diagnostics", "prompt", "reminders"}


def off(feature: str) -> bool:
    assert feature in KNOWN, feature
    return feature in os.environ.get("HX_ABLATE", "").replace(" ", "").split(",")
