"""Shared pytest setup and persistent local artifact directory."""

import os
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).resolve().parent / "test-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

from log_psplines.logger import set_level  # noqa: E402

os.environ.setdefault(
    "LOG_PSPLINES_SLOW_TESTS",
    "0" if os.getenv("GITHUB_ACTIONS") == "true" else "1",
)
set_level("WARNING")


@pytest.fixture(scope="session")
def outdir() -> Path:
    """Return an ignored, persistent directory for fit outputs and plots."""
    return OUTPUT_DIR
