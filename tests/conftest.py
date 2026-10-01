"""Shared pytest setup and isolated output directories."""

import os
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).resolve().parent / "test-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

from log_psplines.logger import set_level  # noqa: E402

set_level("WARNING")


@pytest.fixture
def outdir(tmp_path: Path) -> Path:
    """Give each test an isolated directory for fit outputs and plots."""
    return tmp_path
