import os

import pytest

from log_psplines.logger import set_level

HERE = os.path.dirname(os.path.abspath(__file__))
# HARDCODE TO SLOW FOR MORE ACCURATE TESTS
os.environ["LOG_PSPLINES_SLOW_TESTS"] = "1"

set_level("DEBUG")


def _compute_test_mode() -> str:
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "fast"

    if os.getenv("LOG_PSPLINES_SLOW_TESTS") == "1":
        return "slow"

    return "fast"


TEST_MODE = _compute_test_mode()


@pytest.fixture(scope="session")
def test_mode():
    """Expose the resolved test mode to tests."""
    return TEST_MODE


@pytest.fixture
def outdir():
    # get git branch
    branch = os.getenv("GITHUB_HEAD_REF", "local").replace("/", "_")
    outdir = f"{HERE}/test_output/{branch}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
