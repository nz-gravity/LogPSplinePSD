import os

import pytest

# HARDCODE TO SLOW FOR MORE ACCURATE TESTS
os.environ["LOG_PSPLINES_SLOW_TESTS"] = "1"


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
    outdir = "test_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
