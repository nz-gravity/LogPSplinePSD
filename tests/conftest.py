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
    try:
        from git import Repo

        branch = Repo(".", search_parent_directories=True).active_branch.name
        outdir = f"{HERE}/test_output/branch_[{branch}]"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    except Exception:
        outdir = f"{HERE}/test_output/unknown_branch"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    return outdir
