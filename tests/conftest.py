import os

import pytest

from log_psplines.logger import set_level

if os.getenv("GITHUB_ACTIONS") == "true":
    os.environ.setdefault("LOG_PSPLINES_SLOW_TESTS", "0")
else:
    os.environ.setdefault("LOG_PSPLINES_SLOW_TESTS", "1")

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


def pytest_collection_modifyitems(config, items):
    if TEST_MODE != "fast":
        return
    skip_slow = pytest.mark.skip(reason="Skipping slow tests in fast mode.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def outdir(tmp_path):
    """Give each test a clean output directory managed by pytest."""
    return tmp_path
