import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from log_psplines.datatypes import MultivariateTimeseries, Timeseries
from log_psplines.logger import set_level

HERE = Path(__file__).parent
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
def outdir(request):
    # 1. Get Git branch
    try:
        from git import Repo

        branch = Repo(".", search_parent_directories=True).active_branch.name
    except Exception:
        branch = "unknown_branch"

    # 2. Get the filename (e.g., 'test_physics') and test name (e.g., 'test_simulation')
    # request.path.stem gives 'test_logic' from 'test_logic.py'
    file_stem = request.path.stem
    test_name = request.node.name

    # 3. Build path: .../test_logic/branch_[main]/test_simulation
    target_dir = (
        HERE / "test_output" / f"branch_[{branch}]" / file_stem / test_name
    )

    # 4. Cleanup and Create
    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    return target_dir
