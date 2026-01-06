import os

import numpy as np
import pytest

from log_psplines.datatypes import MultivariateTimeseries, Timeseries
from log_psplines.logger import set_level

HERE = os.path.dirname(os.path.abspath(__file__))
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


@pytest.fixture
def synthetic_univar_timeseries() -> Timeseries:
    rng = np.random.default_rng(123)
    n = 64
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 3.0 * t)
    y += 0.05 * rng.normal(size=n)
    return Timeseries(t=t, y=y)


@pytest.fixture
def synthetic_multivar_timeseries() -> MultivariateTimeseries:
    rng = np.random.default_rng(456)
    n = 64
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    base = np.stack(
        (
            np.sin(2 * np.pi * 2.0 * t),
            np.cos(2 * np.pi * 3.0 * t),
        ),
        axis=1,
    )
    y = base + 0.05 * rng.normal(size=base.shape)
    y[:, 1] += 0.15 * y[:, 0]
    return MultivariateTimeseries(t=t, y=y)
