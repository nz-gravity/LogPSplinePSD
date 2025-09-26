import os
import pytest

FAST_RUN = False

@pytest.fixture(scope="session")
def test_mode():
    """
    Fixture to determine the test mode (fast or slow).

    It checks for the 'GITHUB_ACTIONS' environment variable.
    - If running on GitHub Actions, it returns "fast".
    - Otherwise (for local runs), it returns "slow".

    This value can be used by tests to skip or adapt behavior for
    long-running tasks (e.g., waiting for external services).
    """
    # Check if the GITHUB_ACTIONS environment variable is set to 'true'.
    # This is the standard way to detect a GitHub Actions CI environment.
    is_github_ci = os.getenv("GITHUB_ACTIONS") == "true"

    if is_github_ci or FAST_RUN:
        return "fast"
    else:
        return "slow"


@pytest.fixture
def outdir():
    outdir = "test_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

