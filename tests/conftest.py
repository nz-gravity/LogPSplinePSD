import os

import pytest


def _compute_test_mode() -> str:
    env_mode = os.getenv("LOG_PSPLINES_TEST_MODE")
    if env_mode:
        env_mode = env_mode.lower()
        if env_mode in {"fast", "slow"}:
            return env_mode

    if os.getenv("LOG_PSPLINES_SLOW_TESTS") == "1":
        return "slow"

    if os.getenv("GITHUB_ACTIONS") == "true":
        return "fast"

    return "fast"


TEST_MODE = _compute_test_mode()

# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# if TEST_MODE == "fast":
#     os.environ.setdefault("JAX_DISABLE_JIT", "1")
# else:
#     os.environ.pop("JAX_DISABLE_JIT", None)


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
