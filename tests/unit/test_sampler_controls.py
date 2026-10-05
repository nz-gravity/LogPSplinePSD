"""Package sampler controls, without study inputs or MCMC trajectories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import xarray as xr
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_value

from log_psplines.diagnostics.sampling import _tree_depth_hits
from log_psplines.inference import nuts as nuts_backend


def gaussian_model(*, declared_argument: float = 0.0) -> None:
    numpyro.sample("x", dist.Normal(declared_argument, 1.0))


@pytest.fixture
def sampler_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Use a real NumPyro kernel but replace trajectory execution with a spy."""
    calls: dict[str, Any] = {}

    def kernel(model: Callable, **options: Any) -> NUTS:
        calls["model"] = model
        calls["kernel_options"] = options
        return NUTS(model, **options)

    class RecordedMCMC:
        def __init__(self, kernel: NUTS, **options: Any) -> None:
            calls["kernel"] = kernel
            calls["mcmc_options"] = options

        def run(self, rng_key: jax.Array, **kwargs: Any) -> None:
            calls["rng_key"] = rng_key
            calls["run_kwargs"] = kwargs

        def get_samples(
            self, *, group_by_chain: bool
        ) -> dict[str, np.ndarray]:
            assert group_by_chain
            return {"x": np.array([[0.25, -0.5]])}

        def get_extra_fields(
            self, *, group_by_chain: bool
        ) -> dict[str, np.ndarray]:
            assert group_by_chain
            return {}

    monkeypatch.setattr(nuts_backend, "NUTS", kernel)
    monkeypatch.setattr(nuts_backend, "MCMC", RecordedMCMC)
    return calls


def test_custom_strategy_reaches_numpyro_unchanged(
    sampler_calls: dict[str, Any],
) -> None:
    strategy = init_to_value(values={"x": jnp.asarray(2.5)})
    key = jax.random.PRNGKey(42)
    result = nuts_backend.run_nuts(
        gaussian_model,
        rng_key=key,
        init_strategy=strategy,
        model_kwargs={"declared_argument": 3.0},
        n_warmup=2,
        n_samples=2,
        chain_method="sequential",
        extra_fields=("num_steps",),
    )
    assert sampler_calls["model"] is gaussian_model
    assert sampler_calls["kernel_options"]["init_strategy"] is strategy
    assert sampler_calls["mcmc_options"]["chain_method"] == "sequential"
    np.testing.assert_array_equal(sampler_calls["rng_key"], key)
    assert sampler_calls["run_kwargs"] == {
        "extra_fields": ("num_steps",),
        "declared_argument": 3.0,
    }
    np.testing.assert_array_equal(result.posterior.x, [[0.25, -0.5]])
    assert result.sample_stats is None


def test_explicit_values_still_use_numpyro_value_initialization(
    sampler_calls: dict[str, Any],
) -> None:
    nuts_backend.run_nuts(
        gaussian_model,
        rng_key=jax.random.PRNGKey(43),
        init_values={"x": jnp.asarray(-1.75)},
        n_warmup=2,
        n_samples=2,
    )
    strategy = sampler_calls["kernel_options"]["init_strategy"]
    value = strategy({"type": "sample", "name": "x", "is_observed": False})
    np.testing.assert_array_equal(value, -1.75)


def test_default_initializer_is_left_to_numpyro(
    sampler_calls: dict[str, Any],
) -> None:
    nuts_backend.run_nuts(
        gaussian_model,
        rng_key=jax.random.PRNGKey(44),
        n_warmup=2,
        n_samples=2,
    )
    assert "init_strategy" not in sampler_calls["kernel_options"]


@pytest.mark.parametrize("values", [{}, {"x": jnp.asarray(0.0)}])
def test_initialization_controls_are_exclusive_before_kernel_creation(
    sampler_calls: dict[str, Any], values: dict[str, jax.Array]
) -> None:
    with pytest.raises(
        ValueError, match="choose init_values or init_strategy"
    ):
        nuts_backend.run_nuts(
            gaussian_model,
            rng_key=jax.random.PRNGKey(45),
            init_values=values,
            init_strategy=init_to_value(values={"x": jnp.asarray(1.0)}),
            n_warmup=2,
            n_samples=2,
        )
    assert sampler_calls == {}


def diagnostic_tree(**stats: list[float]) -> xr.DataTree:
    """Represent one chain of statistics with native xarray dimensions."""
    dataset = xr.Dataset(
        {name: (("chain", "draw"), [values]) for name, values in stats.items()}
    )
    return xr.DataTree.from_dict({"sample_stats": dataset})


@pytest.mark.parametrize(
    "depth,below,cap", [(1, 0, 1), (3, 6, 7), (10, 1022, 1023)]
)
def test_step_count_cap_includes_exact_boundary_and_ignores_nonfinite(
    depth: int, below: int, cap: int
) -> None:
    tree = diagnostic_tree(n_steps=[below, cap, cap + 1, np.nan, np.inf])
    assert _tree_depth_hits(tree, depth) == 2


def test_finite_tree_depth_takes_precedence_over_step_counts() -> None:
    tree = diagnostic_tree(tree_depth=[2, 3, 4, np.nan], n_steps=[7, 0, 0, 7])
    assert _tree_depth_hits(tree, 3) == 2


def test_nonfinite_tree_depth_falls_back_to_finite_step_counts() -> None:
    tree = diagnostic_tree(tree_depth=[np.nan, np.inf], n_steps=[6, 7])
    assert _tree_depth_hits(tree, 3) == 1


def test_missing_cap_or_sampler_statistics_have_no_depth_hits() -> None:
    assert _tree_depth_hits(diagnostic_tree(n_steps=[1023]), None) == 0
    assert _tree_depth_hits(xr.DataTree(), 10) == 0
    assert _tree_depth_hits(diagnostic_tree(diverging=[0.0]), 10) == 0
