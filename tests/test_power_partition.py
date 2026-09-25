"""Count-aware WDM likelihood compression and native-grid reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from log_psplines import (
    LogPSpline,
    PowerPartition,
    PowerSpectrum,
    PowerSplineConfig,
    PSDResult,
    SplineBasis,
    coarse_grain_power,
    fit,
    mask_power,
    select_power_partition,
)
from log_psplines.inference.power import prepare_power_model
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood


def test_masked_pooling_conserves_power_and_counts():
    time = np.array([0., 1., 2., 7., 8.])
    freq = np.array([0.1, 0.2, 0.3, 0.4])
    power = np.arange(1, 21, dtype=float).reshape(5, 4)
    counts = np.ones_like(power)
    counts[0, 0] = 2
    mask = np.ones_like(power, dtype=bool)
    mask[1, 2] = False
    mask[3, :] = False
    native = mask_power(PowerSpectrum(power, counts, freq, time), mask)
    partition = PowerPartition(np.array([0, 2, 3, 4]), np.array([0, 2]))
    coarse = coarse_grain_power(native, partition)
    np.testing.assert_allclose(coarse.power.sum(), native.power.sum())
    np.testing.assert_allclose(coarse.counts.sum(), native.counts.sum())
    assert np.all(coarse.counts[2] == 0)
    assert np.all(coarse.power[2] == 0)
    np.testing.assert_allclose(coarse.time, [0.5, 2., 7., 8.])
    np.testing.assert_allclose(coarse.frequency, [0.15, 0.35])
    assert coarse.counts[0, 0] == 5


def test_pilot_only_sets_boundaries_and_gap_splits():
    time = np.array([0., 1., 2., 10., 11.])
    pilot = np.array([[0., 0.01, 2., 2.01]])
    counts = np.ones((5, 4))
    counts[2, :] = 0
    partition = select_power_partition(
        pilot, time, counts=counts, time_bin=4,
        max_frequency_bin=4, max_log_range=0.25,
    )
    np.testing.assert_array_equal(partition.frequency_starts, [0, 2])
    np.testing.assert_array_equal(partition.time_starts, [0, 2, 3])
    raw = np.arange(1, 21., dtype=float).reshape(5, 4)
    raw[2, :] = 0
    data = PowerSpectrum(raw, counts, np.arange(1., 5.), time)
    pooled = coarse_grain_power(data, partition)
    assert pooled.power[0, 0] == raw[:2, :2].sum()
    with pytest.raises(ValueError, match="time_starts"):
        coarse_grain_power(data, PowerPartition([1], [0]))


def test_block_constant_likelihood_identity():
    power = np.array([[1., 2.], [3., 0.]])
    counts = np.array([[1., 2.], [1., 0.]])
    data = PowerSpectrum(power, counts, [1., 2.], [0., 1.])
    pooled = coarse_grain_power(data, PowerPartition([0], [0]))
    log_s = np.log(3.)
    native_ll = power_whittle_log_likelihood(
        power, counts, np.full_like(power, log_s)
    )
    coarse_ll = power_whittle_log_likelihood(
        pooled.power, pooled.counts, np.array([[log_s]])
    )
    np.testing.assert_allclose(native_ll, coarse_ll)
    native_grad = jax.grad(
        lambda value: power_whittle_log_likelihood(
            power, counts, jnp.full(power.shape, value)
        )
    )(log_s)
    coarse_grad = jax.grad(
        lambda value: power_whittle_log_likelihood(
            pooled.power, pooled.counts, jnp.full((1, 1), value)
        )
    )(log_s)
    np.testing.assert_allclose(native_grad, coarse_grad)


def test_explicit_interior_knots():
    grid = np.linspace(0, 1, 12)
    basis = SplineBasis.from_grid(
        grid, interior_knots=np.array([0.2, 0.8])
    )
    np.testing.assert_allclose(basis.knots[4:6], [0.2, 0.8])
    with pytest.raises(ValueError, match="interior_knots"):
        SplineBasis.from_grid(grid, interior_knots=[0.8, 0.2])
    with pytest.raises(ValueError, match="exactly one"):
        SplineBasis.from_grid(grid, 2, interior_knots=[0.2, 0.8])
    with pytest.raises(ValueError, match="exactly one"):
        SplineBasis.from_grid(grid)


def test_halfnormal_prior_log_density():
    with jax.enable_x64(True):
        data = PowerSpectrum(np.ones((4, 5)), 1, np.arange(1., 6.),
                             np.arange(4.))
        spline = LogPSpline(SplineBasis.from_grid(data.frequency, 1),
                            SplineBasis.from_grid(data.time, 1))
        config = PowerSplineConfig(roughness_scale=3.)
        model, init, _ = prepare_power_model(data, spline, config)
        from numpyro import handlers
        # Both roughness scales are sampled directly from HalfNormal priors.
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        for axis in ("time", "freq"):
            sigma = np.asarray(trace[f"sigma_{axis}"]["value"])
            expected = (np.log(np.sqrt(2 / np.pi) / 3)
                        - sigma**2 / 18)
            np.testing.assert_allclose(
                trace[f"sigma_{axis}"]["fn"].log_prob(sigma), expected
            )
        assert np.isfinite(log_density(model, (), {}, init)[0])


@pytest.mark.parametrize("gapped", [False, True])
def test_short_partition_fit_and_roundtrip(gapped, tmp_path):
    with jax.enable_x64(True):
        time = np.linspace(0, 70, 8)
        freq = np.linspace(1, 10, 8)
        power = np.ones((8, 8))
        counts = np.ones_like(power)
        if gapped:
            counts[3:5] = 0
            power[3:5] = 0
        data = PowerSpectrum(power, counts, freq, time)
        pilot = np.zeros((2, 8))
        partition = select_power_partition(
            pilot, time, counts=counts, time_bin=2,
            max_frequency_bin=2, max_log_range=0.25,
        )
        spline = LogPSpline(
            SplineBasis.from_grid(freq / freq[-1], 1),
            SplineBasis.from_grid(time / time[-1], 1),
        )
        result = fit(
            data, PowerSplineConfig(n_warmup=2, n_samples=2,
                                    progress_bar=False, max_tree_depth=3),
            model=spline, partition=partition,
        )
        assert result.psd.shape == (1, 2, 8, 8)
        assert np.isfinite(result.psd).all()
        np.testing.assert_array_equal(
            result.metadata["partition_time_starts"], partition.time_starts
        )
        np.testing.assert_array_equal(
            result.metadata["partition_frequency_starts"],
            partition.frequency_starts,
        )
        path = tmp_path / "fit.nc"
        result.to_netcdf(path)
        restored = PSDResult.from_netcdf(path)
        np.testing.assert_allclose(restored.psd, result.psd)
        np.testing.assert_array_equal(restored.time, time)
        np.testing.assert_array_equal(restored.frequency, freq)
        np.testing.assert_array_equal(
            restored.metadata["partition_time_starts"], partition.time_starts
        )
