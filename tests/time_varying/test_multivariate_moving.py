"""Exact moving-window statistics and paired complex inference contracts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from numpyro.infer.util import log_density

from log_psplines import (
    ANOVALogPSpline,
    PowerConfig,
    PSDResult,
    SplineBasis,
    TimeSeries,
    WishartGridData,
    fit,
)
from log_psplines.diagnostics.sampling import sampling_diagnostics
from log_psplines.inference.anova_power import prepare_anova_prior
from log_psplines.inference.wishart_grid import (
    _initialize_scattered,
    prepare_wishart_grid_row,
    row_field_labels,
)
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.anova import anova_components
from log_psplines.models.reconstruction import (
    spectral_quantiles,
    wishart_grid_draws,
)
from log_psplines.preprocessing.moving_periodogram import (
    multivariate_moving_periodogram,
    scattered_moving_periodogram,
    tang_moving_periodogram,
)
from log_psplines.preprocessing.wishart_grid import coarse_grain_wishart_grid


def _spline(data):
    return ANOVALogPSpline(
        SplineBasis.from_grid(
            data.reference_frequency, 1, degree=1, penalty_order=1
        ),
        SplineBasis.from_grid(
            data.reference_time, 0, degree=1, penalty_order=1
        ),
    )


def _grid():
    rng = np.random.default_rng(784)
    z = (
        rng.normal(size=(3, 4, 3, 5)) + 1j * rng.normal(size=(3, 4, 3, 5))
    ) / np.sqrt(2)
    return WishartGridData.from_coefficients(
        z, np.linspace(0, 1, 3), np.linspace(0.1, 1, 4)
    )


def _paired(grid):
    shape = (-1, grid.p, grid.U.shape[-1])
    return WishartGridData(
        grid.u_re.reshape(shape),
        grid.u_im.reshape(shape),
        np.repeat(grid.time, len(grid.frequency)),
        np.tile(grid.frequency, len(grid.time)),
        grid.counts.reshape(-1),
        reference_time=grid.reference_time,
        reference_frequency=grid.reference_frequency,
    )


@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("m,thin,n", [(1, 1, 19), (4, 2, 103)])
def test_direct_multichannel_dft_physical_times_and_unused_tail(
    channels, m, thin, n
):
    rng = np.random.default_rng(180 + channels)
    values = rng.normal(size=(n, channels)) * np.arange(1, channels + 1)
    dt, offset, length = 0.3, 17.8, 2 * m + 1
    time = offset + dt * np.arange(n)
    starts, frequencies, expected = [], [], []
    for block in range((n - 2 * m) // (thin * m)):
        for rung in range(1, m + 1):
            start = block * thin * m + rung - 1
            phase = np.exp(-2j * np.pi * rung * np.arange(length) / length)
            expected.append(
                np.sqrt(2 * dt / length)
                * (phase @ values[start : start + length])
            )
            starts.append(start)
            frequencies.append(rung / (length * dt))
    observed = multivariate_moving_periodogram(
        TimeSeries(values, time), m=m, thin=thin
    )
    np.testing.assert_allclose(
        observed.U[..., 0], expected, atol=3e-13, rtol=3e-13
    )
    np.testing.assert_array_equal(observed.time, time[np.asarray(starts) + m])
    np.testing.assert_allclose(observed.frequency, frequencies, rtol=2e-14)
    assert np.all(observed.counts == 1)
    altered = values.copy()
    altered[max(starts) + length :] = 1e8
    trailing = multivariate_moving_periodogram(
        TimeSeries(altered, time), m=m, thin=thin
    )
    np.testing.assert_array_equal(trailing.U, observed.U)
    raw = tang_moving_periodogram(values, m=m, thin=thin)
    np.testing.assert_allclose(
        raw["coeff"] * np.sqrt(4 * np.pi * dt), observed.U[..., 0], atol=3e-13
    )


def test_complex_scalar_likelihood_and_gradients_use_physical_psd_units():
    n, m, dt = 83, 4, 0.4
    x = np.random.default_rng(280).normal(size=n)
    observed = multivariate_moving_periodogram(
        TimeSeries(x, 2 + dt * np.arange(n)), m=m
    )
    scalar = scattered_moving_periodogram(x, dt=dt, m=m)
    q, scale = len(observed.counts), 4 * np.pi * dt
    logs = jnp.linspace(-0.7, 0.4, q)
    empty_theta, empty_previous = np.empty((q, 0)), np.empty((q, 0, 1))

    def complex_likelihood(value):
        return wishart_log_likelihood(
            value,
            empty_theta,
            empty_theta,
            observed.u_re[:, 0],
            observed.u_im[:, 0],
            empty_previous,
            empty_previous,
            counts=observed.counts,
            clip_log_psd=False,
        )

    def scalar_likelihood(value):
        return power_whittle_log_likelihood(
            scale * scalar.power, scalar.counts, value
        )

    np.testing.assert_allclose(
        complex_likelihood(logs), scalar_likelihood(logs), rtol=1e-13
    )
    np.testing.assert_allclose(
        jax.grad(complex_likelihood)(logs),
        jax.grad(scalar_likelihood)(logs),
        atol=1e-13,
    )
    raw_units = power_whittle_log_likelihood(
        scalar.power, scalar.counts, logs - np.log(scale)
    )
    np.testing.assert_allclose(
        complex_likelihood(logs), raw_units - q * np.log(scale), atol=1e-12
    )


def test_lagged_cross_phase_and_exact_white_noise_gram():
    n, m, dt, rung, lag = 100, 8, 0.25, 3, 2
    omega = 2 * np.pi * rung / (2 * m + 1)
    values = np.column_stack(
        (np.cos(omega * np.arange(n)), np.cos(omega * (np.arange(n) - lag)))
    )
    time = 37.5 + dt * np.arange(n)
    observed = multivariate_moving_periodogram(TimeSeries(values, time), m=m)
    selected = np.isclose(observed.frequency, omega / (2 * np.pi * dt))
    cross = observed.Y[selected, 0, 1]
    np.testing.assert_allclose(
        cross / abs(cross), np.exp(1j * omega * lag), atol=2e-14
    )
    assert np.all(abs(cross.imag) > 0.1)
    # Transforming impulses exposes the exact covariance for unit white noise.
    filters = multivariate_moving_periodogram(
        TimeSeries(np.eye(n), time), m=m
    ).U[..., 0]
    covariance, pseudo = filters @ filters.conj().T, filters @ filters.T
    np.testing.assert_allclose(np.diag(covariance), 2 * dt, atol=1e-14)
    np.testing.assert_allclose(np.diag(pseudo), 0, atol=1e-14)
    off_diagonal = ~np.eye(len(covariance), dtype=bool)
    assert abs(covariance[off_diagonal]).max() > 0.01
    assert abs(pseudo[off_diagonal]).max() > 0.01


@pytest.mark.parametrize("row", [0, 1, 2])
def test_paired_rows_match_rectangular_likelihood_and_full_gradients(row):
    grid = _grid().mask(np.arange(12).reshape(3, 4) != 5)
    paired, spline = _paired(grid), _spline(grid)
    config = PowerConfig(structure="anova", centered=True)
    grid_model, params = prepare_wishart_grid_row(grid, spline, config, row)
    paired_model, _ = prepare_wishart_grid_row(paired, spline, config, row)
    rng = np.random.default_rng(380 + row)
    for label in row_field_labels(row):
        for field in ("g", "eta"):
            params[f"{field}_{label}"] = (
                rng.normal(size=params[f"{field}_{label}"].shape) * 0.15
            )
        params[f"sigma_g_{label}"], params[f"sigma_eta_{label}"] = 0.7, 0.3

    def grid_density(p):
        return log_density(grid_model, (), {}, p)[0]

    def paired_density(p):
        return log_density(paired_model, (), {}, p)[0]

    np.testing.assert_allclose(
        grid_density(params), paired_density(params), rtol=1e-13
    )
    point_gradient = jax.grad(paired_density)(params)
    for name, gradient in jax.grad(grid_density)(params).items():
        np.testing.assert_allclose(gradient, point_gradient[name], atol=1e-12)
    bt, bf = spline.design()
    point_bt, point_bf = spline.design(paired.time, paired.frequency)
    g, eta = rng.normal(size=3), rng.normal(size=(1, 3))
    mean, deviation = anova_components(bt, bf, g, eta)
    point_mean, point_deviation = anova_components(
        point_bt, point_bf, g, eta, paired=True
    )
    np.testing.assert_allclose(
        point_mean + point_deviation,
        (mean[None] + deviation).reshape(-1),
        atol=1e-14,
    )


def test_batched_paired_anova_preserves_duplicate_and_permuted_sites():
    grid = _grid()
    spline, paired = _spline(grid), _paired(grid)
    indices = np.array([8, 2, 2, 10, 0])
    bt, bf = spline.design()
    point_bt, point_bf = spline.design(
        paired.time[indices], paired.frequency[indices]
    )
    rng = np.random.default_rng(390)
    g, eta = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 3, 1, 3))
    mean, deviation = anova_components(bt, bf, g, eta)
    point_mean, point_deviation = anova_components(
        point_bt, point_bf, jnp.asarray(g), jnp.asarray(eta), paired=True
    )
    expected = (mean[..., None, :] + deviation).reshape(2, 3, -1)[..., indices]
    np.testing.assert_allclose(
        point_mean + point_deviation, expected, atol=1e-14
    )


def test_paired_compression_permutation_and_masks_retain_reference():
    rng = np.random.default_rng(480)
    z = rng.normal(size=(12, 2, 5)) + 1j * rng.normal(size=(12, 2, 5))
    time, frequency = (
        np.repeat(np.linspace(0, 1, 3), 4),
        np.tile(np.linspace(0.1, 1, 4), 3),
    )
    data = WishartGridData.from_scattered_coefficients(z, time, frequency)
    np.testing.assert_allclose(
        data.Y, z @ z.conj().swapaxes(-1, -2), rtol=1e-13, atol=1e-13
    )
    assert data.U.shape == (12, 2, 2) and np.all(data.counts == 5)
    permutation = rng.permutation(12)
    permuted = replace(
        data,
        u_re=data.u_re[permutation],
        u_im=data.u_im[permutation],
        time=data.time[permutation],
        frequency=data.frequency[permutation],
        counts=data.counts[permutation],
    )
    observed = np.arange(12) % 3 != 0
    masked = permuted.mask(observed)
    np.testing.assert_allclose(
        masked.Y[observed], data.Y[permutation][observed]
    )
    assert np.all(masked.U[~observed] == 0) and np.all(
        masked.counts[~observed] == 0
    )
    np.testing.assert_array_equal(masked.reference_time, data.reference_time)
    np.testing.assert_array_equal(
        masked.reference_frequency, data.reference_frequency
    )
    np.testing.assert_allclose(
        _spline(masked).time_basis.mean(axis=0), 0, atol=1e-14
    )
    with pytest.raises(ValueError, match="grid observations"):
        coarse_grain_wishart_grid(masked)


@pytest.mark.parametrize(
    "change",
    [
        {"counts": 1.5},
        {"counts": np.ones((12, 1))},
        {"time": np.zeros(11)},
        {"frequency": np.full(12, np.nan)},
        {"reference_time": np.array([0.2, 1])},
    ],
)
def test_invalid_paired_statistics_rejected(change):
    with pytest.raises(ValueError):
        replace(_paired(_grid()), **change)


def test_invalid_transform_inputs_rejected():
    with pytest.raises(ValueError, match="proper complex"):
        WishartGridData.from_scattered_coefficients(
            np.ones((12, 2)), np.arange(12), np.ones(12)
        )
    with pytest.raises((ValueError, TypeCheckError)):
        tang_moving_periodogram(np.ones((20, 2), dtype=complex), m=2)
    for time in (
        np.r_[np.arange(19), 18],
        np.arange(20) ** 2,
        np.r_[np.arange(19), np.inf],
    ):
        with pytest.raises(ValueError, match="uniformly sampled"):
            multivariate_moving_periodogram(
                TimeSeries(np.ones((20, 2)), time), m=2
            )
    with pytest.raises(ValueError, match="short"):
        multivariate_moving_periodogram(TimeSeries(np.ones((5, 2))), m=4)


def test_scattered_initialization_crosses_chunk_boundary_and_matches_qr():
    q = 4099
    rng = np.random.default_rng(580)
    data = _paired(_grid())
    spline, config = _spline(data), PowerConfig(structure="anova")
    pair = prepare_anova_prior(spline, config)
    time, frequency = rng.uniform(0, 1, q), rng.uniform(0.1, 1, q)
    bt, bf = spline.design(time, frequency)
    counts = rng.integers(1, 4, q).astype(float)
    counts[0] = 0
    target = 0.5 + 0.2 * time - 0.3 * frequency + 0.1 * time * frequency
    marginal = counts * np.exp(target)
    marginal[0] = np.nan
    actual = _initialize_scattered(marginal, counts, bt, bf, pair, config)
    active = counts > 0
    bt_eig, bf_eig = bt[active] @ pair["U_time"], bf[active] @ pair["U_freq"]
    design = np.column_stack(
        (
            bf_eig,
            np.array(
                [np.kron(t, f) for t, f in zip(bt_eig, bf_eig, strict=True)]
            ),
        )
    )
    penalty = (
        np.r_[
            config.init_penalty_freq * pair["lam_freq"],
            (
                config.init_penalty_time * pair["lam_time"][:, None]
                + config.init_penalty_freq * pair["lam_freq"][None]
            ).reshape(-1),
        ]
        + config.ridge_eps
    )
    weights = counts[active] / counts.max()
    from log_psplines.inference.power import power_floor

    mean = marginal[active] / counts[active]
    expected = np.linalg.lstsq(
        np.vstack(
            (np.sqrt(weights)[:, None] * design, np.diag(np.sqrt(penalty)))
        ),
        np.r_[
            np.sqrt(weights) * np.log(mean + power_floor(mean)), np.zeros(6)
        ],
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        np.r_[actual["g"], actual["eta"]], expected, atol=2e-12
    )
    permutation = rng.permutation(q)
    reordered = _initialize_scattered(
        marginal[permutation],
        counts[permutation],
        bt[permutation],
        bf[permutation],
        pair,
        config,
    )
    for name in actual:
        np.testing.assert_allclose(actual[name], reordered[name], atol=2e-12)
    assert all(
        np.isfinite(value).all() for value in map(np.asarray, actual.values())
    )


def test_public_paired_fit_preview_and_native_roundtrip(tmp_path):
    rng = np.random.default_rng(680)
    n, dt, offset = 83, 0.25, 10
    values = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.3], [0.0, 0.8]])
    reference = offset + dt * np.linspace(0, n - 1, 5)
    data = multivariate_moving_periodogram(
        TimeSeries(values, offset + dt * np.arange(n)),
        m=4,
        reference_time=reference,
    )
    order = rng.permutation(len(data.counts))
    data = replace(
        data,
        u_re=data.u_re[order],
        u_im=data.u_im[order],
        time=data.time[order],
        frequency=data.frequency[order],
        counts=data.counts[order],
    )
    result = fit(
        data,
        PowerConfig(
            structure="anova",
            degree_time=1,
            degree_freq=1,
            penalty_order_time=1,
            penalty_order_freq=1,
            n_interior_knots_time=0,
            n_interior_knots_freq=0,
            n_warmup=3,
            n_samples=4,
            max_tree_depth=2,
            spectrum_draws=1,
            spectrum_chunk_size=2,
            progress_bar=False,
            seed=680,
        ),
    )
    assert all(np.isfinite(var).all() for var in result.posterior.values())
    assert np.linalg.eigvalsh(result.spectrum).min() > 0
    np.testing.assert_allclose(
        result.spectrum,
        result.spectrum.conj().transpose(
            "chain", "draw", "time", "frequency", "channel_aux", "channel"
        ),
        atol=1e-14,
    )
    assert result.coherence.min() >= 0 and result.coherence.max() <= 1 + 1e-12
    assert (
        result.spectrum.sizes["draw"] == 1
        and result.posterior.sizes["draw"] == 4
    )
    full = wishart_grid_draws(
        result.posterior,
        result.model_data.basis_time.values,
        result.model_data.basis_frequency.values,
        2,
    )
    for kind in ("complex", "magnitude", "coherence"):
        np.testing.assert_allclose(
            result.quantiles(kind=kind),
            spectral_quantiles(full, kind=kind, axis=(0, 1)),
            rtol=1e-12,
        )
    result.to_netcdf(tmp_path / "moving.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "moving.nc")
    np.testing.assert_array_equal(loaded.time, reference)
    assert loaded.observed_data.u_re.dims == (
        "observation",
        "channel",
        "factor",
    )
    assert (
        loaded.observed_data.time.dims
        == loaded.observed_data.frequency.dims
        == ("observation",)
    )
    assert loaded.observed_data.identical(result.observed_data)
    assert loaded.to_arviz().attrs["data_type"] == "multivariate_gridtv"
    assert len(sampling_diagnostics(loaded)["nuts"]) == 2
