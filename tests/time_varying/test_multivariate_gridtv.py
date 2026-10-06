"""Independent complex-Gaussian mathematics and public GridTV smoke runs."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro import handlers
from numpyro.infer.util import log_density

from log_psplines import (
    ANOVALogPSpline,
    PowerConfig,
    PSDResult,
    SpectralMatrix,
    SplineBasis,
    TimeSeries,
    WishartGridData,
    coarse_grain_wishart_grid,
    fit,
    local_wishart_grid,
)
from log_psplines.data.spectral_utils import Y_to_U
from log_psplines.diagnostics.sampling import (
    _channel_idata,
    _tree_depth_hits,
    sampling_diagnostics,
)
from log_psplines.diagnostics.spectrum import spectrum_diagnostics
from log_psplines.inference.anova_power import (
    anova_init_values,
    collect_anova_samples,
    prepare_anova_power_model,
    prepare_anova_prior,
)
from log_psplines.inference.power_results import power_result_spectra
from log_psplines.inference.wishart_grid import (
    prepare_wishart_grid_row,
    row_field_labels,
)
from log_psplines.likelihoods.whittle import (
    power_whittle_log_likelihood,
    whittle_log_likelihood,
)
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.anova import anova_components
from log_psplines.models.reconstruction import wishart_grid_draws
from log_psplines.plotting.results import plot_posterior_spectrum


def _fixture(channels=3, coherent=True):
    """Seeded proper Gaussians under independently specified HPD truth."""
    rng = np.random.default_rng(913)
    time, frequency = np.linspace(0, 1, 5), np.linspace(0.1, 1, 7)
    base = rng.normal(size=(channels, channels)) + 1j * rng.normal(
        size=(channels, channels)
    )
    base = base @ base.conj().T / channels + np.eye(channels)
    if not coherent:
        base = np.diag(np.diag(base).real)
    amplitude = 1 + 0.2 * time[:, None] + 0.1 * frequency[None, :]
    truth = amplitude[..., None, None] * base
    counts = (np.arange(35).reshape(5, 7) % 4) + 1
    counts[0, 0] = 0
    z = (
        rng.normal(size=(5, 7, channels, 4))
        + 1j * rng.normal(size=(5, 7, channels, 4))
    ) / np.sqrt(2)
    coefficients = np.linalg.cholesky(truth) @ z
    coefficients *= (np.arange(4) < counts[..., None])[..., None, :]
    return (
        WishartGridData(
            coefficients.real, coefficients.imag, time, frequency, counts
        ),
        truth,
        coefficients,
    )


def _components(spectrum):
    """Independent modified-Cholesky conversion for test likelihood inputs."""
    lower = np.linalg.cholesky(spectrum)
    diagonal = np.diagonal(lower, axis1=-2, axis2=-1).real
    triangular = np.linalg.inv(lower / diagonal[..., None, :])
    return 2 * np.log(diagonal), -triangular


def _blocked(data, logs, theta):
    return sum(
        wishart_log_likelihood(
            jnp.asarray(logs[..., row]),
            jnp.asarray(theta[..., row, :row].real),
            jnp.asarray(theta[..., row, :row].imag),
            data.u_re[..., row, :],
            data.u_im[..., row, :],
            data.u_re[..., :row, :],
            data.u_im[..., :row, :],
            counts=data.counts,
            clip_log_psd=False,
        )
        for row in range(data.p)
    )


def _direct(data, spectrum):
    return -np.sum(
        data.counts * np.linalg.slogdet(spectrum)[1]
        + np.trace(np.linalg.solve(spectrum, data.Y), axis1=-2, axis2=-1).real
    )


def _spline(data):
    return ANOVALogPSpline(
        SplineBasis.from_grid(
            data.reference_frequency, 0, degree=1, penalty_order=1
        ),
        SplineBasis.from_grid(
            data.reference_time, 0, degree=1, penalty_order=1
        ),
    )


def test_direct_matrix_likelihood_and_gradients():
    data, truth, _ = _fixture()
    logs, theta = _components(truth)
    np.testing.assert_allclose(
        _blocked(data, logs, theta), _direct(data, truth), rtol=2e-6
    )

    # Differentiate an independent matrix logdet/solve implementation.
    def direct(logs):
        inverse = jnp.linalg.inv(
            jnp.eye(data.p) - jnp.tril(jnp.asarray(theta), -1)
        )
        spectrum = (inverse * jnp.exp(logs)[..., None, :]) @ jnp.swapaxes(
            inverse.conj(), -1, -2
        )
        return -jnp.sum(
            jnp.asarray(data.counts) * jnp.linalg.slogdet(spectrum)[1]
            + jnp.trace(
                jnp.linalg.solve(spectrum, jnp.asarray(data.Y)),
                axis1=-2,
                axis2=-1,
            ).real
        )

    actual = jax.grad(lambda value: _blocked(data, value, theta))(
        jnp.asarray(logs)
    )
    expected = jax.grad(direct)(jnp.asarray(logs))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)


def test_fixed_pooling_preserves_sums_counts_edges_and_likelihood():
    data, _, _ = _fixture()
    pooled = coarse_grain_wishart_grid(data, time_bin=2, frequency_bin=3)
    assert pooled.U.shape == (3, 3, 3, 3)
    assert pooled.counts.sum() == data.counts.sum()
    assert pooled.counts[-1, -1] == data.counts[-1, -1]
    np.testing.assert_array_equal(pooled.reference_time, data.time)
    np.testing.assert_array_equal(pooled.reference_frequency, data.frequency)
    np.testing.assert_allclose(pooled.time, [0.125, 0.625, 1])
    np.testing.assert_allclose(pooled.frequency, [0.25, 0.7, 1])
    rng = np.random.default_rng(9)
    logs = rng.normal(scale=0.2, size=(3, 3, 3))
    theta = rng.normal(scale=0.1, size=(3, 3, 3, 3)) + 1j * rng.normal(
        scale=0.1, size=(3, 3, 3, 3)
    )
    fine_logs = np.repeat(np.repeat(logs, 2, axis=0), 3, axis=1)[:5, :7]
    fine_theta = np.repeat(np.repeat(theta, 2, axis=0), 3, axis=1)[:5, :7]
    np.testing.assert_allclose(
        _blocked(data, fine_logs, fine_theta),
        _blocked(pooled, logs, theta),
        rtol=2e-6,
    )
    twice = coarse_grain_wishart_grid(pooled, time_bin=3, frequency_bin=3)
    np.testing.assert_allclose(
        twice.Y[0, 0], data.Y.sum(axis=(0, 1)), rtol=1e-12, atol=1e-12
    )
    assert twice.counts[0, 0] == data.counts.sum()
    assert coarse_grain_wishart_grid(data) is data


def test_compression_rank_deficiency_scalar_counts_and_omitted_cells():
    data, truth, coefficients = _fixture()
    np.testing.assert_allclose(
        data.Y, coefficients @ coefficients.conj().swapaxes(-1, -2), atol=2e-12
    )
    assert data.U.shape[-1] == data.p < coefficients.shape[-1]
    logs, theta = _components(truth)
    native_sum = sum(
        wishart_log_likelihood(
            jnp.asarray(logs[..., row]),
            theta[..., row, :row].real,
            theta[..., row, :row].imag,
            coefficients[..., row, :].real,
            coefficients[..., row, :].imag,
            coefficients[..., :row, :].real,
            coefficients[..., :row, :].imag,
            counts=data.counts,
            clip_log_psd=False,
        )
        for row in range(data.p)
    )
    np.testing.assert_allclose(
        _blocked(data, logs, theta), native_sum, rtol=2e-6
    )
    single = WishartGridData.from_coefficients(
        coefficients[..., 0], data.time, data.frequency
    )
    assert single.U.shape[-1] == 1
    assert np.isfinite(_blocked(single, logs, theta))
    assert np.linalg.matrix_rank(single.Y[-1, -1]) == 1
    observed = np.ones(data.counts.shape, dtype=bool)
    observed[2, 2] = False
    masked = data.mask(observed)
    assert masked.counts[2, 2] == 0 and np.all(masked.U[2, 2] == 0)
    np.testing.assert_array_equal(masked.reference_time, data.reference_time)


def test_omitted_nonfinite_inputs_have_zero_value_and_gradient():
    logs = jnp.array([[np.nan, 0.1]])
    theta = jnp.array([[[np.nan], [0.2]]])
    factors = jnp.array([[[np.nan], [0.7]]])
    previous = jnp.array([[[[np.nan]], [[0.3]]]])
    counts = jnp.array([[0.0, 1.0]])

    def likelihood(logs, theta):
        return wishart_log_likelihood(
            logs,
            theta,
            theta,
            factors,
            factors,
            previous,
            previous,
            counts=counts,
            clip_log_psd=False,
        )

    value, gradients = jax.value_and_grad(likelihood, argnums=(0, 1))(
        logs, theta
    )
    assert np.isfinite(value)
    for gradient in gradients:
        assert np.isfinite(gradient).all()
        assert np.all(np.asarray(gradient)[:, 0] == 0)
    data = WishartGridData(
        np.asarray(factors[..., None]),
        np.asarray(factors[..., None]),
        np.array([0]),
        np.array([0.1, 0.2]),
        np.asarray(counts),
    )
    assert np.all(data.U[0, 0] == 0)


def test_scalar_reduction_under_matched_complex_normalization():
    data, truth, _ = _fixture(channels=1)
    logs = np.log(truth[..., 0, 0].real)
    power = data.Y[..., 0, 0].real
    actual = _blocked(
        data, logs[..., None], np.zeros((*logs.shape, 1, 1), dtype=complex)
    )
    # Scalar PowerData counts REAL components: two per proper complex vector.
    np.testing.assert_allclose(
        actual,
        power_whittle_log_likelihood(2 * power, 2 * data.counts, logs),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        actual,
        whittle_log_likelihood(
            logs, power, count=data.counts, clip_log_psd=False
        ),
        rtol=2e-6,
    )
    empty_theta = np.zeros((*logs.shape, 0))
    scalar_count = wishart_log_likelihood(
        logs,
        empty_theta,
        empty_theta,
        data.u_re[..., 0, :],
        data.u_im[..., 0, :],
        data.u_re[..., :0, :],
        data.u_im[..., :0, :],
        counts=np.asarray(3.0),
        clip_log_psd=False,
    )
    np.testing.assert_allclose(
        scalar_count,
        whittle_log_likelihood(logs, power, count=3, clip_log_psd=False),
        rtol=2e-6,
    )
    # The historical stationary duration convention remains unchanged.
    np.testing.assert_allclose(
        whittle_log_likelihood(logs, 4 * power, count=data.counts, duration=4),
        actual,
        rtol=2e-6,
    )


def test_matrix_sign_conjugation_and_time_leading_axes():
    logs = np.broadcast_to(np.log([2.0, 3.0]), (3, 4, 2))
    theta = np.full((3, 4, 1), 0.2 + 0.4j)
    spectrum = SpectralMatrix(2)(logs, theta.real, theta.imag)
    expected = np.array(
        [
            [2, 2 * (0.2 - 0.4j)],
            [2 * (0.2 + 0.4j), 3 + 2 * abs(0.2 + 0.4j) ** 2],
        ]
    )
    np.testing.assert_allclose(
        spectrum, np.broadcast_to(expected, spectrum.shape)
    )
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(spectrum).min() > 0
    coherence = SpectralMatrix.coherence(spectrum)
    assert coherence.min() >= 0 and coherence.max() <= 1 + 1e-12
    diagonal = SpectralMatrix(2)(logs, theta.real * 0, theta.imag * 0)
    np.testing.assert_allclose(
        diagonal, np.broadcast_to(np.diag([2, 3]), diagonal.shape)
    )
    assert SpectralMatrix(1)(logs[..., :1]).shape == (3, 4, 1, 1)


def test_fixed_centring_chunked_evaluation_and_independent_row_sites():
    data, _, _ = _fixture(channels=2)
    spline = _spline(data)
    config = PowerConfig(structure="anova", centered=True)
    pair = prepare_anova_prior(spline, config)
    pooled = coarse_grain_wishart_grid(data, time_bin=2, frequency_bin=3)
    rng = np.random.default_rng(55)
    sites = {}
    for row in range(data.p):
        model, init = prepare_wishart_grid_row(
            pooled, spline, config, row, pair=pair
        )
        for label in row_field_labels(row):
            init[f"g_{label}"] = rng.normal(size=2) * 0.1
            init[f"eta_{label}"] = rng.normal(size=2) * 0.1
        trace = handlers.trace(
            handlers.substitute(handlers.seed(model, row), data=init)
        ).get_trace()
        sites.update(
            {
                name: (
                    (
                        "chain",
                        "draw",
                        *[f"{name}_dim_{i}" for i in range(np.ndim(value))],
                    ),
                    np.asarray(value)[None, None],
                )
                for name, value in init.items()
            }
        )
        bt, bf = spline.design(pooled.time, pooled.frequency)
        evaluated = []
        for label in row_field_labels(row):
            g, eta = anova_components(
                bt @ pair["U_time"],
                bf @ pair["U_freq"],
                init[f"g_{label}"],
                init[f"eta_{label}"].reshape(1, 2),
            )
            evaluated.append(g[None, :] + eta)
        logs = evaluated[0]
        re = (
            jnp.stack(evaluated[1::2], axis=-1)
            if row
            else jnp.empty((*logs.shape, 0))
        )
        im = (
            jnp.stack(evaluated[2::2], axis=-1)
            if row
            else jnp.empty((*logs.shape, 0))
        )
        expected = wishart_log_likelihood(
            logs,
            re,
            im,
            pooled.u_re[..., row, :],
            pooled.u_im[..., row, :],
            pooled.u_re[..., :row, :],
            pooled.u_im[..., :row, :],
            counts=pooled.counts,
            clip_log_psd=False,
        )
        np.testing.assert_allclose(
            trace[f"log_likelihood_block_{row}"]["value"], expected
        )
        random_sites = {
            name
            for name, site in trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        assert random_sites == set(init)
    posterior = xr.Dataset(sites)
    for row in range(data.p):
        posterior = collect_anova_samples(
            posterior, pair, labels=row_field_labels(row)
        )
    bt, bf = spline.design()
    np.testing.assert_allclose(bt.mean(axis=0), 0, atol=1e-12)
    full = wishart_grid_draws(posterior, bt, bf, data.p)
    chunks = np.concatenate(
        [
            wishart_grid_draws(posterior, bt, bf[:3], data.p),
            wishart_grid_draws(posterior, bt, bf[3:], data.p),
        ],
        axis=3,
    )
    np.testing.assert_allclose(full, chunks, rtol=1e-6)
    g, eta = spline.components(np.ones(2), np.ones((1, 2)))
    np.testing.assert_allclose(np.asarray(eta).mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(
        spline(np.ones(2), np.zeros((1, 2))), np.broadcast_to(g, eta.shape)
    )
    # A finer output evaluation uses the same transform, rather than a new mean.
    fine_bt, fine_bf = spline.design(
        np.linspace(0, 1, 11), np.linspace(0.1, 1, 13)
    )
    assert wishart_grid_draws(posterior, fine_bt, fine_bf, 2).shape == (
        1,
        1,
        11,
        13,
        2,
        2,
    )


@pytest.mark.parametrize("length", [16, 17])
def test_local_fourier_normalization_and_real_endpoints(length):
    rng = np.random.default_rng(91)
    fs, segments = 16, 256
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    values = rng.multivariate_normal(
        np.zeros(2), covariance, size=segments * length
    )
    series = TimeSeries(values, np.arange(len(values)) / fs)
    grid = local_wishart_grid(series, length)
    assert grid.frequency[0] > 0 and grid.frequency[-1] < fs / 2
    assert grid.U.shape[-1] == 1 and np.all(grid.counts == 1)
    fft = np.fft.rfft(values.reshape(segments, length, 2), axis=1)[
        :, 1 : (-1 if length % 2 == 0 else None)
    ]
    np.testing.assert_allclose(
        grid.U[..., 0], np.sqrt(2 / (fs * length)) * fft
    )
    np.testing.assert_allclose(
        grid.Y.mean(axis=(0, 1)).real, 2 * covariance / fs, rtol=0.12
    )
    np.testing.assert_allclose(
        grid.time,
        np.arange(len(values)).reshape(segments, length).mean(axis=1) / fs,
    )


def test_invalid_statistics_and_incompatible_data_fail_clearly():
    with pytest.raises(ValueError, match="Hermitian"):
        Y_to_U(1e-15 * np.array([[[1, 0.01j], [0, 1]]], dtype=complex))
    roundoff = Y_to_U(np.array([[[1, 0], [0, -1e-15]]], dtype=complex))
    np.testing.assert_allclose(
        roundoff @ roundoff.conj().swapaxes(-1, -2),
        np.array([np.diag([1, 0])]),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        Y_to_U(np.array([[[1, 0], [0, -0.1]]], dtype=complex))
    with pytest.raises(ValueError, match="finite"):
        Y_to_U(np.array([[[np.nan]]], dtype=complex))
    with pytest.raises(ValueError, match="real WDM"):
        WishartGridData.from_coefficients(
            np.ones((3, 3, 2)), np.arange(3), np.arange(3)
        )
    data, _, _ = _fixture(channels=1)
    for counts in (-1, np.nan, 1.5, np.ones((1, 1))):
        with pytest.raises(ValueError, match="counts"):
            WishartGridData(
                data.u_re, data.u_im, data.time, data.frequency, counts
            )
    with pytest.raises(ValueError, match="structure='anova'"):
        fit(data, PowerConfig())
    with pytest.raises(ValueError, match="active complex observation"):
        fit(
            data.mask(np.zeros(data.counts.shape, dtype=bool)),
            PowerConfig(structure="anova"),
            model=_spline(data),
        )
    with pytest.raises(ValueError, match="reference_time"):
        prepare_wishart_grid_row(
            data,
            ANOVALogPSpline(
                _spline(data).frequency,
                SplineBasis.from_grid(
                    data.time[:3], 0, degree=1, penalty_order=1
                ),
            ),
            PowerConfig(),
            0,
        )


@pytest.mark.parametrize(
    "channels,coherent", [(1, False), (2, False), (2, True)]
)
def test_public_nuts_smoke_and_native_results(channels, coherent, outdir):
    data, truth, _ = _fixture(channels=channels, coherent=coherent)
    config = PowerConfig(
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
        progress_bar=False,
        spectrum_draws=2,
        spectrum_chunk_size=3,
    )
    spline = _spline(data)
    pooled = coarse_grain_wishart_grid(data, time_bin=2, frequency_bin=2)
    result = fit(
        pooled, config, model=None if channels == 1 else spline, true_psd=truth
    )
    assert result.spectrum.dims == (
        "chain",
        "draw",
        "time",
        "frequency",
        "channel",
        "channel_aux",
    )
    assert result.spectrum.shape == (1, 2, 5, 7, channels, channels)
    assert result.posterior.sizes["draw"] == 4
    assert np.isfinite(result.spectrum).all()
    assert np.linalg.eigvalsh(result.spectrum).min() > 0
    assert (
        np.min(result.coherence) >= 0 and np.max(result.coherence) <= 1 + 1e-12
    )
    for variable in result.posterior.values():
        assert np.isfinite(variable).all()
    assert result.metadata["row_coefficient_counts"] == [
        4 * (1 + 2 * row) for row in range(channels)
    ]
    assert "posterior" in result.to_arviz().children
    assert result.to_arviz().attrs["data_type"] == "multivariate_gridtv"
    assert len(sampling_diagnostics(result)["nuts"]) == channels
    for row in range(channels):
        assert (
            "diverging" in _channel_idata(result, row)["sample_stats"].dataset
        )
    directory = outdir / f"gridtv-{channels}-{coherent}"
    directory.mkdir(exist_ok=True)
    result.to_netcdf(directory / "result.nc")
    reloaded = PSDResult.from_netcdf(directory / "result.nc")
    assert reloaded.observed_data.attrs == result.observed_data.attrs
    assert reloaded.spectrum_summary.attrs == result.spectrum_summary.attrs
    assert reloaded.metadata["spectrum_draws"] == config.spectrum_draws
    assert not any("__" in key for key in reloaded.metadata)
    np.testing.assert_array_equal(
        reloaded.observed_data["counts"], pooled.counts
    )
    reconstructed = wishart_grid_draws(
        reloaded.posterior,
        np.asarray(reloaded.model_data["basis_time"]),
        np.asarray(reloaded.model_data["basis_frequency"]),
        channels,
    )
    np.testing.assert_allclose(
        reconstructed[:, result.spectrum.draw.values],
        result.spectrum,
        rtol=1e-6,
    )
    np.testing.assert_allclose(reloaded.spectrum, result.spectrum)
    np.testing.assert_allclose(reloaded.quantiles(), result.quantiles())
    plot_posterior_spectrum(reloaded, directory)
    assert (directory / "posterior_spectrum.png").stat().st_size > 0


@pytest.mark.parametrize("retained_draws", [None, 1])
def test_coherence_diagnostics_use_all_draws_before_reduction(retained_draws):
    """Changing phase can cancel a median cross spectrum despite coherence."""
    time, frequency = np.arange(2.0), np.arange(5.0)
    draws = (
        np.broadcast_to(np.eye(2), (1, 5, 2, 5, 2, 2)).astype(complex).copy()
    )
    draws[..., 0, 1] = np.array([0.0, 0.6, -0.6, 0.6, -0.6])[
        None, :, None, None
    ]
    draws[..., 1, 0] = draws[..., 0, 1].conj()
    posterior = xr.Dataset({"x": (("chain", "draw"), np.zeros((1, 5)))})
    spectrum, summary = power_result_spectra(
        posterior,
        lambda section: draws[:, :, :, section],
        time,
        frequency,
        [0, 1],
        PowerConfig(spectrum_draws=retained_draws, spectrum_chunk_size=2),
        matrix=True,
    )
    truth = draws[0, 1]
    result = PSDResult(posterior, spectrum, spectrum_summary=summary)
    assert np.all(result.quantiles((50.0,)).values[..., 0, 1] == 0)
    assert spectrum_diagnostics(result, truth=truth)[
        "coherence_mae"
    ] == pytest.approx(0)
    if retained_draws is None:
        # The uncached stationary path uses the same per-draw definition.
        stationary = PSDResult(posterior, spectrum.isel(time=0, drop=True))
        assert spectrum_diagnostics(stationary, truth=truth[0])[
            "coherence_mae"
        ] == pytest.approx(0)


def test_nuts_step_count_reports_tree_depth_saturation():
    stats = xr.Dataset(
        {"n_steps": (("chain", "draw"), [[127, 254, 255, 255]])}
    )
    view = xr.DataTree.from_dict({"sample_stats": stats})
    assert _tree_depth_hits(view, 8) == 2
    assert _tree_depth_hits(view, 7) == 4


def test_host_reconstruction_keeps_float64_physical_range():
    # Inference precision is configured separately; host spectra remain float64.
    g, eta = anova_components(
        np.ones((2, 1)), np.ones((3, 1)), np.array([-110.0]), np.zeros((1, 1))
    )
    assert isinstance(g, np.ndarray) and g.dtype == np.float64
    spectrum = SpectralMatrix(1)(
        (g[None, :] + eta)[..., None].astype(np.float32)
    )
    assert np.all(spectrum.real > 0)
    np.testing.assert_allclose(spectrum.real, np.exp(-110), rtol=1e-12, atol=0)


@pytest.mark.parametrize("sigma_eta", [0.4, 1e-5])
def test_anova_noncentering_preserves_fields_and_prior_density(sigma_eta):
    """Same physical coefficients, including a narrow funnel and its Jacobian."""
    data, _, _ = _fixture(channels=2)
    spline = _spline(data)
    config = PowerConfig(structure="anova", centered=True)
    pair = prepare_anova_prior(spline, config)
    centered, init = prepare_wishart_grid_row(data, spline, config, 1)
    config = replace(config, centered=False)
    noncentered, _ = prepare_wishart_grid_row(data, spline, config, 1)
    labels = row_field_labels(1)
    coordinates = {}
    for label in labels:
        suffix = f"_{label}" if label else ""
        physical = {
            name: init[f"{name}{suffix}"]
            for name in ("g", "eta", "sigma_g", "sigma_eta")
        }
        physical["sigma_eta"] = sigma_eta
        physical["eta"] = np.full_like(physical["eta"], sigma_eta * 0.1)
        init.update(
            {f"{name}{suffix}": value for name, value in physical.items()}
        )
        coordinates.update(
            anova_init_values(physical, pair, config, label=label)
        )
    centered_density, centered_trace = log_density(centered, (), {}, init)
    raw_density, raw_trace = log_density(noncentered, (), {}, coordinates)
    log_jacobian = 0.0
    for label in labels:
        suffix = f"_{label}" if label else ""
        for field in ("g", "eta"):
            name = f"{field}{suffix}"
            np.testing.assert_allclose(
                raw_trace[name]["value"],
                centered_trace[name]["value"],
                rtol=2e-6,
            )
            raw_site = raw_trace[f"z_{name}"]
            np.testing.assert_allclose(
                raw_site["fn"].log_prob(raw_site["value"]),
                -0.5 * np.asarray(raw_site["value"]) ** 2
                - 0.5 * np.log(2 * np.pi),
                rtol=2e-6,
            )
            log_jacobian += np.log(centered_trace[name]["fn"].scale).sum()
    np.testing.assert_allclose(
        raw_density, centered_density + log_jacobian, rtol=2e-6, atol=2e-5
    )


@pytest.mark.slow
def test_complex_grid_recovery_with_informative_replicates():
    """Recover an independent smooth HPD truth with enough data per cell."""
    rng = np.random.default_rng(45)
    time, frequency = np.linspace(0, 1, 5), np.linspace(0.1, 1, 7)
    base = np.array([[1.2, 0.3 + 0.2j], [0.3 - 0.2j, 0.9]])
    amplitude = np.exp(0.3 * (time[:, None] - 0.5) + 0.2 * frequency[None, :])
    truth = amplitude[..., None, None] * base
    noise = (
        rng.normal(size=(5, 7, 2, 64)) + 1j * rng.normal(size=(5, 7, 2, 64))
    ) / np.sqrt(2)
    coefficients = np.linalg.cholesky(truth) @ noise
    data = WishartGridData.from_coefficients(coefficients, time, frequency)
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
            n_warmup=200,
            n_samples=250,
            num_chains=2,
            seed=45,
            target_accept_prob=0.95,
            spectrum_draws=10,
            progress_bar=False,
        ),
    )
    median = result.quantiles((50.0,)).values[0]
    diagonal = np.diagonal(truth, axis1=-2, axis2=-1).real
    estimated = np.diagonal(median, axis1=-2, axis2=-1).real
    assert np.sqrt(np.mean(np.log(estimated / diagonal) ** 2)) < 0.08
    residual = (median[..., 0, 1] - truth[..., 0, 1]) / np.sqrt(
        diagonal.prod(axis=-1)
    )
    assert np.sqrt(np.mean(abs(residual) ** 2)) < 0.06
    coherence = result.spectrum_summary.coherence_quantiles.sel(
        percentile=50
    ).values
    assert (
        np.sqrt(np.mean((coherence - SpectralMatrix.coherence(truth)) ** 2))
        < 0.07
    )


def test_gridtv_rejects_vi_before_sampling(monkeypatch):
    """Public and internal entry points never silently switch VI to NUTS."""
    from log_psplines.inference import wishart_grid

    data, _, _ = _fixture(channels=2)
    config = PowerConfig(structure="anova", method="vi")

    def unexpected_sampling(*args, **kwargs):
        pytest.fail("unsupported VI must be rejected before NUTS")

    monkeypatch.setattr(wishart_grid, "run_nuts", unexpected_sampling)
    with pytest.raises(ValueError, match="supports method='nuts' only"):
        fit(data, config)
    with pytest.raises(ValueError, match="supports method='nuts' only"):
        wishart_grid.fit_wishart_grid(data, _spline(data), config)


@pytest.mark.parametrize("centered", [False, True])
def test_scalar_anova_retains_centered_sample_sites(centered):
    """Matrix coordinates do not alter the existing scalar VI model."""
    from log_psplines import PowerData

    data, _, _ = _fixture(channels=1)
    powers = PowerData(
        2 * data.Y[..., 0, 0].real, 2 * data.counts, data.frequency, data.time
    )
    model, _, init = prepare_anova_power_model(
        powers, _spline(data), PowerConfig(centered=centered)
    )
    _, trace = log_density(model, (), {}, init)
    assert trace["g"]["type"] == trace["eta"]["type"] == "sample"
    assert "z_g" not in trace and "z_eta" not in trace


def test_matrix_preview_preserves_all_chain_nonlinear_summaries(tmp_path):
    """Preview draws cannot replace all-draw nonlinear summaries."""
    from log_psplines.models.reconstruction import spectral_quantiles

    time, frequency = np.linspace(0, 1, 3), np.linspace(0.1, 1, 4)
    posterior = xr.Dataset({"dummy": (("chain", "draw"), np.zeros((2, 4)))})
    phase = np.arange(8).reshape(2, 4) * np.pi / 4
    draws = np.zeros((2, 4, 3, 4, 2, 2), dtype=complex)
    draws[..., 0, 0] = 1
    draws[..., 1, 1] = 1.5
    draws[..., 0, 1] = 0.5 * np.exp(1j * phase[:, :, None, None])
    draws[..., 1, 0] = draws[..., 0, 1].conj()
    spectrum, summary = power_result_spectra(
        posterior,
        lambda section: draws[:, :, :, section],
        time,
        frequency,
        [0, 1],
        PowerConfig(spectrum_draws=1, spectrum_chunk_size=2),
        matrix=True,
    )
    observed = xr.Dataset(
        {"counts": (("time", "frequency"), np.ones((3, 4)))},
        coords={"time": time, "frequency": frequency},
        attrs={
            "units": "test covariance units",
            "normalization": "complex_covariance",
        },
    )
    result = PSDResult(
        posterior=posterior,
        spectrum=spectrum,
        spectrum_summary=summary,
        observed_data=observed,
    )
    result.to_netcdf(tmp_path / "matrix-preview.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "matrix-preview.nc")
    assert loaded.observed_data.attrs == observed.attrs
    assert loaded.spectrum.sizes["chain"] == 2
    assert loaded.spectrum.sizes["draw"] == 1
    for kind in ("complex", "magnitude", "coherence"):
        expected = spectral_quantiles(draws, kind=kind, axis=(0, 1))
        np.testing.assert_allclose(loaded.quantiles(kind=kind), expected)
    np.testing.assert_allclose(
        loaded.quantiles(kind="magnitude")[1, ..., 0, 1], 0.5
    )
    assert np.max(abs(loaded.quantiles()[1, ..., 0, 1])) < 1e-14
