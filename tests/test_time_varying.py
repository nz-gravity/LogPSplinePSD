"""Mathematical contracts for time-frequency power representations."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines import (
    LogPSpline,
    PowerConfig,
    PowerData,
    PowerPartition,
    SplineBasis,
    TimeSeries,
    coarse_grain_power,
    fit,
    mask_power,
    moving_periodogram,
    scattered_moving_periodogram,
)
from log_psplines.basis.penalty import eigen_prior_scale, whiten_penalty_pair
from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.preprocessing.knot_locator import (
    Component,
    allocate_components,
)
from log_psplines.preprocessing.moving_periodogram import (
    bin_tang_ordinates,
    tang_moving_periodogram,
)
from log_psplines.preprocessing.wdm import wdm_periodogram


def _definition_one(x: np.ndarray, t: int, m: int) -> tuple[complex, float]:
    j = 1 + ((t - 1) % m)
    lam = 2.0 * j / (2 * m + 1)
    nu = np.arange(2 * m + 1)
    window = x[t - m - 1 : t + m]
    coeff = np.sum(window * np.exp(-1j * np.pi * nu * lam))
    return coeff / np.sqrt(2.0 * np.pi * (2 * m + 1)), np.pi * lam


def test_moving_periodogram_matches_definition_and_pools_power_counts():
    raw = tang_moving_periodogram(
        np.random.default_rng(4).standard_normal(127), m=4, thin=2
    )
    centres = np.rint(raw["u"] * 127).astype(int)
    # Evaluate the mathematical definition on the same realization.
    x = np.random.default_rng(4).standard_normal(127)
    expected_coeff = np.asarray(
        [_definition_one(x, int(t), 4)[0] for t in centres]
    )
    expected_omega = np.asarray(
        [_definition_one(x, int(t), 4)[1] for t in centres]
    )
    np.testing.assert_allclose(raw["coeff"], expected_coeff, atol=1e-12)
    np.testing.assert_allclose(raw["omega"], expected_omega, atol=1e-12)

    pooled = bin_tang_ordinates(raw, time_bin=2, freq_bin=2)
    np.testing.assert_allclose(
        pooled["summed_power"].sum(), 2 * raw["mi"].sum()
    )
    np.testing.assert_allclose(pooled["counts"].sum(), 2 * raw["mi"].size)
    assert np.all(pooled["counts"] > 0)


def test_masked_partition_pooling_conserves_power_counts_and_block_likelihood():
    time = np.array([0.0, 1.0, 2.0, 7.0, 8.0])
    frequency = np.array([0.1, 0.2, 0.3, 0.4])
    power = np.arange(1, 21, dtype=float).reshape(5, 4)
    counts = np.ones_like(power)
    counts[0, 0] = 2
    mask = np.ones_like(power, dtype=bool)
    mask[1, 2] = False
    mask[3, :] = False
    native = mask_power(PowerData(power, counts, frequency, time), mask)
    pooled = coarse_grain_power(
        native, PowerPartition(np.array([0, 2, 3, 4]), np.array([0, 2]))
    )
    np.testing.assert_allclose(pooled.power.sum(), native.power.sum())
    np.testing.assert_allclose(pooled.counts.sum(), native.counts.sum())

    log_s = 0.4
    native_ll = power_whittle_log_likelihood(
        native.power, native.counts, np.full_like(native.power, log_s)
    )
    coarse_ll = power_whittle_log_likelihood(
        pooled.power, pooled.counts, np.full_like(pooled.power, log_s)
    )
    np.testing.assert_allclose(native_ll, coarse_ll)


def test_surface_basis_stationary_limit_and_power_likelihood_gradient():
    frequency = np.linspace(0.1, 1.0, 6)
    time = np.linspace(0.0, 1.0, 5)
    freq_basis = SplineBasis.from_grid(frequency, 3)
    time_basis = SplineBasis.from_grid(time, 1)
    surface_model = LogPSpline(freq_basis, time=time_basis)
    stationary = LogPSpline(freq_basis)
    weights = jnp.arange(freq_basis.basis.shape[1], dtype=float) / 10
    coefficients = (
        jnp.arange(time_basis.basis.shape[1] * weights.size).reshape(
            time_basis.basis.shape[1], weights.size
        )
        / 20
    )
    surface = surface_model(coefficients)
    reference = (
        np.asarray(time_basis.basis)
        @ np.asarray(coefficients)
        @ np.asarray(freq_basis.basis).T
    )
    np.testing.assert_allclose(surface, reference, atol=1e-6)

    constant_surface = surface_model(
        jnp.broadcast_to(weights, (time_basis.basis.shape[1], weights.size))
    )
    np.testing.assert_allclose(
        constant_surface,
        np.broadcast_to(stationary(weights), constant_surface.shape),
    )
    power = (
        jnp.arange(1, time.size * frequency.size + 1).reshape(
            time.size, frequency.size
        )
        / 10
    )
    counts = jnp.ones_like(power)
    actual = power_whittle_log_likelihood(power, counts, surface)
    expected = -0.5 * np.sum(
        np.asarray(counts) * reference + np.asarray(power) * np.exp(-reference)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    gradient = jax.grad(
        lambda value: power_whittle_log_likelihood(power, counts, value)
    )(surface)
    assert np.isfinite(gradient).all()


def test_variation_knots_keep_component_bases_separate():
    time = np.linspace(0.0, 1.0, 21)
    frequency = np.linspace(0.1, 1.0, 61)
    interaction = (time[:, None] - 0.5) * frequency[None, :] ** 2
    knots = allocate_components(
        {
            "g": Component(frequency, {"frequency": frequency}),
            "eta": Component(
                interaction, {"time": time, "frequency": frequency}
            ),
        },
        {"g": {"frequency": 3}, "eta": {"time": 2, "frequency": 8}},
        {"g": {"frequency": 0.01}, "eta": {"time": 0.1, "frequency": 0.09}},
    )
    np.testing.assert_allclose(
        knots["g"]["frequency"], np.linspace(0.1, 1.0, 5)[1:-1]
    )
    assert not np.allclose(
        knots["eta"]["frequency"], np.linspace(0.1, 1.0, 10)[1:-1]
    )
    assert (
        np.min(
            np.diff(
                np.r_[frequency[0], knots["eta"]["frequency"], frequency[-1]]
            )
        )
        >= 0.09 - 1e-14
    )
    model = LogPSpline(
        SplineBasis.from_grid(
            frequency, interior_knots=knots["eta"]["frequency"]
        ),
        time=SplineBasis.from_grid(time, interior_knots=knots["eta"]["time"]),
    )
    assert model().shape == (time.size, frequency.size)


def test_tensor_penalty_prior_uses_additive_marginal_precision():
    penalty_time = np.diag([0.0, 2.0])
    penalty_frequency = np.diag([0.0, 3.0, 5.0])
    eigensystem = whiten_penalty_pair(penalty_time, penalty_frequency)
    scale = np.asarray(
        eigen_prior_scale(
            2.0,
            4.0,
            eigensystem["lam_time"],
            eigensystem["lam_freq"],
            eigensystem["joint_null"],
        )
    )
    expected_precision = (
        2 * np.array([0.0, 2.0])[:, None]
        + 4 * np.array([0.0, 3.0, 5.0])[None, :]
    )
    expected = 1 / np.sqrt(expected_precision + 1e-6)
    expected[0, 0] = 1 / np.sqrt(1e-4)
    np.testing.assert_allclose(scale, expected)
    assert eigensystem["joint_null"].sum() == 1


def test_scattered_time_frequency_data_reaches_public_fit():
    x = np.random.default_rng(21).standard_normal(95)
    data = scattered_moving_periodogram(x, dt=0.25, m=4, thin=2)
    model = LogPSpline(
        SplineBasis.from_grid(
            np.linspace(data.frequency.min(), data.frequency.max(), 5),
            1,
            degree=1,
            penalty_order=1,
        ),
        time=SplineBasis.from_grid(
            np.linspace(data.time.min(), data.time.max(), 5),
            1,
            degree=1,
            penalty_order=1,
        ),
    )
    result = fit(
        data,
        PowerConfig(n_warmup=2, n_samples=2, seed=11, progress_bar=False),
        model=model,
    )
    assert result.observed_data is not None
    assert result.observed_data.sizes["ordinate"] == data.power.size
    assert np.isfinite(result.psd).all()
    assert np.all(result.psd > 0)


def test_ls2_wdm_posterior_tracks_analytic_surface(outdir):
    pytest.importorskip("wdm_transform")
    example = LS2Data(n_samples=4096, fs=64.0, seed=42)
    data = wdm_periodogram(
        TimeSeries(example.data[:, 0], t=example.time), nt=64
    )
    model = LogPSpline(
        frequency=SplineBasis.from_grid(
            data.frequency / data.frequency[-1], 8
        ),
        time=SplineBasis.from_grid(data.time, 8),
    )
    result = fit(
        data,
        PowerConfig(
            n_warmup=150,
            n_samples=150,
            num_chains=2,
            seed=42,
            max_tree_depth=7,
            progress_bar=False,
        ),
        model=model,
    )
    truth = example.get_true_psd(
        time_grid=result.time, freq_grid=result.frequency
    )
    assert result.psd.shape == (2, 150, *truth.shape)
    assert np.isfinite(result.psd).all()
    assert np.all(result.psd > 0)
    assert np.all(truth > 0)

    # WDM powers and analytic PSDs use different units. Remove one constant
    # log offset to compare shape, as in the time-varying tutorial.
    log_offset = float(
        np.median(np.log(result.psd) - np.log(truth)[None, None])
    )
    scaled_draws = result.psd / np.exp(log_offset)
    median = np.median(scaled_draws, axis=(0, 1))
    log_ratio = np.log(median / truth)
    shape_rmse = float(np.sqrt(np.mean(log_ratio**2)))
    assert np.isfinite(shape_rmse)
    assert shape_rmse < 0.6
    result.metadata["truth_log_offset"] = log_offset
    result.metadata["truth_shape_log_rmse"] = shape_rmse

    artifact_dir = outdir / "time-varying" / "ls2-wdm"
    result.save(str(artifact_dir), true_psd=truth * np.exp(log_offset))
    limits = (
        np.log(min(truth.min(), median.min())),
        np.log(max(truth.max(), median.max())),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, surface, title in zip(
        axes[:2],
        (truth, median),
        ("Analytic LS2 PSD", "Posterior median, offset removed"),
        strict=True,
    ):
        image = ax.pcolormesh(
            result.time,
            result.frequency,
            np.log(surface).T,
            vmin=limits[0],
            vmax=limits[1],
            shading="auto",
        )
        ax.set(title=title, xlabel="Rescaled time")
    fig.colorbar(image, ax=axes[:2], label="log PSD")
    residual = axes[2].pcolormesh(
        result.time,
        result.frequency,
        log_ratio.T,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    axes[2].set(
        title=f"log median/truth; RMSE={shape_rmse:.2f}",
        xlabel="Rescaled time",
    )
    axes[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(residual, ax=axes[2], label="log ratio")
    fig.savefig(artifact_dir / "truth_comparison.png", dpi=150)
    plt.close(fig)

    assert (artifact_dir / "inference_data.nc").exists()
    assert (artifact_dir / "posterior_spectrum.png").exists()
    assert (artifact_dir / "truth_comparison.png").exists()
    assert (artifact_dir / "diagnostics" / "spectrum_summary.csv").exists()
    assert (artifact_dir / "diagnostics" / "nuts_summary.csv").exists()


def test_public_moving_periodogram_returns_usable_grid():
    data = moving_periodogram(
        np.random.default_rng(6).standard_normal(128), dt=0.1, m=6, thin=2
    )
    assert data.power.shape == data.counts.shape
    assert data.power.shape == (data.time.size, data.frequency.size)
    assert np.all(data.power >= 0)
    assert np.all(data.counts > 0)
    assert np.all(np.diff(data.time) > 0)
    assert np.all(np.diff(data.frequency) > 0)
