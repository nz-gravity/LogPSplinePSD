import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import BSpline

from log_psplines.datatypes import MultivarFFT, MultivariateTimeseries
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.psplines import MultivariateLogPSplines
from log_psplines.psplines.initialisation import init_weights


@pytest.fixture
def mock_fft() -> MultivarFFT:
    """Generate synthetic one-channel AR noise data."""
    data = VARMAData.ar(order=2, n_samples=256, fs=256.0, seed=42)
    return data.ts.standardise_for_psd().to_wishart_stats(Nb=1)


def _plot_p1_spline(
    fft: MultivarFFT,
    spline_model: MultivariateLogPSplines,
):
    freq = np.asarray(fft.freq, dtype=np.float64)
    model = np.exp(
        np.asarray(spline_model.diagonal_models[0](), dtype=np.float64)
    )
    spec = PSDMatrixPlotSpec(
        freq=freq,
        ci_dict={
            "psd": {(0, 0): (model, model, model)},
            "coh": {},
            "re": {},
            "im": {},
            "mag": {},
        },
        empirical_psd=fft.empirical_psd,
        save=False,
        close=False,
        show_knots=False,
    )
    return plot_psd_matrix(spec)


def test_spline_init(mock_fft: MultivarFFT, outdir):
    out = os.path.join(outdir, "out_spline_init")
    os.makedirs(out, exist_ok=True)

    # init splines
    t0 = time.time()
    spline_model = MultivariateLogPSplines.from_multivar_fft(
        mock_fft,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    diag_model = spline_model.diagonal_models[0]
    log_psd = np.asarray(diag_model(), dtype=np.float64)
    psd = np.exp(log_psd)
    runtime = float(time.time()) - t0

    print(f"p=1 spline init runtime: {runtime:.2f} seconds")

    # plotting for verification
    fig, axes = _plot_p1_spline(mock_fft, spline_model)
    fig.savefig(f"{out}/test_spline_init.png")
    diag_model.plot_basis(out)

    assert psd.shape == mock_fft.freq.shape
    assert np.all(np.isfinite(psd))
    assert np.all(psd > 0.0)
    assert (
        runtime < 5
    ), f"Initialization should complete in less than 5 seconds, it took {runtime:.2f} seconds."


def test_spline_basis(mock_fft: MultivarFFT, outdir):
    out = os.path.join(outdir, "out_spline_basis")
    os.makedirs(out, exist_ok=True)

    # init splines
    t0 = time.time()
    spline_model = MultivariateLogPSplines.from_multivar_fft(
        mock_fft,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(frac_log=1.0),
    )
    diag_model = spline_model.diagonal_models[0]

    fig, axes = _plot_p1_spline(mock_fft, spline_model)
    ax = axes[0, 0]
    ax2 = ax.twinx()
    for b in diag_model.basis.T:
        ax2.plot(mock_fft.freq, b, alpha=0.5, lw=0.5, marker=".")
    plt.tight_layout()
    fig.savefig(f"{out}/test_spline_basis.png")


def test_closed_form_weight_initialiser_returns_finite_p1_weights(mock_fft):
    spline_model = MultivariateLogPSplines.from_multivar_fft(
        mock_fft,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    diag_model = spline_model.diagonal_models[0]
    empirical = np.real(mock_fft.empirical_psd.psd[:, 0, 0])

    ls_weights = init_weights(
        jnp.log(jnp.asarray(empirical)),
        diag_model,
        num_steps=0,
    )

    assert ls_weights.shape == diag_model.weights.shape
    assert np.all(np.isfinite(np.asarray(ls_weights)))


def test_basis_log_vs_linear(mock_fft: MultivarFFT, outdir):
    outdir = os.path.join(outdir, "out_basis_log_vs_linear")
    os.makedirs(outdir, exist_ok=True)

    def create_bspline_basis(knots, degree, domain, n_points=200):
        """Create B-spline basis functions"""
        # Add boundary knots
        full_knots = np.concatenate(
            [np.repeat(domain[0], degree), knots, np.repeat(domain[1], degree)]
        )

        # Number of basis functions
        n_basis = len(knots) + degree - 1

        # Evaluation points
        x = np.linspace(domain[0], domain[1], n_points)

        # Compute basis matrix
        basis_matrix = np.zeros((len(x), n_basis))

        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spl = BSpline(full_knots, c, degree)
            basis_matrix[:, i] = spl(x)

        return x, basis_matrix

    # Parameters
    degree = 3
    n_knots = 5
    freq_min, freq_max = 1e-5, 1e-1

    # Create knots - linear and log spaced
    knots_linear = np.linspace(freq_min, freq_max, n_knots)
    knots_log = np.logspace(np.log10(freq_min), np.log10(freq_max), n_knots)

    # For basis construction, normalize to [0,1] domain
    knots_linear_norm = (knots_linear - freq_min) / (freq_max - freq_min)
    knots_log_norm = (np.log10(knots_log) - np.log10(freq_min)) / (
        np.log10(freq_max) - np.log10(freq_min)
    )

    # Create basis functions
    x_linear_norm, basis_linear = create_bspline_basis(
        knots_linear_norm, degree, [0, 1], 300
    )
    x_log_norm, basis_log = create_bspline_basis(
        knots_log_norm, degree, [0, 1], 300
    )

    # Convert back to frequency domain
    freq_linear_basis = freq_min + (freq_max - freq_min) * x_linear_norm
    freq_log_basis = 10 ** (
        np.log10(freq_min)
        + x_log_norm * (np.log10(freq_max) - np.log10(freq_min))
    )

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left plot: Linear scale
    for i in range(basis_linear.shape[1]):
        ax1.plot(
            freq_linear_basis,
            basis_linear[:, i],
            "b-",
            alpha=0.7,
            linewidth=1,
            label="Linear knots" if i == 0 else "",
            marker=".",
        )

    for i in range(basis_log.shape[1]):
        ax1.plot(
            freq_log_basis,
            basis_log[:, i],
            "r-",
            alpha=0.7,
            linewidth=1,
            label="Log knots" if i == 0 else "",
            marker=".",
        )

    # Add knots
    ax1.scatter(
        knots_linear,
        np.full(len(knots_linear), -0.05),
        s=80,
        c="blue",
        alpha=0.8,
        marker="|",
        linewidth=3,
        label="Linear Knots",
    )
    ax1.scatter(
        knots_log,
        np.full(len(knots_log), -0.1),
        s=80,
        c="red",
        alpha=0.8,
        marker="|",
        linewidth=3,
        label="Log Knots",
    )

    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Basis Function Value", fontsize=12)
    ax1.set_title(
        "P-spline Basis Functions (Linear Scale)",
        fontweight="bold",
        fontsize=14,
    )
    ax1.grid(False)
    ax1.set_ylim(-0.15, 1.1)

    # Right plot: Log scale
    for i in range(basis_linear.shape[1]):
        ax2.plot(
            freq_linear_basis,
            basis_linear[:, i],
            "b-",
            alpha=0.7,
            linewidth=1,
            label="Linear knots" if i == 0 else "",
            marker=".",
        )

    for i in range(basis_log.shape[1]):
        ax2.plot(
            freq_log_basis,
            basis_log[:, i],
            "r-",
            alpha=0.7,
            linewidth=1,
            label="Log knots" if i == 0 else "",
            marker=".",
        )

    # Add knots
    ax2.scatter(
        knots_linear,
        np.full(len(knots_linear), -0.05),
        s=80,
        c="blue",
        alpha=0.8,
        marker="|",
        linewidth=3,
    )
    ax2.scatter(
        knots_log,
        np.full(len(knots_log), -0.1),
        s=80,
        c="red",
        alpha=0.8,
        marker="|",
        linewidth=3,
    )

    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_title(
        "P-spline Basis Functions (Log Scale)", fontweight="bold", fontsize=14
    )
    ax2.set_xscale("log")
    ax2.grid(False)
    ax2.set_ylim(-0.15, 1.1)
    ax2.legend(loc="upper right", fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/test_basis_log_vs_linear.png")


def test_p1_timeseries_to_wishart_frequency_bounds():
    fs = 64
    t = np.arange(0, 1, 1 / fs)
    y = np.sin(2 * np.pi * 5 * t)
    ts = MultivariateTimeseries(t=t, y=y, scaling_factor=3.0)

    fft = ts.to_wishart_stats(Nb=1, fmin=3.0, fmax=7.0)

    assert len(fft.freq) == 5
    assert np.all(fft.freq >= 3.0)
    assert np.all(fft.freq <= 7.0)
    assert fft.scaling_factor == pytest.approx(3.0)
    assert fft.p == 1

    clipped = ts.to_wishart_stats(Nb=1, fmin=10.0, fmax=5.0)
    assert len(clipped.freq) == 1
    assert clipped.freq[0] == pytest.approx(10.0)


def test_multivar_fft_cut_preserves_scaling():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3))
    scaling = 5.0
    fft = MultivarFFT.compute_fft(x, fs=32.0, scaling_factor=scaling)

    # Skip the first available frequency to ensure truncation happens
    fmin = float(fft.freq[1])
    fmax = float(fft.freq[-2])
    trimmed = fft.cut(fmin, fmax)

    assert trimmed.N < fft.N
    assert np.all(trimmed.freq >= fmin)
    assert np.all(trimmed.freq <= fmax)
    assert trimmed.scaling_factor == pytest.approx(scaling)

    with pytest.raises(ValueError):
        fft.cut(10.0, 5.0)
