import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from scipy.interpolate import BSpline

from log_psplines.datatypes import MultivarFFT, Periodogram, Timeseries
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.plotting import plot_pdgrm
from log_psplines.psplines import LogPSplines
from log_psplines.samplers.univar.univar_base import log_likelihood


@pytest.fixture
def mock_pdgrm() -> Periodogram:
    """Generate synthetic AR noise data."""
    return ARData(order=4, duration=1.0, fs=256, seed=42).periodogram


def test_spline_init(mock_pdgrm: Periodogram, outdir):
    out = os.path.join(outdir, "out_spline_init")
    os.makedirs(out, exist_ok=True)

    # init splines
    t0 = time.time()
    ln_pdgrm = jnp.log(mock_pdgrm.power)
    zero_param = jnp.zeros(ln_pdgrm.shape[0])
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    zero_weights = jnp.zeros(spline_model.weights.shape)  # model == zeros
    optim_weights = spline_model.weights
    Nh = 1  # model == ones

    # compute LnL at init and optimized weights
    lnl_args = (ln_pdgrm, spline_model.basis, zero_param, Nh)
    lnl_initial = log_likelihood(zero_weights, *lnl_args)
    lnl_final = log_likelihood(optim_weights, *lnl_args)
    runtime = float(time.time()) - t0

    print(
        f"LnL initial: {lnl_initial:.2f}, LnL final: {lnl_final:.2f}, runtime: {runtime:.2f} seconds"
    )

    # plotting for verification
    fig, ax = plot_pdgrm(mock_pdgrm, spline_model)
    fig.savefig(f"{out}/test_spline_init.png")
    spline_model.plot_basis(out)

    assert (
        lnl_final > lnl_initial
    ), "Optimized weights should yield a higher log-likelihood than initial zeros."
    assert (
        runtime < 5
    ), "Initialization should complete in less than 5 seconds."


def test_spline_basis(mock_pdgrm: Periodogram, outdir):
    out = os.path.join(outdir, "out_spline_basis")
    os.makedirs(out, exist_ok=True)

    # init splines
    t0 = time.time()
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(frac_log=1.0),
    )

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model)
    ax2 = ax.twinx()
    for b in spline_model.basis.T:
        ax2.plot(mock_pdgrm.freqs, b, alpha=0.5, lw=0.5, marker=".")
    plt.tight_layout()
    fig.savefig(f"{out}/test_spline_basis.png")


def test_basis_log_vs_linear(mock_pdgrm: Periodogram, outdir):
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


def test_timeseries_to_periodogram_frequency_bounds():
    fs = 64
    t = np.arange(0, 1, 1 / fs)
    y = np.sin(2 * np.pi * 5 * t)
    ts = Timeseries(t=t, y=y, scaling_factor=3.0)

    pdgrm = ts.to_periodogram(fmin=3.0, fmax=7.0)

    assert len(pdgrm.freqs) == 5
    assert np.all(pdgrm.freqs >= 3.0)
    assert np.all(pdgrm.freqs <= 7.0)
    assert pdgrm.scaling_factor == pytest.approx(3.0)

    clipped = ts.to_periodogram(fmin=10.0, fmax=5.0)
    assert len(clipped.freqs) == 1
    assert clipped.freqs[0] == pytest.approx(5.0)


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
