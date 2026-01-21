import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from scipy.interpolate import BSpline
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from log_psplines.datatypes import MultivarFFT, Periodogram, Timeseries
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.plotting import plot_pdgrm
from log_psplines.psplines import LogPSplines
from log_psplines.psplines.initialisation import make_penalty_spd
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
    freq_weights = jnp.ones(spline_model.basis.shape[0])  # model == ones

    # compute LnL at init and optimized weights
    lnl_args = (ln_pdgrm, spline_model.basis, zero_param, freq_weights)
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

    assert trimmed.n_freq < fft.n_freq
    assert np.all(trimmed.freq >= fmin)
    assert np.all(trimmed.freq <= fmax)
    assert trimmed.scaling_factor == pytest.approx(scaling)

    with pytest.raises(ValueError):
        fft.cut(10.0, 5.0)


def test_penalty_cholesky_possible(mock_pdgrm: Periodogram):
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    P = spline_model.penalty_matrix

    # 1st we check min eigenvalue is positive
    eigvals = np.linalg.eigvalsh(P)
    assert eigvals.min() > 0.0, "Penalty matrix is not positive definite."

    # Add small jitter to ensure positive definiteness
    P_jittered = P + 1e-6 * jnp.eye(P.shape[0])
    # Attempt Cholesky decomposition
    L = scipy.linalg.cholesky(P_jittered, lower=True)
    # Reconstruct and compare
    P_reconstructed = L @ L.T
    assert jnp.allclose(P_jittered, P_reconstructed, atol=1e-5)


def test_penalty_spd_vs_raw_plot(mock_pdgrm: Periodogram, outdir):
    out = os.path.join(outdir, "out_penalty_spd_vs_raw")
    os.makedirs(out, exist_ok=True)

    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=6,
        degree=3,
        diffMatrixOrder=2,
    )

    # Rebuild the raw penalty (without SPD fix) for comparison
    bspline_basis = BSplineBasis(
        domain_range=[0, 1],
        order=spline_model.degree + 1,
        knots=spline_model.knots,
    )
    regularization = L2Regularization(
        LinearDifferentialOperator(spline_model.diffMatrixOrder)
    )
    penalty_raw = regularization.penalty_matrix(bspline_basis)

    penalty_spd, penalty_chol, info = make_penalty_spd(
        penalty_raw,
        eps_rel=1e-6,
        eps_abs=0.0,
        do_eig_floor=True,
    )

    # Quick PD sanity check on the SPD matrix
    np.testing.assert_allclose(
        penalty_chol @ penalty_chol.T, penalty_spd, rtol=1e-6, atol=1e-8
    )

    def stats(P: np.ndarray):
        eig = np.linalg.eigvalsh(0.5 * (P + P.T))
        return {
            "max_abs": float(np.max(np.abs(P))),
            "fro": float(np.sqrt(np.sum(P**2))),
            "trace": float(np.trace(P)),
            "eig_min": float(eig.min()),
            "eig_max": float(eig.max()),
            "eig_zero_ct": int(np.sum(np.isclose(eig, 0.0, atol=1e-10))),
        }

    stats_raw = stats(penalty_raw)
    stats_spd = stats(penalty_spd)

    print("penalty_raw stats", stats_raw)
    print("penalty_spd stats", stats_spd)

    # Use independent colorbars for each matrix to see structure
    vmax_raw = np.max(np.abs(penalty_raw))
    vmax_raw = vmax_raw if np.isfinite(vmax_raw) and vmax_raw > 0 else 1.0
    vmax_spd = np.max(np.abs(penalty_spd))
    vmax_spd = vmax_spd if np.isfinite(vmax_spd) and vmax_spd > 0 else 1.0

    print(f"raw_imshow_vrange=({-vmax_raw:.3g}, {vmax_raw:.3g})")
    print(f"spd_imshow_vrange=({-vmax_spd:.3g}, {vmax_spd:.3g})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    im0 = axes[0].imshow(
        penalty_raw, cmap="coolwarm", vmin=-vmax_raw, vmax=vmax_raw
    )
    axes[0].set_title("Raw penalty (no SPD fix)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Raw entry")

    im1 = axes[1].imshow(
        penalty_spd, cmap="coolwarm", vmin=-vmax_spd, vmax=vmax_spd
    )
    axes[1].set_title("SPD penalty (with fix)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="SPD entry")

    fig.suptitle("Penalty matrix comparison (independent scales)", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{out}/penalty_spd_vs_raw.png", bbox_inches="tight")

    tiny = 1e-12
    log_abs_raw = np.log10(np.abs(penalty_raw) + tiny)
    log_abs_spd = np.log10(np.abs(penalty_spd) + tiny)
    vmin_log = float(np.min([log_abs_raw, log_abs_spd]))
    vmax_log = float(np.max([log_abs_raw, log_abs_spd]))

    print(f"log_imshow_vrange=({vmin_log:.3g}, {vmax_log:.3g})")

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axes2[0].imshow(log_abs_raw, cmap="magma", vmin=vmin_log, vmax=vmax_log)
    axes2[0].set_title("log10 |raw|")
    axes2[1].imshow(log_abs_spd, cmap="magma", vmin=vmin_log, vmax=vmax_log)
    axes2[1].set_title("log10 |SPD|")
    fig2.colorbar(
        axes2[1].images[0],
        ax=axes2.ravel().tolist(),
        shrink=0.8,
        label="log10 |P|",
    )
    fig2.suptitle("Penalty matrix log-magnitude", y=1.02)
    plt.tight_layout()
    fig2.savefig(f"{out}/penalty_spd_vs_raw_logabs.png", bbox_inches="tight")
    # Ensure the SPD fixed matrix is positive definite
    eigvals = np.linalg.eigvalsh(penalty_spd)
    assert eigvals.min() > 0.0

    # print both matrices for comparison
    print("Penalty matrix (raw):")
    print(penalty_raw)
    print("Penalty matrix (SPD):")
    print(penalty_spd)
