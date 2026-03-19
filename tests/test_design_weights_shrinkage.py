"""Tests for soft shrinkage toward design weights in the multivariate sampler.

Covers:
- compute_design_weights round-trip (Cholesky → weights → reconstructed PSD)
- sample_pspline_block w_design / tau log-prior shift
- Full run_mcmc end-to-end with design_psd and tau kwargs (smoke test)
- Backward compatibility: no design → identical prior to original
"""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pytest
from numpyro.infer.util import log_density

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.samplers.pspline_block import sample_pspline_block

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

# Small VAR(2) parameters from the 3-channel simulation study
_A1 = np.diag([0.4, 0.3, 0.2])
_A2 = np.array(
    [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]], dtype=np.float64
)
_SIGMA = np.array(
    [[0.25, 0.0, 0.08], [0.0, 0.25, 0.08], [0.08, 0.08, 0.25]],
    dtype=np.float64,
)
_VAR_COEFFS = np.array([_A1, _A2], dtype=np.float64)
_FS = 1.0
_P = 3


def _simulate_var3(n: int, seed: int = 0) -> np.ndarray:
    """Return (n, 3) VAR(2) samples."""
    rng = np.random.default_rng(seed)
    burn = 128
    noise = rng.multivariate_normal(np.zeros(_P), _SIGMA, size=n + burn)
    x = np.zeros((n + burn, _P))
    for t in range(2, n + burn):
        x[t] = noise[t] + _A1 @ x[t - 1] + _A2 @ x[t - 2]
    return x[burn:]


def _true_psd(freqs: np.ndarray) -> np.ndarray:
    """Theoretical VAR(2) PSD matrix, shape (len(freqs), 3, 3)."""
    omega = 2 * np.pi * freqs / _FS
    I = np.eye(_P, dtype=np.complex128)
    out = np.empty((len(freqs), _P, _P), dtype=np.complex128)
    for i, w in enumerate(omega):
        A = I - _A1 * np.exp(-1j * w) - _A2 * np.exp(-2j * w)
        H = np.linalg.inv(A)
        out[i] = (2.0 / _FS) * (H @ _SIGMA @ H.conj().T)
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


# ---------------------------------------------------------------------------
# Unit: compute_design_weights round-trip
# ---------------------------------------------------------------------------


def _build_spline_model(n: int = 64, n_knots: int = 5):
    """Build a MultivariateLogPSplines from small synthetic 3-channel data."""
    from log_psplines.datatypes.multivar import MultivariateTimeseries
    from log_psplines.preprocessing.preprocessing import _preprocess_data

    rng = np.random.default_rng(7)
    t = np.arange(n) / _FS
    y = _simulate_var3(n, seed=7)
    ts = MultivariateTimeseries(t=t, y=y)
    preproc = _preprocess_data(ts, None, n_knots=n_knots, Nb=2)
    from log_psplines.preprocessing.sampler_factory import (
        _build_model_from_data,
    )

    return (
        _build_model_from_data(
            preproc.processed_data, preproc.run_config.model
        ),
        preproc.processed_data,
    )


def test_compute_design_weights_returns_correct_keys():
    model, fft_data = _build_spline_model(n=64, n_knots=5)
    N, p = model.N, model.p
    assert p == _P

    freqs = fft_data.freq  # model frequencies
    design_psd = _true_psd(freqs)

    dw = model.compute_design_weights(design_psd)

    # Should have one key per diagonal channel
    for j in range(p):
        assert f"delta_{j}" in dw, f"Missing 'delta_{j}'"
        assert dw[f"delta_{j}"].shape == (model.n_basis,)

    # Should have theta keys for all lower-triangular pairs
    for j in range(1, p):
        for l in range(j):
            assert f"theta_re_{j}_{l}" in dw
            assert f"theta_im_{j}_{l}" in dw
            assert dw[f"theta_re_{j}_{l}"].shape == (
                model.offdiag_re_model.n_basis,
            )


def test_compute_design_weights_reconstructs_diagonal():
    """Spline weights fit to log(delta_j) should reproduce the log-diagonal well."""
    model, fft_data = _build_spline_model(n=128, n_knots=6)
    freqs = fft_data.freq
    design_psd = _true_psd(freqs)

    dw = model.compute_design_weights(design_psd)
    L = np.linalg.cholesky(design_psd)

    for j in range(_P):
        log_delta_sq_true = 2.0 * np.log(np.abs(L[:, j, j]))
        basis = np.asarray(model.diagonal_models[j].basis)
        log_delta_sq_fit = basis @ np.asarray(dw[f"delta_{j}"])
        # Spline fit should be close on the training grid (not exact, it's a
        # least-squares fit) — check mean absolute error is small
        mae = float(np.mean(np.abs(log_delta_sq_true - log_delta_sq_fit)))
        assert mae < 1.0, f"MAE for delta_{j} = {mae:.3f} exceeds tolerance"


def test_compute_design_weights_bad_shape_raises():
    model, _ = _build_spline_model(n=64, n_knots=5)
    bad = np.eye(2, dtype=np.complex128)[np.newaxis].repeat(model.N, axis=0)
    with pytest.raises(ValueError, match="shape"):
        model.compute_design_weights(bad)


# ---------------------------------------------------------------------------
# Unit: sample_pspline_block prior with w_design / tau
# ---------------------------------------------------------------------------


def _make_penalty(k: int) -> jnp.ndarray:
    """Simple second-difference penalty matrix of size k."""
    D = np.diff(np.eye(k), n=2, axis=0)
    P = D.T @ D + 1e-6 * np.eye(k)
    return jnp.asarray(P)


def _eval_prior(weights_val, penalty, *, w_design=None, tau=None):
    """Return the log_prior_adjustment factor for given weight values."""
    k = penalty.shape[0]
    result = {}

    def model():
        out = sample_pspline_block(
            delta_name="delta",
            phi_name="log_phi",
            weights_name="w",
            penalty_matrix=penalty,
            alpha_phi=2.0,
            beta_phi=1.0,
            alpha_delta=1.0,
            beta_delta=1.0,
            w_design=w_design,
            tau=tau,
        )
        result["weights"] = out["weights"]

    params = {
        "delta": jnp.asarray(1.0),
        "log_phi": jnp.asarray(0.0),
        "w": jnp.asarray(weights_val),
    }
    lp, _ = log_density(model, (), {}, params)
    return float(lp)


def test_sample_pspline_block_no_design_unchanged():
    """w_design=None should give the same log-prob as the original code path."""
    k = 8
    penalty = _make_penalty(k)
    w = np.ones(k) * 0.3

    lp_no_design = _eval_prior(w, penalty, w_design=None, tau=None)
    lp_zero_design = _eval_prior(w, penalty, w_design=jnp.zeros(k), tau=None)

    # w_design=zeros and w_design=None should give the same result (both
    # compute residual=w, same wPw)
    assert abs(lp_no_design - lp_zero_design) < 1e-5


def test_sample_pspline_block_design_shifts_mode():
    """Prior with w_design should be higher at w==w_design than at w==0."""
    k = 8
    penalty = _make_penalty(k)
    w_design = jnp.ones(k) * 2.0

    lp_at_design = _eval_prior(
        np.asarray(w_design), penalty, w_design=w_design, tau=None
    )
    lp_at_zero = _eval_prior(np.zeros(k), penalty, w_design=w_design, tau=None)

    # Mode of the smoothness prior is at residual=0, i.e. w=w_design
    assert lp_at_design > lp_at_zero


def test_sample_pspline_block_tau_shrinks_level():
    """Adding tau should lower log-prob for w far from w_design."""
    k = 8
    penalty = _make_penalty(k)
    w_design = jnp.zeros(k)
    w_far = np.ones(k) * 5.0

    lp_no_tau = _eval_prior(w_far, penalty, w_design=w_design, tau=None)
    lp_with_tau = _eval_prior(w_far, penalty, w_design=w_design, tau=0.5)

    # tau adds a negative L2 term → log-prob at w_far should decrease
    assert lp_with_tau < lp_no_tau


def test_sample_pspline_block_tau_no_design_ignored():
    """tau should have no effect when w_design is None (backward compat)."""
    k = 8
    penalty = _make_penalty(k)
    w = np.ones(k) * 3.0

    lp_no_tau = _eval_prior(w, penalty, w_design=None, tau=None)
    lp_tau_no_design = _eval_prior(w, penalty, w_design=None, tau=0.1)

    assert abs(lp_no_tau - lp_tau_no_design) < 1e-6


# ---------------------------------------------------------------------------
# Comparison: 3D VAR with and without design shrinkage
# ---------------------------------------------------------------------------

_SHARED_MCMC_KWARGS = dict(
    n_knots=10,
    n_samples=500,
    n_warmup=500,
    num_chains=1,
    Nb=2,
    init_from_vi=False,
    verbose=False,
    compute_coherence_quantiles=True,
)


def _run_var3(n: int, seed: int, **extra_kwargs):
    """Helper: simulate 3-ch VAR, compute true PSD, run MCMC, return (idata, metrics)."""
    y = _simulate_var3(n, seed=seed)
    t = np.arange(n) / _FS
    ts = MultivariateTimeseries(t=t, y=y)
    freqs_hz = np.fft.rfftfreq(n, d=1.0 / _FS)[1:]
    true_psd = _true_psd(freqs_hz)

    idata = run_mcmc(
        ts,
        true_psd=(freqs_hz, true_psd),
        **_SHARED_MCMC_KWARGS,
        **extra_kwargs,
    )
    return idata, freqs_hz, true_psd


def _extract_metrics(idata) -> dict:
    """Pull key scalars from idata.attrs and posterior_psd group."""
    attrs = idata.attrs
    m = {
        "riae_matrix": float(attrs.get("riae_matrix", float("nan"))),
        "coverage": float(attrs.get("coverage", float("nan"))),
    }
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is not None and "psd_matrix_real" in psd_group:
        psd_real = np.asarray(psd_group["psd_matrix_real"].values)
        percentiles = np.asarray(
            psd_group["psd_matrix_real"].coords.get("percentile", []),
            dtype=float,
        )
        if psd_real.shape[0] >= 2:
            idx05 = int(np.argmin(np.abs(percentiles - 5.0)))
            idx95 = int(np.argmin(np.abs(percentiles - 95.0)))
            width = np.maximum(psd_real[idx95] - psd_real[idx05], 0.0)
            p = width.shape[1]
            diag_mask = np.eye(p, dtype=bool)
            offdiag_mask = ~diag_mask
            m["ciw_diag_mean"] = float(np.mean(width[:, diag_mask]))
            m["ciw_offdiag_mean"] = float(np.mean(width[:, offdiag_mask]))
    return m


def _print_comparison(m_base: dict, m_design: dict) -> None:
    keys = sorted(set(m_base) | set(m_design))
    header = (
        f"{'metric':<22} {'no design':>12} {'with design':>12} {'delta':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for k in keys:
        v0 = m_base.get(k, float("nan"))
        v1 = m_design.get(k, float("nan"))
        delta = v1 - v0
        print(f"{k:<22} {v0:>12.4f} {v1:>12.4f} {delta:>+10.4f}")
    print("=" * len(header))


@pytest.mark.slow
def test_var3_design_vs_no_design_comparison(outdir):
    """Run 3-ch VAR with and without design shrinkage and compare metrics.

    With design_psd = true PSD and moderate tau, we expect:
    - riae_matrix: similar or slightly lower (design pulls median toward truth)
    - coverage: possibly improved for off-diagonal elements
    - CI widths: narrower (stronger prior concentrates posterior)
    """

    outdir = f"{outdir}/out_shrinkage_comparison"
    os.makedirs(outdir, exist_ok=True)
    n = 512
    seed = 0

    print("\n--- Running WITHOUT design shrinkage ---")
    idata_base, freqs_hz, true_psd = _run_var3(n, seed)

    print("\n--- Running WITH design shrinkage (tau=1.0, design=true PSD) ---")
    idata_design, _, _ = _run_var3(
        n,
        seed,
        design_psd=(freqs_hz, true_psd),
        tau=1.0,
    )

    m_base = _extract_metrics(idata_base)
    m_design = _extract_metrics(idata_design)
    _print_comparison(m_base, m_design)

    # Both runs must produce valid posterior draws
    expected_draws = _SHARED_MCMC_KWARGS["n_samples"]
    assert idata_base.posterior.sizes["draw"] == expected_draws
    assert idata_design.posterior.sizes["draw"] == expected_draws

    # RIAE should be finite and reasonable in both cases
    assert np.isfinite(m_base["riae_matrix"])
    assert np.isfinite(m_design["riae_matrix"])
    assert m_base["riae_matrix"] < 1.0
    assert m_design["riae_matrix"] < 1.0

    # Design shrinkage should produce narrower CIs (stronger prior)
    if "ciw_diag_mean" in m_base and "ciw_diag_mean" in m_design:
        print(
            f"\nDiag CI width:    base={m_base['ciw_diag_mean']:.4f}  "
            f"design={m_design['ciw_diag_mean']:.4f}"
        )
        print(
            f"Offdiag CI width: base={m_base['ciw_offdiag_mean']:.4f}  "
            f"design={m_design['ciw_offdiag_mean']:.4f}"
        )
        assert (
            m_design["ciw_diag_mean"] <= m_base["ciw_diag_mean"] * 1.1
        ), "Expected design run to have comparable or narrower diagonal CIs"

    # ----- Overlay plot: base vs design on the same axes -----
    # Plot on the same frequency grid used by idata quantiles; otherwise
    # plot_psd_matrix drops true_psd when frequency lengths differ.
    plot_freq = np.asarray(idata_base.attrs.get("frequencies", freqs_hz))
    true_psd_plot = _true_psd(plot_freq)

    # First pass: no-design run (creates the figure/axes grid)
    spec_base = PSDMatrixPlotSpec(
        idata=idata_base,
        true_psd=true_psd_plot,
        freq=plot_freq,
        outdir=None,
        save=False,
        close=False,
        label="No design",
        model_color="tab:blue",
        show_knots=False,
    )
    fig, ax = plot_psd_matrix(spec_base)

    # Second pass: design run overlaid on the same axes
    spec_design = PSDMatrixPlotSpec(
        idata=idata_design,
        fig=fig,
        ax=ax,
        outdir=None,
        save=False,
        close=False,
        label=f"Design (tau=1.0)",
        model_color="tab:orange",
        show_knots=False,
        true_psd=true_psd_plot,  # show same true PSD for both
        freq=plot_freq,
    )
    plot_psd_matrix(spec_design)

    # Add a shared legend and title to the first axis
    ax.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"3-ch VAR: no design vs design=true PSD (tau=1.0)\n"
        f"RIAE: {m_base['riae_matrix']:.3f} → {m_design['riae_matrix']:.3f}  |  "
        f"Coverage: {m_base['coverage']:.2f} → {m_design['coverage']:.2f}",
        fontsize=9,
    )
    fig.savefig(
        f"{outdir}/design_vs_no_design_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\nPlot saved to {outdir}/design_vs_no_design_comparison.png")
