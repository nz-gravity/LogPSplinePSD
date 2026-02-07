import os
from pathlib import Path

import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainSpec,
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)
from log_psplines.datatypes import Periodogram
from log_psplines.datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    _get_coherence,
)
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.plotting.pdgrm import plot_pdgrm
from log_psplines.plotting.psd_matrix import _pack_ci_dict, plot_psd_matrix
from log_psplines.psplines.psplines import LogPSplines


def _ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _plot_block_likelihood_components(
    freqs: np.ndarray,
    u_complex: np.ndarray,
    outdir: str,
    tag: str = "coarse",
) -> None:
    """Visualize the u_j blocks and least-squares theta fits for a given FFT grid."""
    import matplotlib.pyplot as plt

    N, p, n_reps = u_complex.shape
    for j in range(p):
        u_j = u_complex[:, j, :]  # (freq, rep)
        delta_raw = np.sqrt(np.mean(np.abs(u_j) ** 2, axis=-1))
        theta_hat = None
        delta_resid = None

        if j > 0:
            theta_hat = np.zeros((N, j), dtype=np.complex128)
            delta_resid = np.zeros(N, dtype=np.float64)
            for k in range(N):
                design = np.transpose(u_complex[k, :j, :])  # (rep, j)
                target = u_j[k]
                if not np.any(design):
                    theta_hat[k] = 0.0
                    delta_resid[k] = np.sqrt(np.mean(np.abs(target) ** 2))
                    continue
                try:
                    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
                except np.linalg.LinAlgError:
                    sol = np.zeros(j, dtype=np.complex128)
                theta_hat[k] = sol
                resid = target - design @ sol
                delta_resid[k] = np.sqrt(np.mean(np.abs(resid) ** 2))

        n_rows = 1 if j == 0 else 3
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(7.0, 3.0 * n_rows), sharex=True
        )
        axes = np.atleast_1d(axes)

        axes[0].loglog(freqs, delta_raw, label=r"$||u_j||/\sqrt{p}$")
        if delta_resid is not None:
            axes[0].loglog(
                freqs,
                delta_resid,
                label=r"$||u_j - U_{<j}\hat{\theta}_j||/\sqrt{p}$",
            )
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"Channel {j + 1}: block amplitude proxies")
        axes[0].grid(True, which="both", ls=":", alpha=0.4)
        axes[0].legend(fontsize=8)

        if theta_hat is not None:
            for idx in range(j):
                axes[1].plot(
                    freqs,
                    theta_hat[:, idx].real,
                    label=rf"$\Re(\theta_{{{j + 1},{idx + 1}}})$",
                )
            axes[1].set_ylabel("Re(theta)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8, ncol=2)

            for idx in range(j):
                axes[2].plot(
                    freqs,
                    theta_hat[:, idx].imag,
                    label=rf"$\Im(\theta_{{{j + 1},{idx + 1}}})$",
                )
            axes[2].set_ylabel("Im(theta)")
            axes[2].set_xlabel("Frequency [Hz]")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=8, ncol=2)
        else:
            axes[0].set_xlabel("Frequency [Hz]")

        fig.tight_layout()
        fname = os.path.join(
            outdir, f"{tag}_blocked_components_channel_{j + 1}.png"
        )
        fig.savefig(fname, dpi=150)
        plt.close(fig)


@pytest.mark.parametrize("seed", [0])
def test_plot_univariate_fitted_data_blocks(outdir: str, seed: int, test_mode):
    """
    Plot what the univariate P-spline actually fits:
    - Original vs coarse-grained periodogram on the coarse frequency grid
    - Coarse-grain frequency weights used in the likelihood
    - Data + initial P-spline (showing knot locations)
    """

    # Output paths
    outdir = _ensure_dir(os.path.join(outdir, "out_plot_blocks", "univar"))

    # Simulated AR data (small for test speed)
    n_seconds = 4.0 if test_mode != "fast" else 2.0
    fs = 2048.0 if test_mode != "fast" else 256.0
    ar = ARData(order=2, duration=n_seconds, fs=fs, sigma=1.0, seed=seed)
    pdgrm_full = ar.periodogram

    # Coarse-graining specification
    N = pdgrm_full.freqs.size
    freqs = pdgrm_full.freqs
    Nc = N // 2 if test_mode != "fast" else N // 4
    spec: CoarseGrainSpec = compute_binning_structure(
        freqs,
        Nc=Nc,
        f_min=None,
        f_max=None,
    )

    # # 1) Original vs coarse-grained data (what we actually fit)
    # fig, ax, weights = plot_coarse_vs_original(
    #     freqs=pdgrm_full.freqs,
    #     power=pdgrm_full.power,
    #     spec=spec,
    #     scaling_factor=pdgrm_full.scaling_factor,
    # )
    # fig.savefig(os.path.join(outdir, "coarse_vs_original.png"), dpi=150)

    # 2) Coarse-bin Nh used in the likelihood scaling
    # fig_w, ax_w = plot_coarse_grain_weights(spec=spec, Nh=spec.Nh)
    # fig_w.savefig(os.path.join(outdir, "coarse_weights.png"), dpi=150)

    # 3) Build a P-spline on the coarse grid and plot with knots
    power_coarse, Nh = apply_coarse_graining_univar(
        pdgrm_full.power[spec.selection_mask], spec, freqs[spec.selection_mask]
    )
    pdgrm_coarse = Periodogram(
        freqs=spec.f_coarse,
        power=power_coarse,
        scaling_factor=pdgrm_full.scaling_factor,
        Nh=Nh,
    )
    n_knots = 10 if test_mode != "fast" else 6
    model = LogPSplines.from_periodogram(
        pdgrm_coarse, n_knots=n_knots, degree=3, diffMatrixOrder=2
    )

    fig3, ax3 = plot_pdgrm(
        pdgrm=pdgrm_coarse,
        spline_model=model,
        weights=None,
        show_knots=True,
        show_parametric=False,
        ax=None,
    )
    fig3.savefig(
        os.path.join(outdir, "coarse_pspline_with_knots.png"), dpi=150
    )
    fig3.clear()

    # Basic existence checks
    for fname in [
        # "coarse_vs_original.png",
        # "coarse_weights.png",
        "coarse_pspline_with_knots.png",
    ]:
        path = os.path.join(outdir, fname)
        assert os.path.exists(path) and os.path.getsize(path) > 0


@pytest.mark.parametrize("seed", [1])
def test_plot_multivariate_fitted_data_blocks(
    outdir: str, seed: int, test_mode
):
    """
    Plot the multivariate data the P-splines see after Wishart/time blocking
    and optional frequency coarse-graining.
    Produces a PSD matrix plot driven purely by the empirical/coarse data.
    """

    outdir = _ensure_dir(os.path.join(outdir, "out_plot_blocks", "multivar"))

    # Simulate small VARMA dataset
    n = 2048 if test_mode != "fast" else 256
    varma = VARMAData(n_samples=n, seed=seed)

    # Build block-averaged (Wishart) FFT statistics with a few time blocks
    Nb = 2 if test_mode != "fast" else 1
    x = varma.data  # (n, p)
    fft_full: MultivarFFT = MultivarFFT.compute_wishart(x, fs=varma.fs, Nb=Nb)

    N = fft_full.freq.size
    Nc = N // 2 if test_mode != "fast" else N // 4
    # Coarse-grain along frequency for what the likelihood actually fits
    spec = compute_binning_structure(
        fft_full.freq,
        Nc=Nc,
        f_min=None,
        f_max=None,
    )
    # Returns a MultivarFFT on coarse grid and the frequency weights
    # fft_coarse.raw_psd holds the aggregated PSD matrix we fit
    # (n_freq_coarse, p, p)
    fft_coarse, weights = apply_coarse_grain_multivar_fft(fft_full, spec)

    # Prepare empirical PSD wrapper for plotting overlays (coarse grid)
    psd_matrix = np.asarray(fft_coarse.raw_psd)
    coherence = _get_coherence(psd_matrix)
    empirical = EmpiricalPSD(
        freq=np.asarray(fft_coarse.freq),
        psd=psd_matrix,
        coherence=coherence,
    )

    # Build a trivial CI dictionary using empirical data as q05=q50=q95
    ci_dict = _pack_ci_dict(
        psd_samples=psd_matrix[None, ...], show_coherence=True
    )

    fig, ax = plot_psd_matrix(
        ci_dict=ci_dict,
        freq=np.asarray(fft_coarse.freq),
        empirical_psd=empirical,
        outdir=outdir,
        filename="psd_matrix_empirical.png",
        dpi=150,
        show_coherence=True,
        save=True,
    )

    # Full-resolution PSD/CSD matrix for reference
    psd_matrix_full = np.asarray(fft_full.raw_psd)
    coherence_full = _get_coherence(psd_matrix_full)
    empirical_full = EmpiricalPSD(
        freq=np.asarray(fft_full.freq),
        psd=psd_matrix_full,
        coherence=coherence_full,
    )
    ci_dict_full = _pack_ci_dict(
        psd_samples=psd_matrix_full[None, ...], show_coherence=True
    )
    plot_psd_matrix(
        ci_dict=ci_dict_full,
        freq=np.asarray(fft_full.freq),
        empirical_psd=empirical_full,
        outdir=outdir,
        filename="psd_matrix_empirical_full.png",
        dpi=150,
        show_coherence=True,
        save=True,
    )

    # Plot |CSD_ij| magnitudes for lower triangle
    ci_dict_abs = _pack_ci_dict(
        psd_samples=psd_matrix_full[None, ...],
        show_coherence=False,
        show_csd_magnitude=True,
    )
    plot_psd_matrix(
        ci_dict=ci_dict_abs,
        freq=np.asarray(fft_full.freq),
        empirical_psd=empirical_full,
        outdir=outdir,
        filename="psd_matrix_abs_csd.png",
        dpi=150,
        show_coherence=False,
        show_csd_magnitude=True,
        save=True,
    )

    # Also save frequency weights for reference
    import matplotlib.pyplot as plt

    figw, axw = plt.subplots()
    axw.semilogx(fft_coarse.freq, weights, "-", color="C2", lw=1.8)
    axw.set_xlabel("Frequency [Hz]")
    axw.set_ylabel("Frequency weight")
    axw.set_title("Coarse-grain weights (multivariate)")
    figw.tight_layout()
    figw.savefig(os.path.join(outdir, "coarse_weights_multivar.png"), dpi=150)
    plt.close(figw)

    # Plot block-wise likelihood ingredients (u_j, best-fit theta)
    components_dir = _ensure_dir(os.path.join(outdir, "blocked_components"))
    u_complex = np.asarray(fft_coarse.u_re) + 1j * np.asarray(fft_coarse.u_im)
    _plot_block_likelihood_components(
        freqs=np.asarray(fft_coarse.freq),
        u_complex=u_complex,
        outdir=components_dir,
        tag="coarse",
    )

    # Also plot the full-resolution (pre–coarse-grained) versions for comparison
    u_complex_full = np.asarray(fft_full.u_re) + 1j * np.asarray(fft_full.u_im)
    _plot_block_likelihood_components(
        freqs=np.asarray(fft_full.freq),
        u_complex=u_complex_full,
        outdir=components_dir,
        tag="full",
    )

    # Existence checks
    expected_files = [
        "psd_matrix_empirical.png",
        "psd_matrix_empirical_full.png",
        "psd_matrix_abs_csd.png",
        "coarse_weights_multivar.png",
    ]
    for tag in ("coarse", "full"):
        expected_files.extend(
            [
                os.path.join(
                    "blocked_components",
                    f"{tag}_blocked_components_channel_{idx + 1}.png",
                )
                for idx in range(fft_coarse.p)
            ]
        )
    for fname in expected_files:
        path = os.path.join(outdir, fname)
        assert os.path.exists(path) and os.path.getsize(path) > 0

    print(
        f"Tested plotting fitted data blocks with seed={seed} and test_mode={test_mode}."
    )
