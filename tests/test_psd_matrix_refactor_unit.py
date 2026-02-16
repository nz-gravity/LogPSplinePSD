import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.example_datasets.lisa_data import LISAData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.plotting.psd_matrix import (
    PSDMatrixPlotSpec,
    _pack_ci_dict,
    plot_psd_matrix,
)

OUTDIR = "out_PSD_matrix"


def _build_ci(p: int = 2, n: int = 32):
    """Build confidence interval dict for testing."""
    freq = np.linspace(0.1, 1.0, n)
    psd_samples = np.ones((3, n, p, p), dtype=np.complex128)
    for i in range(p):
        for j in range(p):
            scale = 1.0 + 0.1 * (i + j)
            psd_samples[:, :, i, j] *= scale
    return freq, _pack_ci_dict(psd_samples=psd_samples, show_coherence=True)


def test_plot_psd_matrix_accepts_spec_object():
    freq, ci = _build_ci()
    spec = PSDMatrixPlotSpec(ci_dict=ci, freq=freq, save=False, close=False)
    fig, axes = plot_psd_matrix(spec)
    assert axes.shape == (2, 2)
    fig.clf()


def test_plot_psd_matrix_overlays_extra_empirical_labels():
    freq, ci = _build_ci()
    psd_emp = np.ones((freq.size, 2, 2), dtype=np.complex128)
    empirical = EmpiricalPSD(
        freq=freq,
        psd=psd_emp,
        coherence=_get_coherence(psd_emp),
        channels=np.arange(2),
    )
    spec = PSDMatrixPlotSpec(
        ci_dict=ci,
        freq=freq,
        empirical_psd=empirical,
        extra_empirical_psd=[empirical],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -20}],
        save=False,
        close=False,
    )
    fig, axes = plot_psd_matrix(spec)
    labels = [line.get_label() for line in axes[0, 0].lines]
    assert "Welch" in labels
    fig.clf()


def test_plot_psd_matrix_conflicting_modes_raise():
    freq, ci = _build_ci()
    spec = PSDMatrixPlotSpec(
        ci_dict=ci,
        freq=freq,
        show_coherence=True,
        show_csd_magnitude=True,
        save=False,
    )
    with pytest.raises(ValueError):
        plot_psd_matrix(spec)


def test_plot_psd_matrix_validates_axes_shape():
    """Test that providing mismatched axes shape raises error."""
    freq, ci = _build_ci()
    fig, ax = plt.subplots(1, 2)

    with pytest.raises(ValueError):
        spec = PSDMatrixPlotSpec(
            ci_dict=ci,
            freq=freq,
            fig=fig,
            ax=ax,
            save=False,
        )
        plot_psd_matrix(spec)

    plt.close(fig)


def test_plot_psd_matrix_scales_and_limits(outdir):
    """Test plotting with coherence and frequency range settings."""
    out = f"{outdir}/{OUTDIR}"
    os.makedirs(out, exist_ok=True)

    freq, ci_dict = _build_ci(p=3, n=1000)
    freq_range = (0.2, 0.9)

    # Test with coherence
    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict,
        freq=freq,
        freq_range=freq_range,
        show_coherence=True,
        outdir=out,
        filename="with_coh.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(spec)

    diag_ax = axes[0, 0]
    coh_ax = axes[1, 0]
    upper_ax = axes[0, 1]

    assert diag_ax.get_yscale() == "log"
    assert coh_ax.get_yscale() == "linear"
    assert not upper_ax.axison
    assert coh_ax.get_ylim()[1] == pytest.approx(1.0)
    assert diag_ax.get_xlim() == pytest.approx(freq_range)

    plt.close(fig)

    # Test without coherence (real/imaginary parts)
    ci_dict_reim = _pack_ci_dict(
        psd_samples=np.ones((2, 1000, 3, 3), dtype=float), show_coherence=False
    )
    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_reim,
        freq=freq,
        freq_range=freq_range,
        show_coherence=False,
        outdir=out,
        filename="without_coh.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(spec)

    diag_ax = axes[0, 0]
    re_ax = axes[1, 0]
    im_ax = axes[0, 1]

    assert diag_ax.get_yscale() == "log"
    assert re_ax.get_yscale() == "linear"
    assert im_ax.get_yscale() == "linear"
    assert im_ax.axison  # upper triangle now used for imag parts
    assert diag_ax.get_xlim() == pytest.approx(freq_range)

    plt.close(fig)


def test_plot_psd_matrix_extra_empirical_zorder():
    """Test that extra empirical PSD overlays respect zorder."""
    freq, ci = _build_ci(p=2, n=20)
    N = freq.size
    psd_emp = np.ones((N, 2, 2), dtype=np.complex128)
    psd_welch = 2.0 * np.ones((N, 2, 2), dtype=np.complex128)

    empirical = EmpiricalPSD(
        freq=freq,
        psd=psd_emp,
        coherence=_get_coherence(psd_emp),
        channels=np.arange(2),
    )
    welch = EmpiricalPSD(
        freq=freq,
        psd=psd_welch,
        coherence=_get_coherence(psd_welch),
        channels=np.arange(2),
    )

    spec = PSDMatrixPlotSpec(
        ci_dict=ci,
        freq=freq,
        empirical_psd=empirical,
        extra_empirical_psd=[welch],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -10}],
        save=False,
        close=False,
    )
    fig, axes = plot_psd_matrix(spec)

    z_by_label = {
        line.get_label(): line.get_zorder() for line in axes[0, 0].lines
    }
    assert z_by_label["Welch"] < z_by_label["Empirical"]
    plt.close(fig)


def test_plot_psd_matrix_with_varma_coherence(outdir):
    """Test PSD matrix plotting with VARMA data and coherence."""
    out = f"{outdir}/{OUTDIR}"
    os.makedirs(out, exist_ok=True)

    var_coeffs = np.array(
        [
            [[0.35, 0.05, 0.02], [0.0, 0.25, 0.04], [0.03, 0.02, 0.30]],
            [[0.08, 0.0, 0.0], [0.0, 0.07, 0.01], [0.0, 0.0, 0.06]],
            [[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.02]],
        ]
    )
    vma_coeffs = np.eye(3)[None, ...]
    sigma = np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]], dtype=float
    )
    data = VARMAData(
        n_samples=512,
        sigma=sigma,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        seed=42,
        fs=10.0,
    )

    periodogram = data.get_periodogram()
    true_psd = data.get_true_psd()
    freq = data.freq
    ci_dict_coh = _pack_ci_dict(
        psd_samples=periodogram[None, ...], show_coherence=True
    )
    ci_dict_reim = _pack_ci_dict(
        psd_samples=periodogram[None, ...], show_coherence=False
    )
    ci_dict_mag = _pack_ci_dict(
        psd_samples=periodogram[None, ...],
        show_coherence=False,
        show_csd_magnitude=True,
    )

    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_coh,
        freq=freq,
        true_psd=true_psd,
        show_coherence=True,
        outdir=out,
        filename="varma_coherence.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(spec)

    coh_ax = axes[2, 1]  # coherence panel for channels (3,2)
    median_line = next(
        line
        for line in coh_ax.lines
        if line.get_color() == "tab:blue" and line.get_label() == "Median"
    )
    true_line = next(line for line in coh_ax.lines if line.get_color() == "k")

    expected_coh = ci_dict_coh["coh"][(2, 1)][1]
    true_coh = np.abs(true_psd[:, 2, 1]) ** 2 / (
        np.abs(true_psd[:, 2, 2]) * np.abs(true_psd[:, 1, 1])
    )

    np.testing.assert_allclose(median_line.get_ydata(), expected_coh)
    np.testing.assert_allclose(true_line.get_ydata(), true_coh)
    assert np.any(expected_coh < 1.0)
    assert np.any(expected_coh > 0.0)

    plt.close(fig)

    # Test other display modes
    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_reim,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        outdir=out,
        filename="varma_base.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(spec)
    plt.close(fig)

    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_mag,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        show_csd_magnitude=True,
        outdir=out,
        filename="varma_csd_abs.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(spec)
    plt.close(fig)


def test_plot_psd_matrix_with_lisa(outdir):
    """Test PSD matrix plotting with LISA data."""
    out = f"{outdir}/{OUTDIR}"
    os.makedirs(out, exist_ok=True)

    lisa = LISAData.load()
    freq = np.asarray(lisa.freq)
    periodogram = np.asarray(lisa.matrix)
    true_psd = np.asarray(lisa.true_matrix)

    ci_dict_coh = _pack_ci_dict(
        psd_samples=periodogram[None, ...], show_coherence=True
    )
    ci_dict_reim = _pack_ci_dict(
        psd_samples=periodogram[None, ...], show_coherence=False
    )
    ci_dict_mag = _pack_ci_dict(
        psd_samples=periodogram[None, ...],
        show_coherence=False,
        show_csd_magnitude=True,
    )

    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_coh,
        freq=freq,
        true_psd=true_psd,
        show_coherence=True,
        outdir=out,
        filename="lisa_coherence.png",
        save=True,
        xscale="log",
    )
    fig, axes = plot_psd_matrix(spec)

    coh_ax = axes[2, 1]  # coherence panel for channels (Z,Y)
    median_line = next(
        line
        for line in coh_ax.lines
        if line.get_color() == "tab:blue" and line.get_label() == "Median"
    )
    true_line = next(line for line in coh_ax.lines if line.get_color() == "k")

    expected_coh = ci_dict_coh["coh"][(2, 1)][1]
    true_coh = np.abs(true_psd[:, 2, 1]) ** 2 / (
        np.abs(true_psd[:, 2, 2]) * np.abs(true_psd[:, 1, 1])
    )

    np.testing.assert_allclose(median_line.get_ydata(), expected_coh)
    np.testing.assert_allclose(true_line.get_ydata(), true_coh)
    assert expected_coh.min() >= 0.0
    assert expected_coh.max() <= 1.05

    plt.close(fig)

    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_reim,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        outdir=out,
        filename="lisa_re_im.png",
        save=True,
        xscale="log",
    )
    fig, axes = plot_psd_matrix(spec)
    assert axes[0, 1].axison
    plt.close(fig)

    spec = PSDMatrixPlotSpec(
        ci_dict=ci_dict_mag,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        show_csd_magnitude=True,
        outdir=out,
        filename="lisa_csd_abs.png",
        save=True,
        xscale="log",
    )
    fig, axes = plot_psd_matrix(spec)
    assert axes[2, 1].axison
    plt.close(fig)
