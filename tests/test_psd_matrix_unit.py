import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.example_datasets.lisa_data import LISAData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.plotting.psd_matrix import _pack_ci_dict, plot_psd_matrix

OUTDIR = "out_PSD_matrix"


def _make_simple_ci_dict(
    n_channels: int = 3, n_freq: int = 1000, show_coherence: bool = True
):
    freq = np.linspace(0.1, 1.0, n_freq)
    psd_samples = np.ones((2, n_freq, n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(n_channels):
            scale = 1.0 + 0.1 * (i + j)
            psd_samples[:, :, i, j] *= scale
    ci_dict = _pack_ci_dict(
        psd_samples=psd_samples, show_coherence=show_coherence
    )
    return freq, ci_dict


def test_plot_psd_matrix_conflicting_modes_raises():
    with pytest.raises(ValueError):
        plot_psd_matrix(
            ci_dict={},
            freq=np.array([1.0]),
            show_coherence=True,
            show_csd_magnitude=True,
            save=False,
        )


def test_plot_psd_matrix_validates_axes_shape():
    freq, ci_dict = _make_simple_ci_dict()
    fig, ax = plt.subplots(1, 2)

    with pytest.raises(ValueError):
        plot_psd_matrix(
            ci_dict=ci_dict,
            freq=freq,
            fig=fig,
            ax=ax,
            save=False,
        )

    plt.close(fig)


def test_plot_psd_matrix_scales_and_limits(outdir):
    out = f"{outdir}/{OUTDIR}"
    os.makedirs(out, exist_ok=True)

    freq, ci_dict = _make_simple_ci_dict(show_coherence=True)
    freq_range = (0.2, 0.9)

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict,
        freq=freq,
        freq_range=freq_range,
        show_coherence=True,
        outdir=out,
        filename="with_coh.png",
        save=True,
    )

    diag_ax = axes[0, 0]
    coh_ax = axes[1, 0]
    upper_ax = axes[0, 1]

    assert diag_ax.get_yscale() == "log"
    assert coh_ax.get_yscale() == "linear"
    assert not upper_ax.axison
    assert coh_ax.get_ylim()[1] == pytest.approx(1.0)
    assert diag_ax.get_xlim() == pytest.approx(freq_range)

    plt.close(fig)

    freq, ci_dict = _make_simple_ci_dict(show_coherence=False)
    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict,
        freq=freq,
        freq_range=freq_range,
        show_coherence=False,
        outdir=out,
        filename="without_coh.png",
        save=True,
    )

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
    freq, ci_dict = _make_simple_ci_dict(n_channels=2, n_freq=20)
    n_freq = freq.size
    psd_emp = np.ones((n_freq, 2, 2), dtype=np.complex128)
    psd_welch = 2.0 * np.ones((n_freq, 2, 2), dtype=np.complex128)

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

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict,
        freq=freq,
        empirical_psd=empirical,
        extra_empirical_psd=[welch],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -10}],
        save=False,
        close=False,
    )

    z_by_label = {
        line.get_label(): line.get_zorder() for line in axes[0, 0].lines
    }
    assert z_by_label["Welch"] < z_by_label["Empirical"]
    plt.close(fig)


def test_plot_psd_matrix_with_varma_coherence(outdir):
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

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict_coh,
        freq=freq,
        true_psd=true_psd,
        show_coherence=True,
        outdir=out,
        filename="varma_coherence.png",
        save=True,
    )

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

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict_reim,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        outdir=out,
        filename="varma_base.png",
        save=True,
    )
    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict_mag,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        show_csd_magnitude=True,
        outdir=out,
        filename="varma_csd_abs.png",
        save=True,
    )


def test_plot_psd_matrix_with_lisa(outdir):
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

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict_coh,
        freq=freq,
        true_psd=true_psd,
        show_coherence=True,
        outdir=out,
        filename="lisa_coherence.png",
        save=True,
        xscale="log",
    )

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

    fig, axes = plot_psd_matrix(
        ci_dict=ci_dict_reim,
        freq=freq,
        true_psd=true_psd,
        show_coherence=False,
        outdir=out,
        filename="lisa_re_im.png",
        save=True,
        xscale="log",
    )
    assert axes[0, 1].axison
    plt.close(fig)

    fig, axes = plot_psd_matrix(
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
    assert axes[2, 1].axison
    plt.close(fig)
