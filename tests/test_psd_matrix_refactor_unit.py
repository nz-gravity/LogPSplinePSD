import numpy as np
import pytest

from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.plotting.psd_matrix import (
    PSDMatrixPlotSpec,
    _pack_ci_dict,
    plot_psd_matrix,
)


def _build_ci(p: int = 2, n: int = 32):
    freq = np.linspace(0.1, 1.0, n)
    psd_samples = np.ones((3, n, p, p), dtype=np.complex128)
    for i in range(p):
        psd_samples[:, :, i, i] *= 1.0 + i
    return freq, _pack_ci_dict(psd_samples=psd_samples, show_coherence=True)


def test_plot_psd_matrix_accepts_spec_object():
    freq, ci = _build_ci()
    spec = PSDMatrixPlotSpec(ci_dict=ci, freq=freq, save=False, close=False)
    fig, axes = plot_psd_matrix(spec)
    assert axes.shape == (2, 2)
    fig.clf()


def test_plot_psd_matrix_accepts_legacy_kwargs():
    freq, ci = _build_ci()
    fig, axes = plot_psd_matrix(ci_dict=ci, freq=freq, save=False, close=False)
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
    fig, axes = plot_psd_matrix(
        ci_dict=ci,
        freq=freq,
        empirical_psd=empirical,
        extra_empirical_psd=[empirical],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -20}],
        save=False,
        close=False,
    )
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
