import numpy as np

from log_psplines.datatypes.univar import Periodogram
from log_psplines.mcmc import (
    _align_true_psd_to_freq,
)
from log_psplines.mcmc_utils import _prepare_true_psd_for_freq


def test_prepare_true_psd_interpolates_on_mismatched_grid_same_length():
    freq_src = np.array([0.0, 1.0, 2.0], dtype=float)
    psd = np.array([0.0, 10.0, 20.0], dtype=float)
    freq_tgt = np.array([0.5, 1.5, 2.5], dtype=float)

    aligned = _prepare_true_psd_for_freq((freq_src, psd), freq_tgt)

    np.testing.assert_allclose(aligned, np.array([5.0, 15.0, 20.0]))


def test_align_true_psd_preserves_explicit_frequency_metadata():
    freq_src = np.array([0.0, 1.0, 2.0], dtype=float)
    psd = np.array([0.0, 10.0, 20.0], dtype=float)
    freq_tgt = np.array([0.5, 1.5, 2.5], dtype=float)
    periodogram = Periodogram(freqs=freq_tgt, power=np.ones_like(freq_tgt))

    aligned = _align_true_psd_to_freq((freq_src, psd), periodogram)

    np.testing.assert_allclose(aligned, np.array([5.0, 15.0, 20.0]))
