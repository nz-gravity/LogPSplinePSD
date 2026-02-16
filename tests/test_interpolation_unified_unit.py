import numpy as np

from log_psplines.datatypes.multivar_utils import (
    _interp_complex_matrix,
    interp_matrix,
)
from log_psplines.mcmc import _interp_psd_array


def test_matrix_interpolators_match_on_unsorted_duplicate_grid():
    freq_src = np.array([0.2, 0.0, 0.1, 0.1], dtype=float)
    freq_tgt = np.array([0.0, 0.05, 0.1, 0.15, 0.2], dtype=float)

    mat = np.zeros((4, 2, 2), dtype=np.complex128)
    mat[:, 0, 0] = np.array([2.0, 0.0, 1.0, 1.5])
    mat[:, 1, 1] = np.array([4.0, 2.0, 3.0, 3.5])
    mat[:, 0, 1] = np.array([1.0j, 0.0j, 0.5j, 0.75j])
    mat[:, 1, 0] = np.conj(mat[:, 0, 1])

    out_a = interp_matrix(freq_src, mat, freq_tgt)
    out_b = _interp_complex_matrix(freq_src, freq_tgt, mat)

    np.testing.assert_allclose(out_a, out_b)


def test_psd_and_matrix_interpolators_are_consistent():
    freq_src = np.array([0.2, 0.0, 0.1, 0.1], dtype=float)
    freq_tgt = np.array([0.0, 0.05, 0.1, 0.15, 0.2], dtype=float)
    psd = np.array([20.0, 0.0, 10.0, 15.0], dtype=float)

    out_psd = _interp_psd_array(psd, freq_src, freq_tgt)
    out_matrix = _interp_complex_matrix(
        freq_src,
        freq_tgt,
        psd[:, None],
    )[:, 0]

    np.testing.assert_allclose(out_psd, out_matrix)
