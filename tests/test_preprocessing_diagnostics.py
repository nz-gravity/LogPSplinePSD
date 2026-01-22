import numpy as np
import pytest

from log_psplines.diagnostics.preprocessing import (
    eigenvalue_separation_diagnostics,
    save_eigenvalue_separation_plot,
)


def test_eigenvalue_separation_ratios_diagonal_matrix():
    freq = np.array([0.1, 0.2, 0.3], dtype=float)
    # Each frequency has eigenvalues 3 >= 2 >= 1
    matrix = np.zeros((freq.size, 3, 3), dtype=np.complex128)
    matrix[:, 0, 0] = 3.0
    matrix[:, 1, 1] = 2.0
    matrix[:, 2, 2] = 1.0

    diag = eigenvalue_separation_diagnostics(freq=freq, matrix=matrix)
    assert diag.eigvals_desc.shape == (3, 3)
    assert np.allclose(diag.eigvals_desc[:, 0], 3.0)
    assert np.allclose(diag.eigvals_desc[:, 1], 2.0)
    assert np.allclose(diag.eigvals_desc[:, 2], 1.0)

    r12 = diag.ratios["r_12"]
    r23 = diag.ratios["r_23"]
    assert np.allclose(r12, 2.0 / 3.0)
    assert np.allclose(r23, 1.0 / 2.0)


def test_eigenvalue_separation_handles_zero_denom():
    freq = np.array([0.1, 0.2], dtype=float)
    matrix = np.zeros((freq.size, 2, 2), dtype=np.complex128)
    # All-zero matrices -> λ1 = 0, ratio undefined
    diag = eigenvalue_separation_diagnostics(freq=freq, matrix=matrix)
    assert "r_12" in diag.ratios
    assert np.all(np.isnan(diag.ratios["r_12"]))


def test_eigenvalue_separation_quantile_mask():
    freq = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    matrix = np.zeros((freq.size, 2, 2), dtype=np.complex128)
    # λ1 varies across frequencies; λ2 fixed.
    matrix[:, 0, 0] = np.array([1.0, 2.0, 3.0, 4.0])
    matrix[:, 1, 1] = 1.0

    diag = eigenvalue_separation_diagnostics(
        freq=freq, matrix=matrix, min_lambda1_quantile=0.5
    )
    assert diag.lambda1_cutoff is not None
    assert diag.mask.shape == (freq.size,)
    assert 0 < int(np.count_nonzero(diag.mask)) < freq.size


def test_save_eigenvalue_separation_plot(tmp_path):
    pytest.importorskip("matplotlib")

    freq = np.array([0.1, 0.2, 0.3], dtype=float)
    matrix = np.zeros((freq.size, 3, 3), dtype=np.complex128)
    matrix[:, 0, 0] = 3.0
    matrix[:, 1, 1] = 2.0
    matrix[:, 2, 2] = 1.0
    diag = eigenvalue_separation_diagnostics(freq=freq, matrix=matrix)

    out = tmp_path / "eig_ratios.png"
    save_eigenvalue_separation_plot(diag, str(out), warn_threshold=0.8)
    assert out.exists()
    assert out.stat().st_size > 0
