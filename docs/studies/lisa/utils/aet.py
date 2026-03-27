"""XYZ <-> AET channel transformation utilities for LISA TDI channels.

The AET (or A, E, T) channels are orthogonal combinations of the XYZ Michelson
TDI channels that approximately diagonalise the noise covariance at most
frequencies.  The transformation is:

    A = (Z - X) / sqrt(2)
    E = (X - 2Y + Z) / sqrt(6)
    T = (X + Y + Z) / sqrt(3)

In matrix form  [A, E, T]^T = M_AET @ [X, Y, Z]^T  where M_AET is the 3x3
unitary-like mixing matrix defined below.

Spectral matrices transform as:
    S_AET(f) = M_AET @ S_XYZ(f) @ M_AET^H

References
----------
Prince et al. (2002); Vallisneri (2005); lisatools documentation.
"""

from __future__ import annotations

import numpy as np

# ── AET mixing matrix ──────────────────────────────────────────────────────
# Rows: [A, E, T],  Columns: [X, Y, Z]
M_AET: np.ndarray = np.array(
    [
        [-1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)],
        [1.0 / np.sqrt(6), -2.0 / np.sqrt(6), 1.0 / np.sqrt(6)],
        [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
    ],
    dtype=np.float64,
)

CHANNEL_LABELS_XYZ = ["X", "Y", "Z"]
CHANNEL_LABELS_AET = ["A", "E", "T"]


def xyz_to_aet_timeseries(y_xyz: np.ndarray) -> np.ndarray:
    """Transform (N, 3) XYZ timeseries to (N, 3) AET timeseries.

    Parameters
    ----------
    y_xyz : np.ndarray, shape (N, 3)
        Columns are [X, Y, Z] timeseries.

    Returns
    -------
    y_aet : np.ndarray, shape (N, 3)
        Columns are [A, E, T] timeseries.
    """
    y_xyz = np.asarray(y_xyz, dtype=np.float64)
    if y_xyz.ndim != 2 or y_xyz.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {y_xyz.shape}.")
    return (M_AET @ y_xyz.T).T


def xyz_to_aet_matrix(S_xyz: np.ndarray) -> np.ndarray:
    """Transform a (*, 3, 3) spectral matrix from XYZ to AET basis.

    Parameters
    ----------
    S_xyz : np.ndarray, shape (..., 3, 3)
        Spectral (cross-spectral density) matrix in XYZ basis.
        May be complex (off-diagonal cross-spectra).

    Returns
    -------
    S_aet : np.ndarray, shape (..., 3, 3)
        Spectral matrix in AET basis.
    """
    S_xyz = np.asarray(S_xyz)
    if S_xyz.shape[-2:] != (3, 3):
        raise ValueError(f"Last two dims must be (3, 3), got {S_xyz.shape}.")
    M = M_AET.astype(S_xyz.dtype)
    # S_AET = M @ S_XYZ @ M^H  (broadcast over leading frequency axis)
    return M @ S_xyz @ M.conj().T


def transform_ci_curves_to_aet(ci_source: str | dict) -> dict:
    """Return AET-transformed CI data from compact curves or an in-memory dict.

    The transformation is applied independently at each percentile (q05, q50,
    q95) to keep the quantile structure consistent.  For each percentile k the
    full complex spectral matrix is formed, rotated to AET, then split back into
    real and imaginary parts.

    Parameters
    ----------
    ci_source : str or dict
        Either a path to ``compact_ci_curves.npz`` or a dict with the same keys.

    Returns
    -------
    dict with keys matching compact_ci_curves.npz but in the AET basis:
        freq, psd_real_q05, psd_real_q50, psd_real_q95,
        psd_imag_q05, psd_imag_q50, psd_imag_q95,
        offdiag_pairs, true_psd_real, true_psd_imag
    """
    if isinstance(ci_source, dict):
        data = {k: np.asarray(v) for k, v in ci_source.items()}
    else:
        npz = np.load(ci_source, allow_pickle=True)
        data = {k: np.asarray(npz[k]) for k in npz.files}

    freq = np.asarray(data["freq"])
    Nf = len(freq)

    def _transform_percentile(
        real_key: str, imag_key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        S_re = np.asarray(data[real_key])  # (Nf, 3, 3)
        S_im = np.asarray(data[imag_key])  # (Nf, 3, 3)
        S_xyz = S_re + 1j * S_im
        # Force diagonal to be purely real (auto-spectra are real by definition)
        for k in range(3):
            S_xyz[:, k, k] = S_xyz[:, k, k].real.astype(complex)
        S_aet = xyz_to_aet_matrix(S_xyz)  # (Nf, 3, 3)
        return S_aet.real, S_aet.imag

    re_q05, im_q05 = _transform_percentile("psd_real_q05", "psd_imag_q05")
    re_q50, im_q50 = _transform_percentile("psd_real_q50", "psd_imag_q50")
    re_q95, im_q95 = _transform_percentile("psd_real_q95", "psd_imag_q95")
    re_true, im_true = _transform_percentile("true_psd_real", "true_psd_imag")

    p = 3
    offdiag_pairs = np.array(
        [(i, j) for i in range(p) for j in range(i + 1, p)], dtype=int
    )

    return dict(
        freq=freq,
        psd_real_q05=re_q05,
        psd_real_q50=re_q50,
        psd_real_q95=re_q95,
        psd_imag_q05=im_q05,
        psd_imag_q50=im_q50,
        psd_imag_q95=im_q95,
        offdiag_pairs=offdiag_pairs,
        true_psd_real=re_true,
        true_psd_imag=im_true,
    )
