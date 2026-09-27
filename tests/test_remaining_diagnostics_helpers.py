import numpy as np
import xarray as xr

from log_psplines.diagnostics.psd_compare import (
    _coherence,
    _compute_multivar_diagnostics_from_arrays,
    _extract_percentile_slice,
    _handle_multivariate,
    compute_multivar_riae_diagnostics,
)


def test_psd_compare_helpers_and_multivariate_dataset_path() -> None:
    freq = np.linspace(0.1, 1.0, 8)
    truth = np.zeros((8, 2, 2), dtype=np.complex128)
    estimate = np.zeros_like(truth)
    for idx in range(8):
        truth[idx] = np.asarray(
            [[2.0 + idx, 0.2 + 0.1j], [0.2 - 0.1j, 1.5 + idx]]
        )
        estimate[idx] = truth[idx] * 1.05

    coherence = _coherence(truth)
    assert coherence.shape == truth.shape
    assert np.all(coherence.real >= 0.0)
    values = np.asarray([truth * 0.9, truth, truth * 1.1])
    np.testing.assert_allclose(
        _extract_percentile_slice(values, np.asarray([5.0, 50.0, 95.0]), 50.0),
        truth,
    )

    diagnostics = _compute_multivar_diagnostics_from_arrays(
        estimate,
        truth.real,
        freq,
        posterior_psd_quantiles=values,
    )
    assert "riae_matrix" in diagnostics
    assert "riae_matrix_errorbars" in diagnostics
    assert "coverage" in diagnostics
    assert "riae_bands" in diagnostics

    public = compute_multivar_riae_diagnostics(
        estimate,
        truth.real,
        freq,
        psd_quantiles={"posterior_psd": values},
    )
    assert "coherence_riae" in public

    psd_group = xr.Dataset(
        {
            "psd_matrix_real": xr.DataArray(
                values.real,
                dims=("percentile", "freq", "channel", "channel_aux"),
                coords={"percentile": [5.0, 50.0, 95.0], "freq": freq},
            ),
            "psd_matrix_imag": xr.DataArray(
                values.imag,
                dims=("percentile", "freq", "channel", "channel_aux"),
                coords={"percentile": [5.0, 50.0, 95.0], "freq": freq},
            ),
        }
    )
    handled = _handle_multivariate(psd_group, truth.real)
    assert "riae_diag_mean" in handled
