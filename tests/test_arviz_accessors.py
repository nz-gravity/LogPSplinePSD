"""Native result accessors and the diagnostics-only ArviZ adapter."""

import numpy as np
import xarray as xr

from log_psplines.results import PSDResult


def _result() -> PSDResult:
    posterior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "coefficient"),
                np.arange(12.0).reshape(1, 3, 4),
            )
        }
    )
    stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((1, 3))),
            "n_steps": (("chain", "draw"), np.ones((1, 3))),
        }
    )
    spectrum = xr.DataArray(
        np.ones((1, 3, 5, 2, 2), dtype=np.complex128),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={
            "frequency": np.linspace(0.1, 0.5, 5),
            "channel": [0, 1],
            "channel_aux": [0, 1],
        },
        name="spectral_density",
    )
    return PSDResult(
        posterior=posterior,
        sample_stats=stats,
        spectrum=spectrum,
        metadata={"units": "test"},
    )


def test_native_result_accessors():
    result = _result()
    assert result.posterior["weights"].shape == (1, 3, 4)
    assert result.spectral_density.shape == (1, 3, 5, 2, 2)
    np.testing.assert_array_equal(result.frequency, np.linspace(0.1, 0.5, 5))
    assert result.time is None
    assert result.psd.shape == (1, 3, 5, 2)
    assert result.coherence.shape == result.spectral_density.shape

    q = result.quantiles((10.0, 50.0, 90.0))
    assert q.dims == (
        "percentile",
        "frequency",
        "channel",
        "channel_aux",
    )
    assert q.shape == (3, 5, 2, 2)


def test_arviz_is_diagnostics_adapter_only():
    result = _result()
    idata = result.to_arviz()
    assert "posterior" in idata.children
    assert "sample_stats" in idata.children
    assert "weights" in idata["posterior"].dataset
    assert "spectral_density" not in idata["posterior"].dataset


def test_native_result_round_trip(tmp_path):
    result = _result()
    path = tmp_path / "result.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    xr.testing.assert_allclose(restored.posterior, result.posterior)
    np.testing.assert_allclose(
        restored.spectral_density, result.spectral_density
    )
    assert restored.metadata["units"] == "test"
