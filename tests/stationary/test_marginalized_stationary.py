"""Public marginalized fits retain full complex-matrix posterior outputs."""

import numpy as np
import pytest

from log_psplines import PSDResult, StationaryConfig, WishartData, fit


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_marginalized_three_channel_roundtrip(parameterization, tmp_path):
    rng = np.random.default_rng(29)
    n, replicates = 24, 8
    covariance = np.full((3, 3), -0.45)
    np.fill_diagonal(covariance, 1.0)
    draws = (
        rng.normal(size=(n, 3, replicates))
        + 1j * rng.normal(size=(n, 3, replicates))
    ) / np.sqrt(2)
    draws = np.linalg.cholesky(covariance) @ draws
    u = np.linalg.cholesky(draws @ draws.conj().swapaxes(-1, -2))
    data = WishartData(
        u_re=u.real,
        u_im=u.imag,
        freq=np.linspace(0.001, 0.05, n),
        N=n,
        p=3,
        Nb=replicates,
    )
    result = fit(
        data,
        StationaryConfig(
            smoothing_parameterization=parameterization,
            n_knots={"delta": 5, "theta_re": 4, "theta_im": 3},
            degree=2,
            knot_kwargs={"method": "uniform"},
            n_warmup=20,
            n_samples=12,
            num_chains=1,
            verbose=False,
        ),
    )
    assert result.posterior.sizes["draw"] == 12
    for j in (1, 2):
        for i in range(j):
            for part in ("re", "im"):
                weights = result.posterior[f"weights_theta_{part}_{j}_{i}"]
                assert np.isfinite(weights).all()
                assert np.any(np.std(weights, axis=1) > 0)
    spectrum = np.asarray(result.spectral_density)
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(spectrum).min() > 0
    assert np.isfinite(result.log_likelihood.log_likelihood).all()
    path = tmp_path / "marginalized.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.metadata["theta_parameterization"] == "marginalized"
    np.testing.assert_allclose(restored.spectral_density, spectrum)
