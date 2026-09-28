"""Stationary public fits on VARMA examples with known spectra."""

import csv

import numpy as np
import pytest

from log_psplines import PSDResult, StationaryConfig, fit
from log_psplines.example_datasets.varma_data import VARMAData


@pytest.mark.parametrize(
    ("channels", "method"),
    [(1, "vi"), (1, "nuts"), (2, "vi"), (2, "nuts")],
)
def test_fit_result_roundtrip_and_scientific_invariants(
    channels, method, outdir
):
    if channels == 1:
        example = VARMAData.ar(order=4, n_samples=2048, fs=64.0, seed=7)
    else:
        example = VARMAData(n_samples=2048, fs=64.0, seed=7)
    artifact_dir = outdir / "stationary" / f"{channels}-channel-{method}"
    config = StationaryConfig(
        n_knots=10 if channels == 1 else 8,
        degree=3 if channels == 1 else 2,
        Nb=8,
        method=method,
        vi_steps=1500,
        vi_posterior_draws=80,
        n_samples=100,
        n_warmup=100,
        num_chains=2 if method == "nuts" else 1,
        chain_method="sequential",
        rng_key=7,
        true_psd=(example.freq, example.get_true_psd()),
        verbose=False,
        outdir=str(artifact_dir),
        vi_progress_bar=False,
    )
    result = fit(example.ts, config)

    assert isinstance(result, PSDResult)
    assert result.posterior is not None
    chains = 2 if method == "nuts" else 1
    draws = 100 if method == "nuts" else 80
    assert result.posterior.sizes["chain"] == chains
    assert result.posterior.sizes["draw"] == draws
    spectrum = np.asarray(result.spectral_density)
    assert spectrum.shape[:3] == (chains, draws, result.frequency.size)
    assert spectrum.shape[-2:] == (channels, channels)
    assert np.isfinite(spectrum).all()
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(spectrum).min() > 0
    coherence = np.asarray(result.coherence)
    assert np.all((coherence >= 0) & (coherence <= 1 + 1e-10))

    saved_result = artifact_dir / "inference_data.nc"
    diagnostics_dir = artifact_dir / "diagnostics"
    assert saved_result.exists()
    assert (artifact_dir / "posterior_spectrum.png").exists()
    assert (diagnostics_dir / "spectrum_summary.csv").exists()
    assert (diagnostics_dir / f"{method}_summary.csv").exists()
    assert list(diagnostics_dir.glob("*.png"))
    with (diagnostics_dir / "spectrum_summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert np.isfinite(float(summary["riae"]))
    assert float(summary["riae"]) < 0.6
    assert float(summary["coverage"]) > 0.4

    restored = PSDResult.from_netcdf(saved_result)
    np.testing.assert_allclose(restored.spectral_density, spectrum)
    assert restored.posterior is not None
    assert restored.posterior.sizes["draw"] == result.posterior.sizes["draw"]
