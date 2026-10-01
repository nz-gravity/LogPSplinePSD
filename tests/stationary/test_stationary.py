"""Stationary public fits on VARMA examples with known spectra."""

import csv

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from log_psplines import PSDResult, SpectralMatrix, StationaryConfig, fit
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix


@pytest.mark.slow
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


@pytest.fixture
def matrix_result():
    rng = np.random.default_rng(25)
    logs = rng.normal(scale=0.6, size=(2, 5, 7, 2))
    theta = rng.normal(scale=0.4, size=(2, 5, 7, 1))
    values = SpectralMatrix(2)(logs, theta, -0.7 * theta)
    coords = dict(
        chain=[0, 1],
        draw=np.arange(5),
        frequency=np.linspace(0.1, 1.0, 7),
        channel=[0, 1],
        channel_aux=[0, 1],
    )
    spectrum = xr.DataArray(values, dims=tuple(coords), coords=coords)
    posterior = xr.Dataset(
        {"parameter": (("chain", "draw"), np.arange(10.0).reshape(2, 5))},
        coords={key: coords[key] for key in ("chain", "draw")},
    )
    return PSDResult(posterior=posterior, spectrum=spectrum)


@pytest.mark.parametrize(
    "kind", ["complex", "real", "imag", "magnitude", "coherence"]
)
def test_all_draw_quantiles_and_preview_roundtrip(
    matrix_result, kind, tmp_path
):
    result = matrix_result
    values = result.spectral_density
    if kind == "complex":
        expected = np.percentile(
            values.real, [5, 50, 95], axis=(0, 1)
        ) + 1j * np.percentile(values.imag, [5, 50, 95], axis=(0, 1))
    else:
        diagonal = np.diagonal(values, axis1=-2, axis2=-1).real
        transforms = {
            "real": values.real,
            "imag": values.imag,
            "magnitude": np.abs(values),
            "coherence": np.abs(values) ** 2
            / (diagonal[..., :, None] * diagonal[..., None, :]),
        }
        expected = np.percentile(transforms[kind], [5, 50, 95], axis=(0, 1))
    np.testing.assert_allclose(
        result.quantiles(kind=kind), expected, atol=1e-14
    )
    cache_name = {
        "magnitude": "magnitude_quantiles",
        "coherence": "coherence_quantiles",
    }.get(kind, "quantiles")
    cached = result.quantiles(
        kind=kind if cache_name != "quantiles" else "complex"
    )
    preview = PSDResult(
        result.posterior,
        result.spectrum.isel(draw=[0]),
        spectrum_summary=xr.Dataset({cache_name: cached}),
    )
    np.testing.assert_allclose(
        preview.quantiles(kind=kind), expected, atol=1e-14
    )
    preview.to_netcdf(tmp_path / "preview.nc")
    restored = PSDResult.from_netcdf(tmp_path / "preview.nc")
    np.testing.assert_allclose(
        restored.quantiles(kind=kind), expected, atol=1e-14
    )
    with pytest.raises(ValueError, match="preview"):
        restored.quantiles((25.0,), kind=kind)
    restored.spectrum_summary = None
    with pytest.raises(ValueError, match="preview"):
        restored.quantiles(kind=kind)


def test_quantiles_reject_incomplete_chains_and_unverified_cache(
    matrix_result,
    tmp_path,
):
    result = matrix_result
    cached = result.quantiles()
    invalid = PSDResult(result.posterior, result.spectrum.copy(deep=True))
    invalid.spectrum.values[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite draws"):
        invalid.quantiles()
    result.spectrum = result.spectrum.isel(chain=[0])
    with pytest.raises(ValueError, match="preview"):
        result.quantiles()
    result.to_netcdf(tmp_path / "partial-chains.nc")
    restored = PSDResult.from_netcdf(tmp_path / "partial-chains.nc")
    assert restored.spectrum.sizes["chain"] == 1
    with pytest.raises(ValueError, match="preview"):
        restored.quantiles()
    cached.attrs = {}
    result.spectrum_summary = xr.Dataset({"quantiles": cached})
    with pytest.raises(ValueError, match="chain/draw counts"):
        result.quantiles()


@pytest.mark.parametrize("mode", ["csd", "magnitude", "coherence"])
def test_matrix_plot_modes_use_all_draw_summaries(
    matrix_result, mode, tmp_path
):
    result = matrix_result
    result.spectrum_summary = xr.Dataset(
        {
            "quantiles": result.quantiles(),
            "magnitude_quantiles": result.quantiles(kind="magnitude"),
            "coherence_quantiles": result.quantiles(kind="coherence"),
        }
    )
    result.spectrum = result.spectrum.isel(draw=[0])
    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            result=result,
            outdir=tmp_path,
            filename=f"{mode}.png",
            show_coherence=mode == "coherence",
            show_csd_magnitude=mode == "magnitude",
            close=False,
            dpi=60,
        )
    )
    assert axes.shape == (2, 2)
    assert "PSD" in axes[0, 0].get_ylabel()
    labels = {
        "csd": "Real[CSD]",
        "magnitude": "|CSD|",
        "coherence": "Coherence",
    }
    assert labels[mode] in axes[1, 0].get_ylabel()
    if mode == "csd":
        assert "Imag[CSD]" in axes[0, 1].get_ylabel()
    assert (tmp_path / f"{mode}.png").stat().st_size > 0
    plt.close(fig)
