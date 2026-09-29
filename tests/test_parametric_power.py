"""Generic multi-channel power inference and bounded-result contracts."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer.util import log_density

from log_psplines import ParametricSpectrum, PowerConfig, PowerData, fit
from log_psplines.inference.parametric_power import (
    prepare_parametric_power_model,
)
from log_psplines.inference.power_results import power_result_spectra
from log_psplines.models.reconstruction import power_draws_from_basis
from log_psplines.results import PSDResult


def test_parametric_joint_likelihood_and_truth_independence(tmp_path):
    time, frequency = np.arange(3.0), np.arange(1.0, 5.0)
    template = jnp.asarray(np.arange(1.0, 25.0).reshape(3, 4, 2))
    model = ParametricSpectrum(
        lambda pars, section: jnp.exp(pars["scale"]) * template[:, section],
        {"scale": dist.Normal(0.0, 0.5)},
        {"scale": 0.0},
    )
    counts = np.ones((3, 4, 2))
    counts[0, 0, 1] = 0
    powers = np.asarray(template) * counts
    data = PowerData(
        powers, counts, frequency, time, channels=np.array(["A", "E"])
    )
    value = 0.2
    density, _ = log_density(
        prepare_parametric_power_model(data, model), (), {}, {"scale": value}
    )
    variance = np.exp(value) * np.asarray(template)
    expected = -0.5 * np.sum(
        counts * np.log(variance) + powers / variance
    ) + dist.Normal(0.0, 0.5).log_prob(value)
    np.testing.assert_allclose(density, expected, rtol=1e-6)
    config = PowerConfig(
        n_warmup=4,
        n_samples=5,
        num_chains=2,
        spectrum_draws=1,
        max_tree_depth=3,
        progress_bar=False,
    )
    result = fit(data, config, model=model, true_psd=np.asarray(template))
    second = fit(data, config, model=model, true_psd=2 * np.asarray(template))
    xr.testing.assert_equal(result.posterior, second.posterior)
    assert result.psd.shape == (2, 1, 3, 4, 2)
    assert np.isfinite(result.log_likelihood.to_array()).all()
    result.to_netcdf(tmp_path / "fit.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "fit.nc")
    assert loaded.posterior.sizes["draw"] == 5
    assert loaded.spectrum.sizes["draw"] == 1
    xr.testing.assert_equal(loaded.quantiles(), result.quantiles())
    np.testing.assert_allclose(loaded.truth, template)
    assert np.linalg.eigvalsh(loaded.spectrum).min() > 0
    with pytest.raises(ValueError, match="Only 5/50/95"):
        loaded.quantiles((25.0,))


def test_chunked_summary_uses_every_draw():
    draws = np.arange(1.0, 2 * 7 * 3 * 5 * 2 + 1).reshape(2, 7, 3, 5, 2)
    posterior = xr.Dataset({"scale": (("chain", "draw"), np.ones((2, 7)))})
    preview, summary = power_result_spectra(
        posterior,
        lambda s: draws[:, :, :, s],
        np.arange(3),
        np.arange(5),
        ("A", "E"),
        PowerConfig(spectrum_draws=2, spectrum_chunk_size=2),
    )
    np.testing.assert_array_equal(
        np.diagonal(preview, axis1=-2, axis2=-1), draws[:, [0, 6]]
    )
    np.testing.assert_allclose(
        np.diagonal(summary["quantiles"], axis1=-2, axis2=-1),
        np.percentile(draws, [5, 50, 95], axis=(0, 1)),
    )
    np.testing.assert_allclose(
        np.diagonal(summary["mean"], axis1=-2, axis2=-1),
        draws.mean(axis=(0, 1)),
    )


def test_default_anova_and_saved_basis_reconstruct_all_draws(tmp_path):
    time, freq = np.linspace(0, 1, 6), np.linspace(0.1, 1, 7)
    reference = np.exp(time[:, None] + freq[None, :])
    config = PowerConfig(
        structure="anova",
        n_interior_knots_time=1,
        n_interior_knots_freq=1,
        n_warmup=2,
        n_samples=4,
        spectrum_draws=1,
        max_tree_depth=3,
        progress_bar=False,
    )
    result = fit(
        PowerData(reference, 1, freq, time), config, reference=reference
    )
    result.to_netcdf(tmp_path / "anova.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "anova.nc")
    all_draws = power_draws_from_basis(loaded.posterior, loaded.model_data)
    np.testing.assert_allclose(
        np.diagonal(loaded.quantiles(), axis1=-2, axis2=-1),
        np.percentile(all_draws, [5, 50, 95], axis=(0, 1)),
    )
    assert loaded.metadata["model"] == "anova"
