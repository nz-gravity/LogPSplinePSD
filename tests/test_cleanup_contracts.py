"""Public contracts exposed by the architecture review."""

import numpy as np
import pytest
import xarray as xr

from log_psplines import (
    LogPSpline,
    PipelineConfig,
    PowerData,
    PSDResult,
    SplineBasis,
)


def test_tv_result_has_native_spectral_accessor():
    frequency = np.linspace(0.1, 1, 5)
    time = np.linspace(0, 1, 4)
    spline = LogPSpline(
        SplineBasis.from_grid(frequency, 0),
        time=SplineBasis.from_grid(time, 0),
    )
    data = PowerData(np.ones((4, 5)), 1, frequency, time)
    posterior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "time_coefficient", "frequency_coefficient"),
                np.zeros((1, 2, 4, 4)),
            )
        }
    )
    from log_psplines.models.reconstruction import reconstruct_power_spectrum

    result = PSDResult(
        posterior=posterior,
        sample_stats=xr.Dataset(),
        spectrum=reconstruct_power_spectrum(
            posterior, spline, time=data.time, frequency=data.frequency
        ),
    )
    assert result.posterior["weights"].shape == (1, 2, 4, 4)
    assert result.spectrum.dims == (
        "chain",
        "draw",
        "time",
        "frequency",
        "channel",
        "channel_aux",
    )
    assert result.spectral_density.shape == (1, 2, 4, 5, 1, 1)
    np.testing.assert_array_equal(result.coherence, 1)
    assert "posterior" in result.to_arviz().children


def test_public_api_has_a_single_canonical_entry_point():
    import log_psplines

    assert callable(log_psplines.fit)
    assert not hasattr(log_psplines, "run_mcmc")
    assert "fit" in log_psplines.__all__
    assert "run_mcmc" not in log_psplines.__all__


def test_unsupported_design_options_are_rejected():
    with pytest.raises(TypeError, match="design_from_vi"):
        PipelineConfig(design_from_vi=True)


def test_chain_method_reaches_numpyro(monkeypatch):
    import jax

    from log_psplines.inference import nuts

    captured = {}

    class CaptureMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kwargs)

        def run(self, *args, **kwargs):
            pass

        def get_samples(self, group_by_chain=True):
            return {}

        def get_extra_fields(self, group_by_chain=True):
            return {}

    monkeypatch.setattr(nuts, "MCMC", CaptureMCMC)
    nuts.run_nuts(
        lambda: None,
        rng_key=jax.random.PRNGKey(1),
        n_warmup=1,
        n_samples=1,
        chain_method="sequential",
    )
    assert captured["chain_method"] == "sequential"


def test_plot_failures_are_not_disguised(monkeypatch, tmp_path):
    from log_psplines.plotting import results

    result = PSDResult(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.zeros((1, 1)))}),
        spectrum=xr.DataArray(
            np.ones((1, 1, 2, 1, 1), dtype=complex),
            dims=("chain", "draw", "frequency", "channel", "channel_aux"),
            coords={
                "frequency": [1.0, 2.0],
                "channel": [0],
                "channel_aux": [0],
            },
        ),
    )

    def broken_plot(*args, **kwargs):
        raise RuntimeError("deliberate plotting error")

    monkeypatch.setattr(results, "plot_psd_matrix", broken_plot)
    with pytest.raises(RuntimeError, match="deliberate"):
        results.plot_posterior_spectrum(result, tmp_path)
    assert not (tmp_path / "posterior_spectrum.png").exists()


def test_power_data_geometry_and_observed_storage():
    from log_psplines.results import observed_power_data

    stationary = PowerData([1.0, 2.0], 1, [0.1, 0.2])
    rectangular = PowerData(np.ones((2, 2)), 1, [0.1, 0.2], [0.0, 1.0])
    scattered = PowerData([1.0, 2.0], 1, [0.1, 0.2], [0.0, 1.0])
    assert stationary.is_grid and not stationary.is_scattered
    assert rectangular.is_grid and not rectangular.is_scattered
    assert scattered.is_scattered and not scattered.is_grid
    assert observed_power_data(stationary)["power"].dims == ("frequency",)
    assert observed_power_data(rectangular)["power"].dims == (
        "time",
        "frequency",
    )
    assert observed_power_data(scattered)["power"].dims == ("ordinate",)
