"""Matched LS2/WDM targets and fits, independent of the sibling repo."""

from pathlib import Path

import jax
import numpy as np
import pytest
from numpyro.infer.util import log_density

from log_psplines import (
    LogPSpline,
    PowerSpectrum,
    PowerSplineConfig,
    PSDResult,
    SplineBasis,
    TimeSeries,
    fit,
)
from log_psplines.inference.power import prepare_power_model

REFERENCE = Path(__file__).parent / "reference" / "ls2_wdm.npz"


@pytest.fixture
def ls2():
    with jax.enable_x64(True), np.load(REFERENCE) as r:
        data = PowerSpectrum(r["power"], 1, r["frequency"], r["time"])
        model = LogPSpline(
            frequency=SplineBasis.from_grid(
                data.frequency / data.frequency[-1], 4
            ),
            time=SplineBasis.from_grid(data.time, 4),
        )
        yield r, data, model


def test_ls2_basis_and_preprocessing(ls2):
    r, data, model = ls2
    for axis, basis in [("time", model.time), ("freq", model.frequency)]:
        for name, value in [
            ("basis", basis.basis),
            ("penalty", basis.penalty),
            ("knots", basis.knots),
        ]:
            np.testing.assert_allclose(value, r[f"{name}_{axis}"], atol=1e-14)
    pytest.importorskip("wdm_transform")
    from log_psplines.preprocessing.wdm import wdm_periodogram

    prepared = wdm_periodogram(
        TimeSeries(r["data"], np.arange(512) * 0.1), nt=32
    )
    # FFT reductions can differ by a few ulps across BLAS/platform builds.
    np.testing.assert_allclose(
        prepared.power, data.power, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_array_equal(prepared.time, data.time)
    np.testing.assert_array_equal(prepared.frequency, data.frequency)
    for nt in (0, 31, 64 + 1):
        with pytest.raises(ValueError, match="WDM requires"):
            wdm_periodogram(TimeSeries(r["data"]), nt=nt)


@pytest.mark.parametrize("centered", [False, True])
def test_ls2_posterior_target_and_nuts(ls2, centered, tmp_path):
    r, data, spline = ls2
    prefix = "centered_" if centered else "noncentered_"
    config = PowerSplineConfig(
        centered=centered,
        n_warmup=24,
        n_samples=16,
        max_tree_depth=6,
        progress_bar=False,
    )
    model, init, _ = prepare_power_model(data, spline, config)
    for key, value in init.items():
        assert np.asarray(value).shape == r[prefix + "init_" + key].shape
        assert np.isfinite(value).all()

    def target(sites):
        return log_density(model, (), {}, sites)[0]

    value, gradient = jax.value_and_grad(target)(init)
    np.testing.assert_allclose(
        value, r[prefix + "log_density"], rtol=1e-5, atol=1e-5
    )
    for _key, value in gradient.items():
        assert np.isfinite(value).all()
    result = fit(data, config, model=spline)
    for key in ("s", "phi_time", "phi_freq", "log_likelihood"):
        assert result.posterior[key].shape == r[prefix + "sample_" + key].shape
        assert np.isfinite(result.posterior[key]).all()
    psd = result.psd
    assert psd.shape == (1, 16, len(data.time), len(data.frequency))
    assert np.isfinite(psd).all() and np.all(psd > 0)
    np.testing.assert_allclose(result.coherence, 1)
    for key in ("diverging", "num_steps"):
        np.testing.assert_array_equal(
            result.idata["sample_stats"][key], r[prefix + "stat_" + key]
        )
    path = tmp_path / "power.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    np.testing.assert_array_equal(restored.psd, psd)
    np.testing.assert_array_equal(restored.time, data.time)
    np.testing.assert_array_equal(restored.frequency, data.frequency)
    assert restored.metadata["units"] == data.units
    result.save(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "posterior_spectrum.png").exists()
    assert (tmp_path / "saved" / "diagnostics" / "nuts_summary.csv").exists()


def test_power_input_validation():
    with pytest.raises(ValueError, match="zero-count"):
        PowerSpectrum(
            np.ones((2, 3)),
            0 * np.eye(2, 3) + [0, 1, 1],
            np.arange(3),
            np.arange(2),
        )
    with pytest.raises(ValueError, match="finite"):
        PowerSpectrum(np.full((2, 3), np.nan), 1, np.arange(3), np.arange(2))
    with pytest.raises(ValueError, match="model="):
        fit(PowerSpectrum(np.ones((2, 3)), 1, np.arange(3), np.arange(2)))
    with pytest.raises(ValueError, match="n_samples"):
        PowerSplineConfig(n_samples=0)


def test_masked_initialization_and_small_power(ls2):
    from log_psplines.inference.power import (
        _mean_power_for_masked_initialization,
    )

    filled = _mean_power_for_masked_initialization(
        np.array([[1.0, 0.0, 9.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(filled, [[1.0, 3.0, 9.0], [5.0, 5.0, 5.0]])
    _, data, spline = ls2
    power = data.power.copy() * 1e-40
    counts = data.counts.copy()
    counts[3:5, 2:7] = 0
    power[counts == 0] = 0
    masked = PowerSpectrum(power, counts, data.frequency, data.time)
    model, init, _ = prepare_power_model(masked, spline, PowerSplineConfig())
    density, trace = log_density(model, (), {}, init)
    assert np.isfinite(density)
    assert np.isfinite(trace["log_likelihood"]["value"])
    np.testing.assert_array_equal(masked.power, power)
