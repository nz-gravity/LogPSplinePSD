"""Public contracts exposed by the architecture review."""

from pathlib import Path

import numpy as np
import pytest

from log_psplines import (
    LogPSpline,
    PowerSpectrum,
    PowerSplineConfig,
    SplineBasis,
)
from log_psplines.arviz_utils.from_arviz import get_psd_dataset
from log_psplines.arviz_utils.to_arviz import pack_power_result
from log_psplines.config import PipelineConfig
from log_psplines.results import PSDResult


def test_tv_result_has_the_same_public_spectral_accessor():
    frequency = np.linspace(0.1, 1, 5)
    time = np.linspace(0, 1, 4)
    spline = LogPSpline(
        SplineBasis.from_grid(frequency, 0),
        time=SplineBasis.from_grid(time, 0),
    )
    data = PowerSpectrum(np.ones((4, 5)), 1, frequency, time)
    config = PowerSplineConfig(n_samples=2)
    result = PSDResult(
        pack_power_result(
            data, spline, config, {"weights": np.zeros((1, 2, 4, 4))}, {}
        )
    )
    # PSDResult is now the public/native result boundary. The legacy DataTree
    # is retained only while the old packers are removed.
    assert result.posterior["weights"].shape == (1, 2, 4, 4)
    assert result.sample_stats is not None
    assert result.spectrum.dims == (
        "chain",
        "draw",
        "time",
        "frequency",
        "channel",
        "channel_aux",
    )
    assert "posterior" in result.to_arviz().children

    dataset = get_psd_dataset(result._tree)
    assert dataset.spectral_density.dims == (
        "chain",
        "draw",
        "channel",
        "channel_aux",
        "time",
        "frequency",
    )
    np.testing.assert_array_equal(
        dataset.spectral_density.transpose(
            "chain", "draw", "time", "frequency", "channel", "channel_aux"
        ),
        result.spectral_density,
    )
    np.testing.assert_array_equal(dataset.coherence, 1)
    from log_psplines.arviz_utils.from_arviz import (
        get_multivar_posterior_psd_quantiles,
        get_weights,
    )

    assert get_weights(result._tree).shape == (2, 4, 4)
    quantiles = get_multivar_posterior_psd_quantiles(
        result._tree, compute_coherence=False
    )
    assert quantiles["spectral_density"].shape == (3, 4, 5, 1, 1)
    assert quantiles["coherence"] is None
    np.testing.assert_array_equal(quantiles["time"], time)
    # A broken selected result must not quietly fall through to another source.
    del result._tree["posterior"]["weights"]
    with pytest.raises(KeyError, match="weights"):
        get_psd_dataset(result.idata)


def test_public_api_has_a_single_canonical_entry_point():
    import log_psplines

    assert callable(log_psplines.fit)
    assert not hasattr(log_psplines, "run_mcmc")
    assert "fit" in log_psplines.__all__
    assert "run_mcmc" not in log_psplines.__all__


def test_unsupported_design_options_are_rejected():
    with pytest.raises(TypeError, match="design_from_vi"):
        PipelineConfig(design_from_vi=True)


def test_study_basis_import_resolves():
    import ast
    import importlib

    study = (
        Path(__file__).parents[1]
        / "docs/studies/eta_testing/eta_validation_study.py"
    )
    imports = [
        node
        for node in ast.walk(ast.parse(study.read_text()))
        if isinstance(node, ast.ImportFrom)
        and any(n.name == "init_basis_and_penalty" for n in node.names)
    ]
    assert len(imports) == 1
    assert callable(
        importlib.import_module(imports[0].module).init_basis_and_penalty
    )


def test_chain_method_reaches_numpyro(monkeypatch):
    import jax

    from log_psplines import TimeSeries, make_pipeline
    from log_psplines.inference import nuts

    pipeline = make_pipeline(
        TimeSeries(np.random.default_rng(3).normal(size=32)),
        PipelineConfig(n_knots=4, chain_method="sequential"),
    )
    assert pipeline.nuts_stage.chain_method == "sequential"
    captured = {}

    class CaptureMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kwargs)

        def run(self, *args, **kwargs):
            pass

    monkeypatch.setattr(nuts, "MCMC", CaptureMCMC)
    nuts.run_nuts(
        lambda: None,
        rng_key=jax.random.PRNGKey(1),
        n_warmup=1,
        n_samples=1,
        chain_method=pipeline.nuts_stage.chain_method,
    )
    assert captured["chain_method"] == "sequential"


def test_basis_conventions_survive_storage():
    import xarray as xr

    from log_psplines.arviz_utils.spline_storage import (
        from_storage_dataset,
        to_storage_payload,
    )

    for basis in (
        SplineBasis.from_grid(np.linspace(0, 1, 8), 2),
        SplineBasis.from_knots(np.linspace(0, 1, 8), np.linspace(0, 1, 4)),
        SplineBasis.from_knots(
            np.linspace(0, 1, 8),
            np.linspace(0, 1, 4),
            normalization="trace",
            ridge=0,
        ),
    ):
        model = LogPSpline(basis)
        payload, coords = to_storage_payload(model)
        dataset = xr.Dataset(payload, coords=coords)
        restored = from_storage_dataset(dataset, degree=3, diffMatrixOrder=2)
        for name in (
            "penalty_normalization",
            "penalty_ridge",
            "knot_convention",
        ):
            assert getattr(restored.frequency, name) == getattr(basis, name)
        np.testing.assert_allclose(restored.basis, model.basis)
        np.testing.assert_allclose(
            restored.penalty_matrix, model.penalty_matrix
        )


def test_plot_failures_are_not_disguised(monkeypatch, tmp_path):
    import xarray as xr

    from log_psplines import PSDResult
    from log_psplines.plotting import results

    def broken_plot(*args, **kwargs):
        raise RuntimeError("deliberate plotting error")

    monkeypatch.setattr(results, "plot_psd_matrix", broken_plot)
    with pytest.raises(RuntimeError, match="deliberate"):
        results.plot_posterior_spectrum(PSDResult(xr.DataTree()), tmp_path)
    assert not (tmp_path / "posterior_spectrum.png").exists()
