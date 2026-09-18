"""Frozen numerical contract captured before the architecture refactor."""
from pathlib import Path
import os

import jax
import numpy as np
from numpyro import handlers

from log_psplines import make_pipeline
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.psplines.psplines import LogPSplines

REFERENCE = Path(__file__).parent / "reference" / "stationary.npz"


def stationary_values():
    values = {}
    model = LogPSplines.from_knots(
        knots=np.array([0., .2, .65, 1.]), degree=3,
        diffMatrixOrder=2, n=9, grid_points=np.linspace(0, 1, 9)**1.3,
    )
    values["basis"] = np.asarray(model.basis)
    values["penalty"] = np.asarray(model.penalty_matrix)
    values["log_psd"] = np.asarray(model(np.arange(model.n_basis) / 7))
    rng = np.random.default_rng(31)
    for channels in (1, 2):
        data = MultivariateTimeseries(rng.normal(size=(32, channels)))
        config = PipelineConfig(n_knots=4, vi_steps=3, vi_posterior_draws=3,
                                n_samples=3, n_warmup=3, rng_key=14)
        pipeline = make_pipeline(data, config)
        trace = handlers.trace(handlers.seed(pipeline.model_fn, jax.random.PRNGKey(9))).get_trace(**pipeline.full_model_kwargs)
        values[f"u_{channels}"] = pipeline.data.U
        for name, site in trace.items():
            if name.startswith("log_likelihood_block_"):
                values[f"{channels}_{name}"] = np.asarray(site["value"])
        shape = (2, 5, channels)
        logs = rng.normal(size=shape) / 4
        theta = rng.normal(size=(2, 5, channels*(channels-1)//2)) / 5
        values[f"matrix_{channels}"] = pipeline.spline_model.reconstruct_psd_matrix(logs, theta, theta/3)
        result = pipeline.run()
        for name, arr in result.idata["posterior"].dataset.data_vars.items():
            values[f"posterior_{channels}_{name}"] = np.asarray(arr)
    return values


def test_stationary_frozen_contract():
    actual = stationary_values()
    if os.environ.get("CAPTURE_STATIONARY_REFERENCE") == "1":
        np.savez(REFERENCE, **actual)
    expected = np.load(REFERENCE)
    assert set(actual) == set(expected.files)
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], rtol=3e-5, atol=3e-6, err_msg=name)
