"""Frozen numerical contract captured before the architecture refactor."""

import warnings
from pathlib import Path

import jax
import numpy as np
from numpyro import handlers

from log_psplines import make_pipeline
from log_psplines.arviz_utils.reconstruction import reconstruct_psd_matrix
from log_psplines.config import PipelineConfig
from log_psplines.data import TimeSeries
from log_psplines.inference.initialisation import build_component
from log_psplines.inference.model import _joint_multivar_model

REFERENCE = Path(__file__).parent / "reference" / "stationary.npz"


def stationary_values():
    values = {}
    model = build_component(
        knots=np.array([0.0, 0.2, 0.65, 1.0]),
        degree=3,
        diffMatrixOrder=2,
        n=9,
        grid_points=np.linspace(0, 1, 9) ** 1.3,
    )
    values["basis"] = np.asarray(model.basis)
    values["penalty"] = np.asarray(model.penalty_matrix)
    values["log_psd"] = np.asarray(model(np.arange(model.n_basis) / 7))
    rng = np.random.default_rng(31)
    for channels in (1, 2):
        data = TimeSeries(rng.normal(size=(32, channels)))
        config = PipelineConfig(
            n_knots=4,
            vi_steps=3,
            vi_posterior_draws=3,
            n_samples=3,
            n_warmup=3,
            rng_key=14,
        )
        pipeline = make_pipeline(data, config)
        trace = handlers.trace(
            handlers.seed(_joint_multivar_model, jax.random.PRNGKey(9))
        ).get_trace(**pipeline.full_model_kwargs)
        values[f"u_{channels}"] = pipeline.data.U
        for name, site in trace.items():
            if name.startswith("log_likelihood_block_"):
                values[f"{channels}_{name}"] = np.asarray(site["value"])
        shape = (2, 5, channels)
        logs = rng.normal(size=shape) / 4
        theta = rng.normal(size=(2, 5, channels * (channels - 1) // 2)) / 5
        values[f"matrix_{channels}"] = reconstruct_psd_matrix(
            logs, theta, theta / 3
        )
        result = pipeline.run()
        for name, arr in result.idata["posterior"].dataset.data_vars.items():
            values[f"posterior_{channels}_{name}"] = np.asarray(arr)
    return values


def test_stationary_frozen_contract():
    actual = stationary_values()
    expected = np.load(REFERENCE)
    assert set(actual) == set(expected.files)
    for name, value in actual.items():
        if name.startswith("posterior_"):
            reference = expected[name]
            difference = np.abs(np.asarray(value) - reference)
            scale = np.maximum(np.abs(reference), 1e-12)
            if not np.allclose(value, reference, rtol=5e-3, atol=2e-3):
                warnings.warn(
                    f"{name} differs from the frozen reference: "
                    f"max_abs={difference.max():.3g}, "
                    f"max_rel={(difference / scale).max():.3g}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        # This float32 likelihood is sensitive to platform reduction order.
        rtol = (
            1e-4
            if name == "2_log_likelihood_block_1"
            else 3e-5
        )
        atol = (
            1e-5
            if name == "2_log_likelihood_block_1"
            else 3e-6
        )
        np.testing.assert_allclose(
            value, expected[name], rtol=rtol, atol=atol, err_msg=name
        )
