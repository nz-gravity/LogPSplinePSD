"""Density, conditioning and all-draw functional contracts for native studies."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro.infer.util import log_density


@pytest.fixture
def targets(monkeypatch):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2] / "examples/vi_nuts_validation"
        )
    )
    import extension_targets

    return extension_targets


def test_matrix_joint_density_gradient_and_frozen_inputs(targets, tmp_path):
    path = tmp_path / "frozen.npz"
    targets.freeze_matrix(path)
    base = targets.load_matrix(path)
    fixed = targets.select_target(base, "fixed")
    assert targets.density_preflight(base, fixed) == {
        "conditional_vs_native_log_density_error": 0.0,
        "joint_vs_native_block_sum_error": 0.0,
    }
    assert base.identity != fixed.identity
    point = {
        **fixed.init,
        **{
            k: jnp.asarray(v)
            for k, v in fixed.descriptor["conditioned_sigma"].items()
        },
    }

    def native(weights):
        return log_density(
            base.model, (), {}, {**point, "weights_delta_1": weights}
        )[0]

    def conditional(weights):
        return log_density(
            fixed.model, (), {}, {**fixed.init, "weights_delta_1": weights}
        )[0]

    np.testing.assert_array_equal(
        jax.grad(native)(point["weights_delta_1"]),
        jax.grad(conditional)(point["weights_delta_1"]),
    )

    # The wrapper uses exactly the native sum, including both channel priors.
    def blocked(weights):
        p = {**point, "weights_delta_1": weights}
        return sum(
            log_density(
                partial(
                    targets._blocked_channel_model,
                    **targets.channel_model_kwargs(base.pair, j),
                ),
                (),
                {},
                p,
            )[0]
            for j in range(2)
        )

    np.testing.assert_array_equal(
        jax.grad(native)(point["weights_delta_1"]),
        jax.grad(blocked)(point["weights_delta_1"]),
    )
    a = dict(np.load(path))
    a["basis_delta_0"][0, 0] += 0.01
    np.savez_compressed(path, **a)
    with pytest.raises(AssertionError):
        targets.load_matrix(path)


def test_matrix_functionals_integrate_each_joint_draw(targets, tmp_path):
    path = tmp_path / "frozen.npz"
    targets.freeze_matrix(path)
    target = targets.select_target(targets.load_matrix(path), "fixed")
    # Analytic constant Cholesky surfaces with different joint draw amplitudes.
    amplitude = np.array([1.0, 2.0, 4.0])
    data = {}
    for name in targets.COMPONENTS:
        w = np.zeros((1, 3, 8))
        if name == "delta_0":
            w[:] = np.log(amplitude)[None, :, None]
        elif name == "delta_1":
            w[:] = np.log(3 * amplitude)[None, :, None]
        elif name == "theta_re_1_0":
            w[:] = 0.5
        data[f"weights_{name}"] = (("chain", "draw", f"{name}_k"), w)
    values, names = targets.feature_values(target, xr.Dataset(data))
    np.testing.assert_allclose(
        np.exp(values[0, :, names.index("log_band_c0_b0")]), amplitude * 0.14
    )
    np.testing.assert_allclose(
        np.exp(values[0, :, names.index("log_band_c1_b0")]),
        amplitude * 3.25 * 0.14,
    )
    np.testing.assert_allclose(
        values[0, :, names.index("csd_re_f0.25")], 0.5 * amplitude
    )
    np.testing.assert_allclose(
        values[0, :, names.index("coherence_f0.25")], 0.25 / 3.25
    )


def test_unresolved_reference_blocks_vi(targets, tmp_path, monkeypatch):
    import extensions

    cfg = {
        "cases": ["matrix", "time_varying"],
        "smoothing": ["fixed", "hierarchical"],
        "seeds": [7101, 7102, 7103],
    }
    monkeypatch.setattr(extensions, "prepare", lambda out: cfg)
    calls = []

    def dispatch(out, args, name):
        calls.append(args)
        assert args[0] == "reference"
        return False

    monkeypatch.setattr(extensions, "dispatch", dispatch)
    extensions.execute(tmp_path)
    assert len(calls) == 4
    for case in cfg["cases"]:
        import json

        result = json.loads(
            (tmp_path / case / "fixed/analysis.json").read_text()
        )
        assert not result["gate"]["exploratory_hierarchy_unlock"]
        assert len(result["seeds"]) == 3
