"""Conditional-spectrum initialization and shared precision contract."""

import importlib

import numpy as np
import pytest

from log_psplines import StationaryConfig, WishartData
from log_psplines.inference.model import prepare_model


def test_initial_delta_targets_conditional_power_in_likelihood_units():
    n, nb, nh, duration = 24, 4, 3, 12.0
    covariance = np.array(
        [[2.0, 0.8, -0.3], [0.8, 1.5, 0.4], [-0.3, 0.4, 1.0]]
    )
    root = np.linalg.cholesky(covariance)
    u = np.broadcast_to(root * np.sqrt(nb * nh * duration), (n, 3, 3))
    data = WishartData(
        u_re=u.copy(),
        u_im=np.zeros_like(u),
        freq=np.linspace(0.01, 0.1, n),
        N=n,
        p=3,
        Nb=nb,
        Nh=nh,
        duration=duration,
    )
    kwargs, components = prepare_model(
        data, StationaryConfig(n_knots=5, knot_kwargs={"method": "uniform"})
    )
    expected = 2 * np.log(np.diag(root))
    for j, model in enumerate(components.diagonal_models):
        np.testing.assert_allclose(model(), expected[j], atol=2e-6)
        np.testing.assert_array_equal(
            kwargs["initial_weights_delta"][j], model.weights
        )


@pytest.mark.parametrize("method", ["nuts", "vi"])
def test_marginalized_model_preparation_supports_both_methods(method):
    data = WishartData(
        u_re=np.broadcast_to(np.eye(2), (12, 2, 2)).copy(),
        u_im=np.zeros((12, 2, 2)),
        freq=np.linspace(0.1, 0.2, 12),
        N=12,
        p=2,
    )
    kwargs, _ = prepare_model(data, StationaryConfig(method=method, n_knots=4))
    assert kwargs["n_channels"] == 2


def test_multivariate_precision_guard_does_not_change_scalar_contract(
    monkeypatch,
):
    from types import SimpleNamespace

    module = importlib.import_module("log_psplines.inference.model")
    monkeypatch.setattr(
        module,
        "jax",
        SimpleNamespace(config=SimpleNamespace(x64_enabled=False)),
    )
    for p in (1, 2):
        data = WishartData(
            u_re=np.broadcast_to(np.eye(p), (12, p, p)).copy(),
            u_im=np.zeros((12, p, p)),
            freq=np.linspace(0.1, 0.2, 12),
            N=12,
            p=p,
        )
        if p == 1:
            prepare_model(data, StationaryConfig(n_knots=4))
        else:
            with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
                prepare_model(data, StationaryConfig(n_knots=4))
