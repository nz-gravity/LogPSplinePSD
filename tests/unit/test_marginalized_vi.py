"""Shared VI guides and theta reconstruction for both formulations."""

import importlib
from dataclasses import replace

import jax
import numpy as np
import pytest
from scipy.linalg import block_diag

from log_psplines import PSDResult, StationaryConfig, WishartData, fit
from log_psplines.inference.model import prepare_model
from log_psplines.inference.vi import VIResult, run_multivariate_vi


def _example(channels=3):
    rng = np.random.default_rng(417)
    n, replicates, duration = 14, 12, 2.3
    lower = np.array(
        [[1, 0, 0], [0.4 + 0.3j, 0.8, 0], [-0.3 + 0.2j, 0.2 - 0.4j, 0.7]]
    )[:channels, :channels]
    draws = (
        rng.normal(size=(n, channels, replicates))
        + 1j * rng.normal(size=(n, channels, replicates))
    ) / np.sqrt(2)
    draws = np.sqrt(duration) * lower @ draws
    u = np.linalg.cholesky(draws @ draws.conj().swapaxes(-1, -2))
    data = WishartData(
        u_re=u.real,
        u_im=u.imag,
        freq=np.linspace(0.01, 0.4, n),
        N=n,
        p=channels,
        Nb=replicates,
        duration=duration,
        enbw=1.2,
    )
    config = StationaryConfig(
        method="vi",
        n_knots={"delta": 4, "theta_re": 3, "theta_im": 2},
        degree=1,
        diffMatrixOrder=1,
        knot_kwargs={"method": "uniform"},
        eta=0.6,
        vi_steps=12,
        vi_posterior_draws=16,
        vi_early_stopping=False,
        verbose=False,
    )
    return data, config


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize(
    "marginalize_theta,theta_backend",
    [(True, "dense"), (True, "banded"), (False, "auto")],
)
def test_public_vi_guides_and_outputs_follow_theta_formulation(
    parameterization, marginalize_theta, theta_backend, monkeypatch, tmp_path
):
    module = importlib.import_module("log_psplines.inference.vi")
    original = module.resolve_guide
    guides = []

    def collect_guide(*args, **kwargs):
        guide, name = original(*args, **kwargs)
        guides.append(guide)
        return guide, name

    monkeypatch.setattr(module, "resolve_guide", collect_guide)
    data, config = _example()
    config = replace(
        config,
        smoothing_parameterization=parameterization,
        theta_backend=theta_backend,
        marginalize_theta=marginalize_theta,
    )
    result = fit(data, config)
    model_kwargs, _ = prepare_model(data, config)
    assert len(guides) == data.p
    for j, guide in enumerate(guides):
        sample_names = {
            name
            for name, site in guide.prototype_trace.items()
            if site["type"] == "sample" and not site.get("is_observed")
        }
        weight_name = f"weights_delta_{j}"
        if parameterization == "noncentered":
            weight_name += "_raw"
        expected_names = {
            weight_name,
            f"sigma_delta_{j}",
            *(
                f"sigma_theta_{part}_{j}_{i}"
                for i in range(j)
                for part in ("re", "im")
            ),
        }
        dimension = model_kwargs["bases_delta"][j].shape[1] + 1 + 2 * j
        if not marginalize_theta:
            for i in range(j):
                for part in ("re", "im"):
                    name = f"weights_theta_{part}_{j}_{i}"
                    expected_names.add(
                        name + "_raw"
                        if parameterization == "noncentered"
                        else name
                    )
                    dimension += model_kwargs[f"bases_theta_{part}"][j][
                        i
                    ].shape[1]
        assert sample_names == expected_names
        assert guide.latent_dim == dimension
        for i in range(j):
            for part in ("re", "im"):
                weights = result.posterior[
                    f"weights_theta_{part}_{j}_{i}"
                ].values
                assert weights.shape == (
                    1,
                    config.vi_posterior_draws,
                    model_kwargs[f"bases_theta_{part}"][j][i].shape[1],
                )
                assert np.isfinite(weights).all()
                assert np.any(np.var(weights, axis=1) > 0)

    assert np.isfinite(result.log_likelihood.log_likelihood).all()
    spectrum = np.asarray(result.spectral_density)
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(spectrum)
    # This short VI contract test is not a converged fit. Noncentred guide
    # draws can be ill-conditioned: allow only eigensolver roundoff relative
    # to each matrix's scale, without adding a ridge to the fitted spectrum.
    tolerance = (
        32
        * np.finfo(spectrum.real.dtype).eps
        * np.max(np.abs(eigenvalues), axis=-1, keepdims=True)
    )
    assert np.all(eigenvalues >= -tolerance)
    assert np.all(np.diagonal(spectrum.real, axis1=-2, axis2=-1) > 0)
    path = tmp_path / "marginalized_vi.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.metadata["theta_backend"] == theta_backend
    assert restored.metadata["theta_parameterization"] == (
        "marginalized" if marginalize_theta else "sampled"
    )
    np.testing.assert_array_equal(
        restored.spectral_density, result.spectral_density
    )
    for name in result.posterior:
        np.testing.assert_array_equal(
            restored.posterior[name], result.posterior[name]
        )
    if marginalize_theta:
        assert restored.vi.timings["conditional_draw_seconds"] > 0
    else:
        assert "conditional_draw_seconds" not in restored.vi.timings


@pytest.mark.parametrize("guide", ["mvn", "lowrank:1", "flow:1", "flowbnaf:1"])
def test_reduced_multivariate_vi_supports_gaussian_and_flow_guides(guide):
    data, config = _example(channels=2)
    config = replace(config, vi_guide=guide, vi_steps=3, vi_posterior_draws=6)
    result = fit(data, config)
    assert result.vi.guide_name == guide
    assert np.isfinite(result.vi.losses).all()
    for name in ("weights_theta_re_1_0", "weights_theta_im_1_0"):
        assert result.posterior[name].shape[:2] == (1, 6)
        assert np.isfinite(result.posterior[name]).all()


def _freeze_reduced_guide(monkeypatch, n_draws):
    """Hold q(z) fixed to isolate the conditional reconstruction contract."""
    module = importlib.import_module("log_psplines.inference.vi")

    def fixed_guide(model, **kwargs):
        values = {
            name: np.broadcast_to(value, (n_draws, *np.shape(value)))
            for name, value in kwargs["init_values"].items()
        }
        return VIResult(
            module._values_to_dataset(values),
            np.zeros(1),
            "frozen-test-guide",
            timings={"steps_run": 0},
        )

    monkeypatch.setattr(module, "fit_vi", fixed_guide)


def test_vi_restores_joint_conditional_covariance_with_fixed_hyperparameters(
    monkeypatch,
):
    count = 3000
    _freeze_reduced_guide(monkeypatch, count)
    data, config = _example()
    kwargs, _ = prepare_model(data, config)
    result = run_multivariate_vi(
        kwargs,
        rng_key=jax.random.PRNGKey(98),
        eta=config.eta,
        posterior_draws=count,
    )
    j = 2
    posterior = result.posterior
    weights = posterior[f"weights_delta_{j}"].values[0, 0]
    logd = np.asarray(kwargs["bases_delta"][j]) @ weights
    u = np.asarray(data.u_re) + 1j * np.asarray(data.u_im)
    observed = u[:, j, :]
    columns, priors, draws = [], [], []
    for i in range(j):
        for part in ("re", "im"):
            basis = np.asarray(kwargs[f"bases_theta_{part}"][j][i])
            columns.append(
                u[:, i, :, None]
                * basis[:, None, :]
                * (1 if part == "re" else 1j)
            )
            sigma = float(
                posterior[f"sigma_theta_{part}_{j}_{i}"].values[0, 0]
            )
            priors.append(
                np.asarray(kwargs[f"penalties_theta_{part}"][j][i]) / sigma**2
            )
            draws.append(posterior[f"weights_theta_{part}_{j}_{i}"].values[0])
    design_complex = np.concatenate(columns, axis=-1).reshape(
        -1, sum(p.shape[0] for p in priors)
    )
    design = np.concatenate([design_complex.real, design_complex.imag])
    y = np.concatenate([observed.real.ravel(), observed.imag.ravel()])
    precision = np.tile(
        np.repeat(
            config.eta / (data.enbw * data.duration * np.exp(logd)), data.p
        ),
        2,
    )
    conditional_precision = block_diag(*priors) + 2 * design.T @ (
        precision[:, None] * design
    )
    covariance = np.linalg.inv(conditional_precision)
    mean = np.linalg.solve(
        conditional_precision, 2 * design.T @ (precision * y)
    )
    restored = np.concatenate(draws, axis=-1)
    # Whitening detects the full covariance, including cross-coefficient
    # dependence. Returning conditional means would fail this check.
    white = np.linalg.solve(
        np.linalg.cholesky(covariance), (restored - mean).T
    ).T
    np.testing.assert_allclose(white.mean(axis=0), 0, atol=0.08)
    np.testing.assert_allclose(
        np.cov(white, rowvar=False), np.eye(white.shape[-1]), atol=0.07
    )
    assert np.std(posterior.weights_delta_2.values, axis=1).max() < 1e-12
    assert result.timings["conditional_draw_seconds"] > 0


def test_multivariate_zero_draw_preview_restores_one_seeded_conditional_draw(
    monkeypatch,
):
    _freeze_reduced_guide(monkeypatch, 1)
    data, config = _example(channels=2)
    kwargs, _ = prepare_model(data, config)
    options = dict(model_kwargs=kwargs, posterior_draws=0, eta=config.eta)
    first = run_multivariate_vi(rng_key=jax.random.PRNGKey(41), **options)
    repeat = run_multivariate_vi(rng_key=jax.random.PRNGKey(41), **options)
    other = run_multivariate_vi(rng_key=jax.random.PRNGKey(42), **options)
    assert first.posterior.sizes["draw"] == 1
    np.testing.assert_array_equal(
        first.posterior.weights_theta_re_1_0,
        repeat.posterior.weights_theta_re_1_0,
    )
    assert not np.array_equal(
        first.posterior.weights_theta_re_1_0,
        other.posterior.weights_theta_re_1_0,
    )
    np.testing.assert_array_equal(
        first.posterior.weights_delta_1, other.posterior.weights_delta_1
    )
