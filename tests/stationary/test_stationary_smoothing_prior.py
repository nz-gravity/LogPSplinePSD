"""Independent prior and likelihood checks for automatic stationary coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import xarray as xr
from numpyro.infer.util import log_density
from scipy.stats import halfnorm

from log_psplines import StationaryConfig, TimeSeries, fit
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.inference.coordinates import (
    coordinate_scales,
    spline_coordinates,
)
from log_psplines.inference.log_likelihood import compute_pointwise_lnl
from log_psplines.inference.model import (
    _sample_pspline_block,
    _scalar_channel_model,
    channel_model_kwargs,
    prepare_channel_model,
    prepare_model,
)
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.preprocessing.spectral import preprocess_to_freq_domain


def test_stationary_model_samples_sigma_and_scaled_coordinates():
    assert StationaryConfig().roughness_scale == 1.28
    coordinates = spline_coordinates(np.eye(3), np.eye(3), 4.0)

    def model():
        _sample_pspline_block(
            "sigma_delta_0", "weights_delta_0", 1.28, coordinates
        )

    density, trace = log_density(
        model,
        (),
        {},
        {
            "sigma_delta_0": 1.0,
            "weights_delta_0_raw": jnp.zeros(3),
        },
    )
    assert np.isfinite(density)
    assert trace["weights_delta_0_raw"]["type"] == "sample"
    assert trace["weights_delta_0"]["type"] == "deterministic"


@pytest.mark.parametrize("sigma", [1e-7, 0.4, 10.0])
@pytest.mark.parametrize("information_weight", [0.0, 2.0, 1e6])
def test_stationary_prior_density_and_scale_gradient(
    sigma, information_weight
):
    penalty = np.array([[2.0, 0.2], [0.2, 1.0]])
    coordinates = spline_coordinates(
        np.array([[1.0, 0.1], [0.2, 1.0], [0.4, 0.6]]),
        penalty,
        information_weight,
    )
    z = jnp.array([0.3, -0.5])

    def model():
        _sample_pspline_block("sigma", "weights", 1.28, coordinates)

    def difference(log_sigma):
        sf = jnp.exp(log_sigma)
        density, trace = log_density(
            model, (), {}, {"sigma": sf, "weights_raw": z}
        )
        physical = trace["weights"]["value"]
        scale = coordinate_scales(
            sf / jnp.sqrt(coordinates["eigenvalues"]),
            coordinates["information"],
        )
        # Independent physical-basis normal density, including its normalization.
        reference = dist.MultivariateNormal(
            jnp.zeros(2), precision_matrix=jnp.asarray(penalty) / sf**2
        ).log_prob(physical)
        reference += dist.HalfNormal(1.28).log_prob(sf)
        return density - reference - jnp.log(scale).sum()

    np.testing.assert_allclose(difference(jnp.log(sigma)), 0, atol=3e-11)
    np.testing.assert_allclose(
        jax.grad(difference)(jnp.log(sigma)), 0, atol=3e-11
    )


def test_sigma_controls_the_existing_stationary_penalty():
    penalty = np.array([[2.0, 0.2], [0.2, 1.0]])
    weights = np.array([0.5, -0.3])
    coordinates = spline_coordinates(np.eye(2), penalty, 3.0)

    def model():
        _sample_pspline_block("sigma", "weights", 1.28, coordinates)

    def density(sigma):
        scale = coordinate_scales(
            sigma / jnp.sqrt(coordinates["eigenvalues"]),
            coordinates["information"],
        )
        raw = coordinates["rotation"].T @ weights / scale
        return float(
            log_density(model, (), {}, {"sigma": sigma, "weights_raw": raw})[0]
            - jnp.log(scale).sum()
        )

    low, high = 0.5, 2.0
    expected = (
        halfnorm.logpdf(low, scale=1.28)
        - halfnorm.logpdf(high, scale=1.28)
        - len(weights) * np.log(low / high)
        - 0.5 * (weights @ penalty @ weights) * (low**-2 - high**-2)
    )
    np.testing.assert_allclose(
        density(low) - density(high), expected, atol=1e-10
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_stationary_nuts_retains_physical_and_sampled_coordinates(channels):
    rng = np.random.default_rng(9)
    example = TimeSeries(rng.normal(size=(512, channels)), np.arange(512) / 64)
    result = fit(
        example,
        StationaryConfig(
            n_knots=5,
            Nb=4,
            n_warmup=20,
            n_samples=20,
            rng_key=12,
            verbose=False,
        ),
    )
    for j in range(channels):
        assert f"sigma_delta_{j}" in result.posterior
        assert f"weights_delta_{j}_raw" in result.posterior
        assert f"weights_delta_{j}" in result.posterior
    spectrum = np.asarray(result.spectral_density)
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert (
        np.isfinite(spectrum).all() and np.linalg.eigvalsh(spectrum).min() > 0
    )
    assert result.metadata["delta_coordinates"].startswith(
        "information_scaled"
    )


@pytest.mark.parametrize("eta", [0.3, 1.0])
def test_stationary_coordinates_keep_full_scalar_posterior_and_tempering(eta):
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    config = StationaryConfig(
        n_knots=5, Nb=4, wishart_window="hann", verbose=False
    )
    data = preprocess_to_freq_domain(example.ts, config)
    kwargs, _ = prepare_model(data, config)
    kwargs["eta"] = eta
    channel = channel_model_kwargs(kwargs, 0)
    coordinates = channel["delta_coordinates"]
    B = np.asarray(channel["basis_delta"]) @ np.asarray(
        coordinates["rotation"]
    )
    expected_info = (
        kwargs["Nb"]
        * kwargs["Nh"]
        * eta
        / kwargs["enbw"]
        * np.sum(B**2, axis=0)
    )
    np.testing.assert_allclose(
        coordinates["information"], expected_info, rtol=1e-13
    )
    penalty = np.asarray(channel["penalty_delta"])
    for sigma in (0.01, 0.4):
        weights = np.linspace(-0.1, 0.1, len(penalty))
        scale = coordinate_scales(
            sigma / jnp.sqrt(coordinates["eigenvalues"]),
            coordinates["information"],
        )
        raw = coordinates["rotation"].T @ weights / scale
        density = log_density(
            _scalar_channel_model,
            (),
            channel,
            {"sigma_delta_0": sigma, "weights_delta_0_raw": raw},
        )[0]
        prior = dist.MultivariateNormal(
            jnp.zeros(len(weights)),
            precision_matrix=jnp.asarray(penalty) / sigma**2,
        ).log_prob(weights)
        empty = jnp.zeros((data.N, 0))
        likelihood = wishart_log_likelihood(
            channel["basis_delta"] @ weights,
            empty,
            empty,
            channel["u_re_channel"],
            channel["u_im_channel"],
            channel["u_re_prev"],
            channel["u_im_prev"],
            Nb=channel["Nb"],
            Nh=channel["Nh"],
            duration=channel["duration"],
            enbw=channel["enbw"],
            eta=eta,
        )
        np.testing.assert_allclose(
            density - jnp.log(scale).sum(),
            prior + dist.HalfNormal(1.28).log_prob(sigma) + likelihood,
            atol=1e-7,
        )
    model, prepared, initial = prepare_channel_model(kwargs, 0)
    _, trace = log_density(model, (), prepared, initial)
    np.testing.assert_allclose(
        trace["weights_delta_0"]["value"],
        kwargs["initial_weights_delta"][0],
        atol=1e-12,
    )


@pytest.mark.parametrize("log_power_shift", [-100.0, 0.0, 100.0])
def test_pointwise_likelihood_matches_sampled_channel_likelihoods(
    log_power_shift,
):
    config = StationaryConfig(
        n_knots=5, Nb=4, verbose=False, wishart_window="hann"
    )
    data = preprocess_to_freq_domain(
        TimeSeries(
            np.random.default_rng(14).normal(size=(128, 3)),
            np.arange(128) / 32.0,
        ),
        config,
    )
    kwargs, _ = prepare_model(data, config)
    assert all(
        leaf.dtype == jnp.float64
        for leaf in jax.tree_util.tree_leaves(kwargs)
        if isinstance(leaf, jax.Array)
    )
    rng = np.random.default_rng(15)
    variables = {}
    for channel in range(data.p):
        name = f"weights_delta_{channel}"
        basis = kwargs["bases_delta"][channel]
        variables[name] = (
            ("chain", "draw", f"k_{name}"),
            rng.normal(scale=0.1, size=(1, 2, basis.shape[1])),
        )
        for kind in ("re", "im"):
            for index, basis in enumerate(
                kwargs[f"bases_theta_{kind}"][channel]
            ):
                name = f"weights_theta_{kind}_{channel}_{index}"
                variables[name] = (
                    ("chain", "draw", f"k_{name}"),
                    rng.normal(scale=0.1, size=(1, 2, basis.shape[1])),
                )
    posterior = xr.Dataset(variables, coords={"chain": [0], "draw": [0, 1]})
    baseline = compute_pointwise_lnl(
        posterior=posterior, data=data, model_kwargs=kwargs
    )
    for channel in range(data.p):
        posterior[f"weights_delta_{channel}"] += log_power_shift
    amplitude_scale = np.exp(log_power_shift / 2)
    kwargs = {
        **kwargs,
        "u_re": kwargs["u_re"] * amplitude_scale,
        "u_im": kwargs["u_im"] * amplitude_scale,
    }
    likelihood = compute_pointwise_lnl(
        posterior=posterior, data=data, model_kwargs=kwargs
    )
    determinant_shift = (
        kwargs["Nb"] * kwargs["Nh"] * log_power_shift / kwargs["enbw"]
    )
    for name in likelihood:
        channel_count = data.p if name == "log_likelihood" else 1
        np.testing.assert_allclose(
            likelihood[name],
            baseline[name] - channel_count * determinant_shift,
            rtol=2e-12,
            atol=2e-10,
        )
    for draw in range(2):
        total = 0.0
        for channel in range(data.p):
            variance = (
                kwargs["bases_delta"][channel]
                @ posterior[f"weights_delta_{channel}"].values[0, draw]
            )
            theta = {}
            for kind in ("re", "im"):
                parts = [
                    basis
                    @ posterior[
                        f"weights_theta_{kind}_{channel}_{index}"
                    ].values[0, draw]
                    for index, basis in enumerate(
                        kwargs[f"bases_theta_{kind}"][channel]
                    )
                ]
                theta[kind] = (
                    np.stack(parts, axis=-1)
                    if parts
                    else np.zeros((data.N, 0))
                )
            expected = float(
                wishart_log_likelihood(
                    jnp.asarray(variance),
                    jnp.asarray(theta["re"]),
                    jnp.asarray(theta["im"]),
                    kwargs["u_re"][:, channel, :],
                    kwargs["u_im"][:, channel, :],
                    kwargs["u_re"][:, :channel, :],
                    kwargs["u_im"][:, :channel, :],
                    Nb=kwargs["Nb"],
                    Nh=kwargs["Nh"],
                    duration=kwargs["duration"],
                    enbw=kwargs["enbw"],
                )
            )
            actual = (
                likelihood[f"log_likelihood_channel_{channel}"]
                .values[0, draw]
                .sum()
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
            total += expected
        np.testing.assert_allclose(
            likelihood.log_likelihood.values[0, draw].sum(), total, rtol=2e-6
        )
