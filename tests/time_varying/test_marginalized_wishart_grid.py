"""Independent Gaussian integrals and the matched stationary tensor limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag

from log_psplines import (
    LogPSpline,
    PowerConfig,
    SplineBasis,
    WishartGridData,
)
from log_psplines.inference.collapsed import gaussian_theta_conditional
from log_psplines.inference.collapsed_grid import (
    grid_theta_conditional,
    grid_theta_scales,
    prepare_grid_theta,
    recover_grid_theta,
)
from log_psplines.inference.wishart_grid import grid_design, prepare_grid_prior
from log_psplines.likelihoods.wishart import wishart_log_likelihood


def _fixture(
    *, enbw=1.0, stationary=False, explicit_constant=False, log_scale=0.0
):
    rng = np.random.default_rng(2348)
    time, frequency = np.array([0.0, 1.0]), np.linspace(0.1, 1.0, 5)
    lower = np.array(
        [[1.0, 0, 0], [0.7 + 0.4j, 0.8, 0], [-0.2 + 0.6j, 0.5 - 0.1j, 0.7]]
    )
    z = (
        rng.normal(size=(2, 5, 3, 4)) + 1j * rng.normal(size=(2, 5, 3, 4))
    ) / np.sqrt(2)
    counts = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    coefficients = np.exp(log_scale / 2) * (lower @ z)
    coefficients *= (np.arange(4) < counts[..., None])[..., None, :]
    data = WishartGridData(
        coefficients.real,
        coefficients.imag,
        time,
        frequency,
        counts,
        enbw=enbw,
    )
    frequency_basis = SplineBasis.from_grid(
        frequency,
        1,
        degree=1,
        penalty_order=1,
        normalization="max",
        ridge=0.04,
    )
    time_basis = (
        None
        if stationary
        else SplineBasis.from_grid(time, 0, degree=1, penalty_order=1)
    )
    if explicit_constant:
        time_basis = SplineBasis(
            time,
            np.array([0.0, 1.0]),
            np.ones((len(time), 1)),
            np.zeros((1, 1)),
            degree=0,
            penalty_order=0,
            penalty_normalization="trace",
            penalty_ridge=0.0,
            knot_convention="clamped",
        )
    spline = LogPSpline(frequency_basis, time_basis)
    # These scalar-ANOVA settings must not silently alter the matrix prior.
    config = PowerConfig(
        structure="tensor", null_precision=0.7, ridge_eps=0.03
    )
    pair = prepare_grid_prior(spline)
    design = grid_design(data, spline, pair)
    return data, spline, config, pair, design


def _complex_design(data, design, channel):
    """Independent complex design: multiply imaginary-theta columns by i."""
    bt, bf = design[2:]
    field = np.einsum("ti,fj->tfij", bt, bf).reshape(len(bt), len(bf), -1)
    return np.concatenate(
        [
            data.U[..., previous, :, None] * field[..., None, :] * multiplier
            for previous in range(channel)
            for multiplier in (1, 1j)
        ],
        axis=-1,
    )


def _dense_marginal(logs, scales, data, design, channel):
    """Integrate using an observation-space covariance, independent of the code."""
    active = data.counts.reshape(-1) > 0
    width = data.U.shape[-1]
    complex_design = _complex_design(data, design, channel).reshape(
        -1, width, len(scales)
    )[active]
    x = jnp.asarray(
        np.concatenate(
            (
                complex_design.real.reshape(-1, len(scales)),
                complex_design.imag.reshape(-1, len(scales)),
            )
        )
    )
    observed = data.U[..., channel, :].reshape(-1, width)[active]
    y = jnp.asarray(np.r_[observed.real.ravel(), observed.imag.ravel()])
    active_logs = jnp.asarray(logs).reshape(-1)[active]
    variance = jnp.tile(
        jnp.repeat(jnp.exp(active_logs) * data.enbw / 2, width), 2
    )
    covariance = jnp.diag(variance) + (x * scales**2) @ x.T
    normal = -0.5 * (
        len(y) * jnp.log(2 * jnp.pi)
        + jnp.linalg.slogdet(covariance)[1]
        + y @ jnp.linalg.solve(covariance, y)
    )
    # Factor width is not the count. Restore the Wishart logdet convention.
    correction = jnp.sum(
        (width - data.counts.reshape(-1)[active] / data.enbw) * active_logs
    ) + len(observed) * width * jnp.log(jnp.pi * data.enbw)
    return normal + correction


def _covariance(conditional):
    _, _, cholesky, diagonal = map(np.asarray, conditional)
    precision = diagonal[:, None] * (cholesky @ cholesky.T) * diagonal[None, :]
    return np.linalg.inv(precision)


@pytest.mark.parametrize("enbw", [1.0, 1.35])
def test_grid_integral_and_gradients_match_independent_dense_gaussian(enbw):
    data, _, config, pair, design = _fixture(enbw=enbw)
    prepared = prepare_grid_theta(data, pair, design, 2)
    parameters = jnp.r_[
        jnp.linspace(-0.2, 0.25, data.counts.size),
        jnp.log(jnp.array([0.8, 0.65, 1.2, 0.9])),
        jnp.log(jnp.array([0.25, 0.3, 0.35, 0.4])),
    ]

    def values(parameters):
        scales = grid_theta_scales(
            pair, jnp.exp(parameters[-8:-4]), jnp.exp(parameters[-4:])
        )
        return parameters[:-8], scales

    def actual(parameters):
        logs, scales = values(parameters)
        return grid_theta_conditional(logs, scales, prepared)[0]

    def expected(parameters):
        logs, scales = values(parameters)
        return _dense_marginal(logs, scales, data, design, 2)

    np.testing.assert_allclose(
        actual(parameters), expected(parameters), atol=3e-12, rtol=2e-13
    )
    np.testing.assert_allclose(
        jax.grad(actual)(parameters),
        jax.grad(expected)(parameters),
        atol=5e-12,
        rtol=3e-12,
    )
    # A masked cell's variance cannot affect the likelihood or the Gaussian integral.
    assert np.all(
        np.asarray(jax.grad(actual)(parameters))[:-8][data.counts.ravel() == 0]
        == 0
    )


@pytest.mark.parametrize("enbw", [1.0, 1.35])
@pytest.mark.parametrize("explicit_constant", [False, True])
@pytest.mark.parametrize("log_scale", [-100.0, 0.0, 100.0])
def test_constant_time_matches_stationary_integral_gradients_and_covariance(
    enbw, explicit_constant, log_scale
):
    data, spline, config, pair, design = _fixture(
        enbw=enbw,
        stationary=True,
        explicit_constant=explicit_constant,
        log_scale=log_scale,
    )
    assert not pair["time_active"]
    np.testing.assert_array_equal(design[0], np.ones((2, 1)))
    prepared = prepare_grid_theta(data, pair, design, 2)
    kf = prepared["field_size"]
    # Pool time by concatenating factors; convert to stationary duration units.
    # Varying counts, including masks, sum to four at every frequency.
    duration = 2.7
    u = np.transpose(data.U, (1, 2, 0, 3)).reshape(5, 3, -1) * np.sqrt(
        duration
    )
    basis = design[1]
    design_complex = np.concatenate(
        [
            u[:, previous, :, None] * basis[:, None, :] * multiplier
            for previous in range(2)
            for multiplier in (1, 1j)
        ],
        axis=-1,
    )
    penalty = block_diag(*([np.asarray(spline.frequency.penalty)] * 4))
    rotation = np.kron(np.eye(4), pair["U_freq"])
    parameters = jnp.r_[
        jnp.array([0.2, -0.1, 0.15]) + log_scale,
        jnp.log(jnp.array([0.8, 0.65, 1.2, 0.9])),
    ]

    def both(parameters):
        log_variance = jnp.asarray(basis) @ parameters[:kf]
        sigmas = jnp.exp(parameters[kf:])
        scales = grid_theta_scales(pair, sigmas)
        grid = grid_theta_conditional(
            jnp.broadcast_to(log_variance, data.counts.shape), scales, prepared
        )
        stationary = gaussian_theta_conditional(
            log_variance,
            sigmas,
            design_re=jnp.asarray(design_complex.real),
            design_im=jnp.asarray(design_complex.imag),
            observed_re=jnp.asarray(u[:, 2].real),
            observed_im=jnp.asarray(u[:, 2].imag),
            theta_sizes=(kf,) * 4,
            theta_penalty=jnp.asarray(penalty),
            theta_penalty_cholesky=jnp.asarray(np.linalg.cholesky(penalty)),
            theta_logdet_penalty=float(np.linalg.slogdet(penalty)[1]),
            Nb=2,
            Nh=2,
            duration=duration,
            enbw=enbw,
        )
        return grid, stationary

    grid, stationary = both(parameters)
    np.testing.assert_allclose(grid[0], stationary[0], atol=2e-12, rtol=2e-13)
    np.testing.assert_allclose(rotation @ grid[1], stationary[1], atol=3e-13)
    np.testing.assert_allclose(
        rotation @ _covariance(grid) @ rotation.T,
        _covariance(stationary),
        atol=4e-13,
    )
    gradient = jax.grad(lambda p: both(p)[0][0])(parameters)
    np.testing.assert_allclose(
        gradient,
        jax.grad(lambda p: both(p)[1][0])(parameters),
        atol=3e-12,
        rtol=2e-12,
    )
    # Changing physical units must not flatten the likelihood through clipping.
    assert np.linalg.norm(gradient[:kf]) > 0.1


@pytest.mark.parametrize("stationary", [False, True])
def test_tensor_precision_retains_supplied_frequency_ridge_and_null_modes(
    stationary,
):
    _, spline, config, pair, _ = _fixture(stationary=stationary)
    sf, st = jnp.array([0.8, 0.65]), jnp.array([0.25, 0.4])
    scales = grid_theta_scales(pair, sf, None if stationary else st)
    kt, kf = len(pair["lam_time"]), len(pair["lam_freq"])
    rotation = np.kron(pair["U_time"], pair["U_freq"])
    pt = np.zeros((1, 1)) if stationary else np.asarray(spline.time.penalty)
    pf = np.asarray(spline.frequency.penalty)
    for index in range(2):
        actual = (
            rotation
            @ np.diag(
                1
                / np.asarray(scales[index * kt * kf : (index + 1) * kt * kf])
                ** 2
            )
            @ rotation.T
        )
        expected = np.kron(np.eye(kt), pf) / sf[index] ** 2
        if not stationary:
            expected += np.kron(pt, np.eye(kf)) / st[index] ** 2
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-13)
    if stationary:
        # Bt=1 has no normalization by the number of time cells or time modes.
        expected = np.concatenate(
            [sf[index] / np.sqrt(pair["lam_freq"]) for index in range(2)]
        )
        np.testing.assert_allclose(scales, expected, atol=2e-15)


def test_rectangular_and_paired_integrals_preserve_masks_counts_and_enbw():
    data, _, config, pair, design = _fixture(enbw=1.35)
    q = data.counts.size
    paired = WishartGridData(
        data.u_re.reshape(q, 3, -1),
        data.u_im.reshape(q, 3, -1),
        np.repeat(data.time, len(data.frequency)),
        np.tile(data.frequency, len(data.time)),
        data.counts.ravel(),
        reference_time=data.time,
        reference_frequency=data.frequency,
        enbw=data.enbw,
    )
    paired_design = tuple(
        np.repeat(value, len(data.frequency), axis=0)
        if index % 2 == 0
        else np.tile(value, (len(data.time), 1))
        for index, value in enumerate(design)
    )
    scales = grid_theta_scales(
        pair,
        jnp.array([0.8, 0.65, 1.2, 0.9]),
        jnp.array([0.25, 0.3, 0.35, 0.4]),
    )
    logs = jnp.linspace(-0.2, 0.25, q)
    grid = grid_theta_conditional(
        logs, scales, prepare_grid_theta(data, pair, design, 2)
    )
    points = grid_theta_conditional(
        logs, scales, prepare_grid_theta(paired, pair, paired_design, 2)
    )
    for actual, expected in zip(grid, points, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-13)


@pytest.mark.parametrize("log_scale", [-100.0, 0.0, 100.0])
def test_recovered_joint_draws_have_correct_covariance_and_original_likelihood(
    log_scale,
):
    data, _, config, pair, design = _fixture(enbw=1.35, log_scale=log_scale)
    prepared = prepare_grid_theta(data, pair, design, 2)
    size = prepared["field_size"]
    rotation = np.kron(pair["U_time"], pair["U_freq"])
    delta = np.linspace(-0.2, 0.25, size) + log_scale * (
        rotation.T @ np.ones(size)
    )
    sigma_freq, sigma_time = (
        np.array([0.8, 0.65, 1.2, 0.9]),
        np.array([0.25, 0.3, 0.35, 0.4]),
    )
    scales = grid_theta_scales(pair, sigma_freq, sigma_time)
    logs = prepared["field_basis"] @ delta
    conditional = grid_theta_conditional(logs, scales, prepared)
    n_draws = 8000
    posterior = {
        "s_delta_2": np.broadcast_to(delta, (1, n_draws, size)),
    }
    for index, label in enumerate(prepared["theta_labels"]):
        posterior[f"sigma_freq_{label}"] = np.full(
            (1, n_draws), sigma_freq[index]
        )
        posterior[f"sigma_time_{label}"] = np.full(
            (1, n_draws), sigma_time[index]
        )
    recovered = recover_grid_theta(
        posterior, pair, config, prepared, jax.random.PRNGKey(128)
    )
    beta = np.concatenate(
        [recovered[f"s_{label}"][0] for label in prepared["theta_labels"]],
        axis=-1,
    )
    covariance = _covariance(conditional)
    whitened = np.linalg.solve(
        np.linalg.cholesky(covariance), (beta - np.asarray(conditional[1])).T
    ).T
    assert np.max(abs(whitened.mean(axis=0))) < 0.045
    np.testing.assert_allclose(
        np.cov(whitened.T), np.eye(len(covariance)), atol=0.075, rtol=0
    )
    # The covariance must be joint: independent draws for each field lose these terms.
    assert np.max(abs(covariance[:size, size : 2 * size])) > 1e-4
    for index in (0, 2, 5):
        theta = (
            np.asarray(prepared["field_basis"])
            @ beta[index].reshape(4, size).T
        ).reshape(2, 5, 4)
        expected = wishart_log_likelihood(
            logs.reshape(2, 5),
            theta[..., ::2],
            theta[..., 1::2],
            data.u_re[..., 2, :],
            data.u_im[..., 2, :],
            data.u_re[..., :2, :],
            data.u_im[..., :2, :],
            counts=data.counts,
            enbw=data.enbw,
        )
        residual = data.U[..., 2, :] - np.einsum(
            "tfj,tfjr->tfr",
            theta[..., ::2] + 1j * theta[..., 1::2],
            data.U[..., :2, :],
        )
        direct = (
            -np.sum(
                data.counts * logs.reshape(2, 5)
                + np.sum(abs(residual) ** 2, axis=-1)
                * np.exp(-logs.reshape(2, 5))
            )
            / data.enbw
        )
        np.testing.assert_allclose(expected, direct, atol=2e-12)
        np.testing.assert_allclose(
            recovered["log_likelihood_block_2"][0, index], expected, atol=2e-12
        )
    # Conditional recovery also exposes standardized theta coefficients.
    for field, label in enumerate(prepared["theta_labels"]):
        z = recovered[f"z_{label}"][0]
        np.testing.assert_allclose(
            z,
            beta[:, field * size : (field + 1) * size]
            / np.asarray(scales)[field * size : (field + 1) * size],
            atol=2e-13,
        )
