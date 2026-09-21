"""Scalar power/count inference using the WDM tensor prior.

The transform stays in preprocessing; the model evaluates a LogPSpline.
No WDM package is required for fitting already prepared powers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from log_psplines.basis.penalty import eigen_prior_scale, whiten_penalty_pair
from log_psplines.config import PowerSplineConfig
from log_psplines.data.spectral import PowerSpectrum, ScatteredPowerSpectrum
from log_psplines.inference.nuts import run_nuts
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.models.spectrum import LogPSpline

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def power_floor(power: np.ndarray) -> float:
    """Scale-free floor for ``log(power + floor)`` targets.

    A small fraction of a low percentile of the *nonzero* power, so the floor
    tracks the data scale (absolute floors like 1e-8 swamp tiny-amplitude data,
    e.g. LISA fractional-frequency series with power ~1e-40).
    """
    positive = power[power > 0]
    if positive.size == 0:
        return 1.0
    return 0.05 * float(np.percentile(positive, 10.0))


def _sample_log_gamma(
    name: str,
    alpha: float,
    beta: float,
    base_scale: float,
) -> jnp.ndarray:
    """Sample ``phi`` with a ``Gamma(alpha, beta)`` prior on the log scale.

    The site itself is ``log phi`` with a broad Normal reference measure; a
    ``factor`` corrects the density to the exact ``Gamma`` prior (with the
    log-Jacobian), giving an unconstrained, well-scaled sampling variable. This
    mirrors the approach used in ``log_psplines``.
    """
    base = dist.Normal(0.0, base_scale)
    log_phi = numpyro.sample(name, base)
    phi = jnp.exp(log_phi)
    gamma = dist.Gamma(alpha, beta)
    numpyro.factor(
        f"{name}_prior",
        gamma.log_prob(phi) + log_phi - base.log_prob(log_phi),
    )
    return phi


def sample_eigen_coefficients(
    name: str,
    scale: jnp.ndarray,
    shape: tuple[int, ...],
    config: PowerSplineConfig,
) -> jnp.ndarray:
    """Sample eigen-coefficients while honoring ``config.centered``.

    In centered form the sampling site is the coefficient itself and has prior
    ``Normal(0, scale)``. In non-centered form the site is standard Normal and
    is multiplied by ``scale`` before being returned.
    """
    n_weights = int(np.prod(shape))
    flat_scale = jnp.broadcast_to(scale, shape).reshape(-1)
    with numpyro.plate(f"{name}_plate", n_weights):
        if config.centered:
            site = numpyro.sample(name, dist.Normal(0.0, flat_scale))
            coeffs = site
        else:
            site = numpyro.sample(name, dist.Normal(0.0, 1.0))
            coeffs = site * flat_scale
    return coeffs.reshape(shape)


def initialize_with_penalized_least_squares(
    observed_power: np.ndarray,
    B_time: np.ndarray,
    B_freq: np.ndarray,
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    config: PowerSplineConfig,
) -> dict[str, np.ndarray | float]:
    """Penalized least-squares warm start in the *coefficient* basis.

    Returns the fitted coefficient matrix ``W`` and heuristic smoothing
    precisions; :func:`whitened_init_values` converts these to the model sites.
    """
    floor = power_floor(observed_power)
    target = np.log(observed_power + floor)
    kron_time = np.kron(np.eye(B_freq.shape[1]), penalty_time)
    kron_freq = np.kron(penalty_freq, np.eye(B_time.shape[1]))
    # The design is kron(B_freq, B_time), so its Gram is the Kronecker of the
    # per-axis Grams and the projection is B_t^T Y B_f: never form the
    # O(n_time*n_freq x n_basis) design matrix.
    n_basis = B_time.shape[1] * B_freq.shape[1]
    system = (
        np.kron(B_freq.T @ B_freq, B_time.T @ B_time)
        + config.init_penalty_time * kron_time
        + config.init_penalty_freq * kron_freq
        + config.ridge_eps * np.eye(n_basis)
    )
    rhs = (B_time.T @ target @ B_freq).reshape(-1, order="F")
    weights = np.linalg.solve(system, rhs)
    W_fit = weights.reshape((B_time.shape[1], B_freq.shape[1]), order="F")
    fitted = B_time @ W_fit @ B_freq.T

    penalty_time_energy = float(weights @ kron_time @ weights)
    penalty_freq_energy = float(weights @ kron_freq @ weights)
    phi_time_init = max(1e-2, fitted.size / (penalty_time_energy + 1e-6))
    phi_freq_init = max(1e-2, fitted.size / (penalty_freq_energy + 1e-6))

    return {
        "W": W_fit,
        "phi_time": phi_time_init,
        "phi_freq": phi_freq_init,
        "log_psd": fitted,
    }


def whitened_init_values(
    pls_init: dict[str, np.ndarray | float],
    whitened: dict[str, np.ndarray],
    config: PowerSplineConfig,
) -> dict[str, np.ndarray]:
    """Map least-squares coefficients to the whitened sampling sites."""
    U_t = whitened["U_time"]
    U_f = whitened["U_freq"]
    lam_t = whitened["lam_time"]
    lam_f = whitened["lam_freq"]
    joint_null = whitened["joint_null"]

    phi_time = float(pls_init["phi_time"])
    phi_freq = float(pls_init["phi_freq"])
    eig_coeffs = U_t.T @ np.asarray(pls_init["W"]) @ U_f  # Z in the eigenbasis

    if config.centered:
        s = eig_coeffs
    else:
        d = phi_time * lam_t[:, None] + phi_freq * lam_f[None, :]
        inv_scale = np.where(
            joint_null,
            np.sqrt(config.null_precision),
            np.sqrt(d + config.ridge_eps),
        )
        s = eig_coeffs * inv_scale
    return {
        "s": s.reshape(-1),
        "phi_time": float(np.log(phi_time)),
        "phi_freq": float(np.log(phi_freq)),
    }


def _mean_power_for_masked_initialization(
    summed_power: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Fill masked cells for initialization without changing the target.

    Retained cells use their per-component mean power. Missing cells are filled
    by log-linear frequency interpolation within each likelihood row. A fully
    masked row receives the global retained-cell median. These values seed NUTS
    only; masked cells still have zero power and zero count in the likelihood.
    """
    summed_power = np.asarray(summed_power, dtype=float)
    counts = np.broadcast_to(
        np.asarray(counts, dtype=float), summed_power.shape
    )
    valid = (counts > 0.0) & (summed_power > 0.0)
    retained = summed_power[valid] / counts[valid]
    global_fill = float(np.median(retained)) if retained.size else 1.0
    output = np.empty_like(summed_power)
    frequency_index = np.arange(summed_power.shape[1], dtype=float)
    for row in range(summed_power.shape[0]):
        row_valid = valid[row]
        if np.any(row_valid):
            locations = frequency_index[row_valid]
            values = np.log(
                summed_power[row, row_valid] / counts[row, row_valid]
            )
            output[row] = np.exp(np.interp(frequency_index, locations, values))
        else:
            output[row] = global_fill
    return output


def initialize_scattered_with_penalized_least_squares(
    observed_power: np.ndarray,
    B_time: np.ndarray,
    B_freq: np.ndarray,
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    config: PowerSplineConfig,
) -> dict[str, np.ndarray | float]:
    """Penalized least-squares warm start for scattered (u, omega) ordinates.

    Unlike :func:`initialize_with_penalized_least_squares`, ``B_time`` and
    ``B_freq`` are evaluated per-ordinate (``(P, Kt)``/``(P, Kf)``), not on a
    shared grid, so the design cannot be factored as a Kronecker product of
    marginal Grams and is built explicitly instead.
    """
    floor = power_floor(observed_power)
    target = np.log(observed_power + floor)
    n_time, n_freq = B_time.shape[1], B_freq.shape[1]
    n_basis = n_time * n_freq
    design = np.einsum("pt,pf->ptf", B_time, B_freq).reshape(-1, n_basis)
    kron_time = np.kron(penalty_time, np.eye(n_freq))
    kron_freq = np.kron(np.eye(n_time), penalty_freq)
    system = (
        design.T @ design
        + config.init_penalty_time * kron_time
        + config.init_penalty_freq * kron_freq
        + config.ridge_eps * np.eye(n_basis)
    )
    rhs = design.T @ target
    weights = np.linalg.solve(system, rhs)
    W_fit = weights.reshape(n_time, n_freq)
    fitted = np.einsum("pt,tf,pf->p", B_time, W_fit, B_freq)

    penalty_time_energy = float(weights @ kron_time @ weights)
    penalty_freq_energy = float(weights @ kron_freq @ weights)
    phi_time_init = max(1e-2, fitted.size / (penalty_time_energy + 1e-6))
    phi_freq_init = max(1e-2, fitted.size / (penalty_freq_energy + 1e-6))

    return {
        "W": W_fit,
        "phi_time": phi_time_init,
        "phi_freq": phi_freq_init,
        "log_psd": fitted,
    }


def prepare_power_model(
    data: PowerSpectrum,
    spline: LogPSpline,
    config: PowerSplineConfig,
) -> tuple[Callable, dict, dict[str, np.ndarray]]:
    """Build the model and WDM least-squares initial sites on matched grids."""
    if spline.time is None or data.time is None:
        raise ValueError(
            "power fitting currently requires a time basis and grid"
        )
    if spline.time.basis.shape[0] != len(data.time) or spline.n != len(
        data.frequency
    ):
        raise ValueError("spline and power grids must have matching shapes")
    pair = whiten_penalty_pair(spline.time.penalty, spline.frequency.penalty)
    # Evaluate through the same scalar model in the eigenbasis. The original
    # basis and penalties are retained for initialization and result storage.
    eigen_spline = LogPSpline(
        frequency=replace(
            spline.frequency,
            basis=jnp.asarray(spline.frequency.basis @ pair["U_freq"]),
            penalty=np.diag(pair["lam_freq"]),
        ),
        time=replace(
            spline.time,
            basis=jnp.asarray(spline.time.basis @ pair["U_time"]),
            penalty=np.diag(pair["lam_time"]),
        ),
    )
    lam_t, lam_f = jnp.asarray(pair["lam_time"]), jnp.asarray(pair["lam_freq"])
    null = jnp.asarray(pair["joint_null"])
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    def model() -> None:
        phi_time = _sample_log_gamma(
            "phi_time",
            config.alpha_phi,
            config.beta_phi,
            config.phi_log_base_scale,
        )
        phi_freq = _sample_log_gamma(
            "phi_freq",
            config.alpha_phi,
            config.beta_phi,
            config.phi_log_base_scale,
        )
        scale = eigen_prior_scale(
            phi_time,
            phi_freq,
            lam_t,
            lam_f,
            null,
            null_precision=config.null_precision,
            ridge_eps=config.ridge_eps,
        )
        coefficients = sample_eigen_coefficients(
            "s", scale, scale.shape, config
        )
        log_like = power_whittle_log_likelihood(
            power, counts, eigen_spline(coefficients)
        )
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.factor("whittle", log_like)

    mean_power = np.divide(
        data.power,
        data.counts,
        out=np.zeros_like(data.power),
        where=data.counts > 0,
    )
    if np.any(data.counts == 0):
        mean_power = _mean_power_for_masked_initialization(
            data.power, data.counts
        )
    pls = initialize_with_penalized_least_squares(
        mean_power,
        np.asarray(spline.time.basis),
        np.asarray(spline.basis),
        np.asarray(spline.time.penalty),
        np.asarray(spline.frequency.penalty),
        config,
    )
    return model, whitened_init_values(pls, pair, config), pair


def prepare_scattered_power_model(
    data: ScatteredPowerSpectrum,
    spline: LogPSpline,
    config: PowerSplineConfig,
) -> tuple[Callable, dict, dict[str, np.ndarray]]:
    """Build the model and PLS initial sites for scattered (u, omega) ordinates.

    Unlike :func:`prepare_power_model`, the spline's ``time``/``frequency``
    grids need not match the data: each ordinate is evaluated at its own
    exact ``(time, frequency)`` coordinate via ``SplineBasis.design_at``,
    so ``log S(u_i, omega_i)`` is used directly rather than a rectangular
    ``log S(t, f)`` surface.
    """
    if spline.time is None:
        raise ValueError(
            "scattered power fitting currently requires a time basis"
        )
    pair = whiten_penalty_pair(spline.time.penalty, spline.frequency.penalty)
    Bt_raw = np.asarray(spline.time.design_at(data.time))
    Bf_raw = np.asarray(spline.frequency.design_at(data.frequency))
    Bt_eigen = jnp.asarray(Bt_raw @ pair["U_time"])
    Bf_eigen = jnp.asarray(Bf_raw @ pair["U_freq"])
    lam_t, lam_f = jnp.asarray(pair["lam_time"]), jnp.asarray(pair["lam_freq"])
    null = jnp.asarray(pair["joint_null"])
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    def model() -> None:
        phi_time = _sample_log_gamma(
            "phi_time",
            config.alpha_phi,
            config.beta_phi,
            config.phi_log_base_scale,
        )
        phi_freq = _sample_log_gamma(
            "phi_freq",
            config.alpha_phi,
            config.beta_phi,
            config.phi_log_base_scale,
        )
        scale = eigen_prior_scale(
            phi_time,
            phi_freq,
            lam_t,
            lam_f,
            null,
            null_precision=config.null_precision,
            ridge_eps=config.ridge_eps,
        )
        coefficients = sample_eigen_coefficients(
            "s", scale, scale.shape, config
        )
        log_psd = jnp.einsum(
            "pi,ij,pj->p",
            Bt_eigen,
            coefficients,
            Bf_eigen,
            optimize="optimal",
        )
        log_like = power_whittle_log_likelihood(power, counts, log_psd)
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.factor("whittle", log_like)

    mean_power = np.divide(
        data.power,
        data.counts,
        out=np.zeros_like(data.power),
        where=data.counts > 0,
    )
    pls = initialize_scattered_with_penalized_least_squares(
        mean_power,
        Bt_raw,
        Bf_raw,
        np.asarray(spline.time.penalty),
        np.asarray(spline.frequency.penalty),
        config,
    )
    return model, whitened_init_values(pls, pair, config), pair


def _run_power_nuts(
    model: Callable, init: dict, config: PowerSplineConfig
):
    return run_nuts(
        model,
        rng_key=jax.random.PRNGKey(config.seed),
        init_values=init,
        n_warmup=config.n_warmup,
        n_samples=config.n_samples,
        num_chains=config.num_chains,
        chain_method="sequential",
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        progress_bar=config.progress_bar,
        extra_fields=(
            "diverging",
            "accept_prob",
            "num_steps",
            "potential_energy",
            "energy",
        ),
    )


def _collect_power_samples(
    mcmc, pair: dict[str, np.ndarray], config: PowerSplineConfig
) -> dict[str, np.ndarray]:
    """Reshape NUTS draws into eigen-coefficients and rotate to ``weights``."""
    samples = {
        key: np.asarray(value)
        for key, value in mcmc.get_samples(group_by_chain=True).items()
    }
    coefficients = samples["s"].reshape(
        *samples["s"].shape[:2], len(pair["lam_time"]), len(pair["lam_freq"])
    )
    if not config.centered:
        scale = jax.vmap(
            jax.vmap(
                lambda pt, pf: eigen_prior_scale(
                    jnp.exp(pt),
                    jnp.exp(pf),
                    jnp.asarray(pair["lam_time"]),
                    jnp.asarray(pair["lam_freq"]),
                    jnp.asarray(pair["joint_null"]),
                    null_precision=config.null_precision,
                    ridge_eps=config.ridge_eps,
                )
            )
        )(samples["phi_time"], samples["phi_freq"])
        coefficients = coefficients * np.asarray(scale)
    samples["weights"] = np.einsum(
        "ia,cdab,jb->cdij",
        pair["U_time"],
        coefficients,
        pair["U_freq"],
        optimize=True,
    )
    return samples


def fit_power_spline(
    data: PowerSpectrum,
    spline: LogPSpline,
    config: PowerSplineConfig,
) -> PSDResult:
    """Run NUTS and return the common PSDResult with compact coefficients."""
    from log_psplines.arviz_utils.to_arviz import pack_power_result
    from log_psplines.results import PSDResult

    model, init, pair = prepare_power_model(data, spline, config)
    mcmc = _run_power_nuts(model, init, config)
    samples = _collect_power_samples(mcmc, pair, config)
    return PSDResult(
        pack_power_result(
            data,
            spline,
            config,
            samples,
            mcmc.get_extra_fields(group_by_chain=True),
        )
    )


def fit_scattered_power_spline(
    data: ScatteredPowerSpectrum,
    spline: LogPSpline,
    config: PowerSplineConfig,
) -> PSDResult:
    """Run NUTS on scattered (u, omega) ordinates; no pooled grid is built."""
    from log_psplines.arviz_utils.to_arviz import (
        pack_scattered_power_result,
    )
    from log_psplines.results import PSDResult

    model, init, pair = prepare_scattered_power_model(data, spline, config)
    mcmc = _run_power_nuts(model, init, config)
    samples = _collect_power_samples(mcmc, pair, config)
    return PSDResult(
        pack_scattered_power_result(
            data,
            spline,
            config,
            samples,
            mcmc.get_extra_fields(group_by_chain=True),
        )
    )

