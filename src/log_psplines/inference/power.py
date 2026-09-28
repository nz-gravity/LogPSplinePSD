"""Scalar power/count inference using the WDM tensor prior.

The transform stays in preprocessing; the model evaluates a LogPSpline.
No WDM package is required for fitting already prepared powers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from jaxtyping import Float

from log_psplines.basis.penalty import eigen_prior_scale, whiten_penalty_pair
from log_psplines.config import PowerConfig
from log_psplines.data.spectral import PowerData
from log_psplines.inference.nuts import run_nuts
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.models.anova import ANOVALogPSpline
from log_psplines.models.reconstruction import reconstruct_power_spectrum
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


def _sample_precision(name: str, config: PowerConfig) -> jax.Array | float:
    """Sample a HalfNormal roughness scale and derive its precision."""
    sigma = numpyro.sample(name, dist.HalfNormal(config.roughness_scale))
    return sigma**-2


def sample_eigen_coefficients(
    name: str,
    scale: jnp.ndarray,
    shape: tuple[int, ...],
    config: PowerConfig,
) -> jax.Array | np.ndarray:
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
    observed_power: Float[np.ndarray, "N_t N_f"],
    B_time: Float[np.ndarray, "N_t K_t"],
    B_freq: Float[np.ndarray, "N_f K_f"],
    penalty_time: Float[np.ndarray, "K_t K_t"],
    penalty_freq: Float[np.ndarray, "K_f K_f"],
    config: PowerConfig,
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
    config: PowerConfig,
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
        "sigma_time": float(phi_time**-0.5),
        "sigma_freq": float(phi_freq**-0.5),
    }


def _mean_power_for_masked_initialization(
    summed_power: Float[np.ndarray, "N_t N_f"],
    counts: Float[np.ndarray, "N_t N_f"],
) -> Float[np.ndarray, "N_t N_f"]:
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
    observed_power: Float[np.ndarray, "Q"],
    B_time: Float[np.ndarray, "Q K_t"],
    B_freq: Float[np.ndarray, "Q K_f"],
    penalty_time: Float[np.ndarray, "K_t K_t"],
    penalty_freq: Float[np.ndarray, "K_f K_f"],
    config: PowerConfig,
) -> dict[str, np.ndarray | float]:
    """Penalized least-squares warm start for scattered (u, omega) ordinates.

    Unlike :func:`initialize_with_penalized_least_squares`, ``B_time`` and
    ``B_freq`` are evaluated per-ordinate (``(Q, K_t)``/``(Q, K_f)``), not on a
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
    data: PowerData,
    spline: LogPSpline,
    config: PowerConfig,
) -> tuple[Callable, dict, dict[str, np.ndarray]]:
    """Prepare the power likelihood and PLS sites on either geometry."""
    if spline.time is None or data.time is None:
        raise ValueError("power fitting currently requires a time basis")
    if data.is_grid and (
        spline.time.basis.shape[0] != len(data.time)
        or spline.n != len(data.frequency)
    ):
        raise ValueError("spline and power grids must have matching shapes")

    pair = whiten_penalty_pair(spline.time.penalty, spline.frequency.penalty)
    lam_t, lam_f = jnp.asarray(pair["lam_time"]), jnp.asarray(pair["lam_freq"])
    null = jnp.asarray(pair["joint_null"])
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    if data.is_grid:
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
        evaluate = eigen_spline
    else:
        bt_raw = np.asarray(spline.time.design_at(data.time))
        bf_raw = np.asarray(spline.frequency.design_at(data.frequency))
        bt_eigen = jnp.asarray(bt_raw @ pair["U_time"])
        bf_eigen = jnp.asarray(bf_raw @ pair["U_freq"])

        def evaluate(coefficients):
            return jnp.einsum(
                "pi,ij,pj->p",
                bt_eigen,
                coefficients,
                bf_eigen,
                optimize="optimal",
            )

    def model() -> None:
        phi_time = _sample_precision("sigma_time", config)
        phi_freq = _sample_precision("sigma_freq", config)
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
            power, counts, evaluate(coefficients)
        )
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.factor("whittle", log_like)

    mean_power = np.divide(
        data.power,
        data.counts,
        out=np.zeros_like(data.power),
        where=data.counts > 0,
    )
    if data.is_grid:
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
    else:
        pls = initialize_scattered_with_penalized_least_squares(
            mean_power,
            bt_raw,
            bf_raw,
            np.asarray(spline.time.penalty),
            np.asarray(spline.frequency.penalty),
            config,
        )
    return model, whitened_init_values(pls, pair, config), pair


def _run_power_nuts(model: Callable, init: dict, config: PowerConfig):
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
    result, pair: dict[str, np.ndarray], config: PowerConfig
):
    """Rotate sampled eigen-coefficients to spline weights."""
    import xarray as xr

    posterior = result.posterior.copy()
    samples = {
        name: np.asarray(var.values)
        for name, var in posterior.data_vars.items()
    }
    coefficients = samples["s"].reshape(
        *samples["s"].shape[:2], len(pair["lam_time"]), len(pair["lam_freq"])
    )
    if not config.centered:
        scale = jax.vmap(
            jax.vmap(
                lambda pt, pf: eigen_prior_scale(
                    pt**-2,
                    pf**-2,
                    jnp.asarray(pair["lam_time"]),
                    jnp.asarray(pair["lam_freq"]),
                    jnp.asarray(pair["joint_null"]),
                    null_precision=config.null_precision,
                    ridge_eps=config.ridge_eps,
                )
            )
        )(samples["sigma_time"], samples["sigma_freq"])
        coefficients = coefficients * np.asarray(scale)

    weights = np.einsum(
        "ia,cdab,jb->cdij",
        pair["U_time"],
        coefficients,
        pair["U_freq"],
        optimize=True,
    )
    posterior["weights"] = xr.DataArray(
        weights,
        dims=(
            "chain",
            "draw",
            "time_coefficient",
            "frequency_coefficient",
        ),
    )
    return posterior


def fit_power(
    data: PowerData,
    spline: LogPSpline | ANOVALogPSpline,
    config: PowerConfig,
    *,
    partition=None,
    reference=None,
    true_psd=None,
) -> PSDResult:
    """Fit grid or scattered powers using the same likelihood and prior."""
    from log_psplines.preprocessing.power_partition import coarse_grain_power
    from log_psplines.results import PSDResult, observed_power_data

    anova = isinstance(spline, ANOVALogPSpline)
    if reference is not None and not data.is_grid:
        raise ValueError("reference requires rectangular PowerData")
    if true_psd is not None and not data.is_grid:
        raise ValueError("true_psd requires rectangular PowerData")
    if anova and not data.is_grid:
        raise ValueError(
            "ANOVALogPSpline requires rectangular GridTV PowerData"
        )
    if anova and (
        len(spline.time.grid) != len(data.time)
        or len(spline.frequency.grid) != len(data.frequency)
        or not np.array_equal(spline.time.grid, data.time)
        or not np.array_equal(spline.frequency.grid, data.frequency)
    ):
        raise ValueError("ANOVA model must use the native PowerData grid")
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if (
            reference.shape != data.power.shape
            or not np.isfinite(reference).all()
            or np.any(reference <= 0)
        ):
            raise ValueError(
                "reference must be finite, positive and match native power"
            )
        # Native-cell division precedes pooling. The sum of log R is data-only.
        data_for_fit = PowerData(
            data.power / reference,
            data.counts,
            data.frequency,
            data.time,
            f"reference-normalized {data.units}",
        )
    else:
        data_for_fit = data
    if true_psd is not None:
        true_psd = np.asarray(true_psd, dtype=float)
        if (
            true_psd.shape != data.power.shape
            or not np.isfinite(true_psd).all()
            or np.any(true_psd <= 0)
        ):
            raise ValueError(
                "true_psd must be finite, positive and match native power"
            )

    if partition is not None and not data.is_grid:
        raise ValueError("partition requires rectangular PowerData")
    fit_data = data_for_fit
    fit_spline = spline
    if partition is not None:
        if (
            spline.time is None
            or data.time is None
            or len(spline.time.grid) != len(data.time)
            or len(spline.frequency.grid) != len(data.frequency)
        ):
            raise ValueError("partitioned fits require a native-grid spline")
        fit_data = coarse_grain_power(data_for_fit, partition)
        ts = np.asarray(partition.time_starts)
        fs = np.asarray(partition.frequency_starts)
        model_time = np.add.reduceat(spline.time.grid, ts) / np.diff(
            np.r_[ts, len(data.time)]
        )
        model_frequency = np.add.reduceat(spline.frequency.grid, fs) / np.diff(
            np.r_[fs, len(data.frequency)]
        )
        if not anova:
            fit_spline = LogPSpline(
                frequency=replace(
                    spline.frequency,
                    grid=model_frequency,
                    basis=spline.frequency.design_at(model_frequency),
                ),
                time=replace(
                    spline.time,
                    grid=model_time,
                    basis=spline.time.design_at(model_time),
                ),
            )
    if anova:
        from log_psplines.inference.anova_power import (
            collect_anova_samples,
            prepare_anova_power_model,
        )

        model, pair, init = prepare_anova_power_model(fit_data, spline, config)
    else:
        model, init, pair = prepare_power_model(fit_data, fit_spline, config)
    result = _run_power_nuts(model, init, config)
    posterior = (
        collect_anova_samples(result.posterior, pair)
        if anova
        else _collect_power_samples(result, pair, config)
    )
    fitted = PSDResult(
        posterior=posterior,
        sample_stats=result.sample_stats,
        spectrum=reconstruct_power_spectrum(
            posterior,
            spline,
            time=data.time if data.is_grid else None,
            frequency=data.frequency if data.is_grid else None,
            reference=reference,
        ),
        metadata={
            **asdict(config),
            "data_type": "power",
            "likelihood": "power_whittle",
            "units": data.units,
            "reference_applied": reference is not None,
            "reference_normalization": (
                "native_power_divided_before_pooling"
                if reference is not None
                else None
            ),
            **(
                {
                    "model": "anova",
                    "centered": True,
                    "sigma_eta_prior": spline.sigma_eta_prior,
                }
                if anova
                else {}
            ),
        },
        log_likelihood=result.log_likelihood,
        observed_data=observed_power_data(fit_data),
        truth=(
            None
            if true_psd is None
            else xr.DataArray(
                true_psd,
                dims=("time", "frequency"),
                coords={"time": data.time, "frequency": data.frequency},
                name="true_psd",
            )
        ),
    )
    if partition is not None:
        fitted.metadata["partition_time_starts"] = np.asarray(
            partition.time_starts
        )
        fitted.metadata["partition_frequency_starts"] = np.asarray(
            partition.frequency_starts
        )
    return fitted
