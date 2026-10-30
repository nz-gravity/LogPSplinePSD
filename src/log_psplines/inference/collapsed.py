"""Exact Gaussian integration of stationary Cholesky regression coefficients.

Conditional on a row's log innovation spectrum and smoothing scales, its
complex regression coefficients form a Gaussian linear model. Integrating
those coefficients reduces the state for NUTS and VI without changing the
posterior target.
Independent conditional draws recover the ordinary coefficient posterior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular
from scipy.linalg import block_diag

from log_psplines.inference import banded
from log_psplines.inference.model import _sample_pspline_block
from log_psplines.likelihoods.whittle import LOG_SPECTRUM_LIMIT

_DESIGN_CACHE_BYTES = 32 * 1024**2


def _prepare_gram_cache(
    design_re: np.ndarray,
    design_im: np.ndarray,
    *,
    max_bytes: int = _DESIGN_CACHE_BYTES,
    banded_plan: banded.BandedPlan | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Cache exact, nonzero upper-triangle cross-products within a budget.

    B-splines have local support: only a few coefficients in each component
    are active at any frequency. Store their cross-products as a sparse map
    from frequency precisions to the flattened Gram matrix. No thresholding
    or spectral approximation is used. Designs exceeding the cache budget
    fall back to the original contractions rather than an unbounded cache.
    """
    n_frequency, _, n_weights = design_re.shape
    active = np.any((design_re != 0) | (design_im != 0), axis=1)
    counts = active.sum(axis=1, dtype=np.int64)
    n_entries = int(np.sum(counts * (counts + 1) // 2))
    # Two int32 index arrays and one float64 value array. Check before
    # allocating cross-products; never materialize (frequency, K, K).
    if n_weights**2 >= 2**31 or n_frequency >= 2**31:
        return None
    if n_entries * 16 > max_bytes:
        return None
    indices = np.empty(n_entries, dtype=np.int32)
    frequencies = np.empty(n_entries, dtype=np.int32)
    values = np.empty(n_entries, dtype=np.float64)
    offset = 0
    for f in range(n_frequency):
        columns = np.flatnonzero(active[f])
        re = design_re[f][:, columns]
        im = design_im[f][:, columns]
        gram = re.T @ re + im.T @ im
        row, col = np.triu_indices(columns.size)
        end = offset + row.size
        indices[offset:end] = columns[row] * n_weights + columns[col]
        frequencies[offset:end] = f
        values[offset:end] = gram[row, col]
        offset = end
    nonzero = values != 0
    indices, frequencies, values = (
        array[nonzero] for array in (indices, frequencies, values)
    )
    if banded_plan is not None:
        inverse = np.asarray(banded_plan.inverse_permutation)
        width = banded_plan.penalty.shape[1] - 1
        row, col = inverse[indices // n_weights], inverse[indices % n_weights]
        offset = np.abs(row - col)
        if np.any(offset > width):
            raise ValueError("banded plan does not contain all Gram entries")
        indices = (np.minimum(row, col) * (width + 1) + offset).astype(
            np.int32
        )
    pairs, inverse = np.unique(indices, return_inverse=True)
    dense_bytes = pairs.size * n_frequency * 8 + pairs.nbytes
    # A spline pair is active in only a local frequency interval. Batched
    # products on at most 16 contiguous blocks avoid multiplying the mostly
    # zero global map. Padding contains exact zeros, never truncated values.
    block_size = max(32, (n_frequency + 15) // 16)
    n_blocks = (n_frequency + block_size - 1) // block_size
    if n_blocks > 1 and pairs.size:
        blocks = frequencies // block_size
        block_pairs = [
            np.unique(indices[blocks == block], return_inverse=True)
            for block in range(n_blocks)
        ]
        width = max(pair.size for pair, _ in block_pairs)
        blocked_bytes = n_blocks * width * (block_size * 8 + 4)
        if blocked_bytes < dense_bytes and blocked_bytes <= max_bytes:
            by_block = np.zeros(
                (n_blocks, width, block_size), dtype=np.float64
            )
            block_indices = np.zeros((n_blocks, width), dtype=np.int32)
            for block, (pair, positions) in enumerate(block_pairs):
                selected = blocks == block
                by_block[
                    block, positions, frequencies[selected] % block_size
                ] = values[selected]
                block_indices[block, : pair.size] = pair
            return (
                jnp.asarray(block_indices),
                jnp.empty(0, dtype=jnp.int32),
                jnp.asarray(by_block),
            )
    # Dense designs and short grids need no frequency blocking. Prefer a
    # compact matrix-vector map to many repeated sparse scatter updates.
    if dense_bytes <= max_bytes:
        by_pair = np.zeros((pairs.size, n_frequency), dtype=np.float64)
        by_pair[inverse, frequencies] = values
        return (
            jnp.asarray(pairs),
            jnp.empty(0, dtype=jnp.int32),
            jnp.asarray(by_pair),
        )
    order = np.argsort(indices, kind="stable")
    return (
        jnp.asarray(indices[order]),
        jnp.asarray(frequencies[order]),
        jnp.asarray(values[order]),
    )


def _weighted_gram_cache(cache, frequency_precision, size):
    """Apply a fixed Gram map to precisions in dense-upper or band storage."""
    indices, frequencies, values = cache
    if values.ndim == 3:
        n_blocks, _, block_size = values.shape
        padded = jnp.pad(
            frequency_precision,
            (0, n_blocks * block_size - frequency_precision.size),
        )
        products = jnp.einsum(
            "bpf,bf->bp", values, padded.reshape(n_blocks, block_size)
        )
        return (
            jnp.zeros(size, dtype=values.dtype)
            .at[indices.ravel()]
            .add(products.ravel())
        )
    if values.ndim == 2:
        return (
            jnp.zeros(size, dtype=values.dtype)
            .at[indices]
            .set(values @ frequency_precision)
        )
    return jax.ops.segment_sum(
        values * frequency_precision[frequencies],
        indices,
        num_segments=size,
        indices_are_sorted=True,
    )


def _prepare_linear_cache(
    design_re, design_im, observed_re, observed_im, *, max_bytes
):
    """Cache the fixed design-response product, retaining a bounded fallback."""
    if design_re.shape[0] * design_re.shape[-1] * 8 > max_bytes:
        return None
    return jnp.asarray(
        2
        * (
            np.einsum("frk,fr->kf", design_re, observed_re)
            + np.einsum("frk,fr->kf", design_im, observed_im)
        )
    )


@jax.custom_jvp
def _logdet_at_precision(precision, cholesky, scale):
    """Log determinant from the already computed, diagonally balanced factor.

    The factor and scales must describe this precision. The derivative is
    tr(H^-1 dH), so differentiating through the balancing and Cholesky steps
    separately is unnecessary. Higher derivatives still follow the factor's
    dependence on H through the covariance used in the JVP.
    """
    return 2 * (jnp.sum(jnp.log(scale)) + jnp.sum(jnp.log(jnp.diag(cholesky))))


@_logdet_at_precision.defjvp
def _logdet_at_precision_jvp(primals, tangents):
    _, cholesky, scale = primals
    precision_tangent, _, _ = tangents
    inverse_factor = solve_triangular(
        cholesky, jnp.diag(1 / scale), lower=True
    )
    covariance = inverse_factor.T @ inverse_factor
    return _logdet_at_precision(*primals), jnp.sum(
        covariance * precision_tangent
    )


def _conditional_quadratic(
    mean,
    expanded_sigma,
    penalty_cholesky,
    design_re,
    design_im,
    observed_re,
    observed_im,
    frequency_precision,
    penalty_columns=None,
):
    residual_re = observed_re - jnp.einsum("frk,k->fr", design_re, mean)
    residual_im = observed_im - jnp.einsum("frk,k->fr", design_im, mean)
    scaled_mean = mean / expanded_sigma
    if penalty_columns is None:
        prior_residual = penalty_cholesky.T @ scaled_mean
    else:
        prior_residual = jnp.sum(
            penalty_cholesky * scaled_mean[penalty_columns], axis=1
        )
    return jnp.sum(
        (residual_re**2 + residual_im**2) * frequency_precision[:, None]
    ) + 0.5 * jnp.sum(prior_residual**2)


@jax.custom_jvp
def _quadratic_at_conditional_mean(*args):
    """Evaluate the stable quadratic at its solved minimizer.

    The envelope theorem removes the derivative through the mean solve:
    the derivative of this quadratic with respect to its mean is zero at
    the conditional mean. The JVP retains dependence on the primal mean,
    allowing higher derivatives through the surrounding solve as usual.
    This helper is only valid at the conditional mean, not an arbitrary beta.
    """
    return _conditional_quadratic(*args)


@_quadratic_at_conditional_mean.defjvp
def _quadratic_at_conditional_mean_jvp(primals, tangents):
    mean, *parameters = primals
    _, *parameter_tangents = tangents
    return jax.jvp(
        lambda *parameters: _conditional_quadratic(mean, *parameters),
        tuple(parameters),
        tuple(parameter_tangents),
    )


def prepare_collapsed_channel(channel_kwargs: Mapping[str, Any]) -> dict:
    """Prepare fixed design operators for a row with preceding channels.

    ``channel_kwargs`` is the output of ``channel_model_kwargs``. Every
    real/imaginary component may have its own basis and coefficient count.
    Preparation runs on the host; its returned dictionary is model kwargs.
    The first Cholesky row has no regression to integrate and stays explicit.
    """
    if not jax.config.x64_enabled:
        raise ValueError(
            "Multivariate Gaussian marginalization requires JAX_ENABLE_X64=true"
        )
    j = int(channel_kwargs["channel_index"])
    if j < 1:
        raise ValueError("collapsed rows require a preceding channel")
    previous_re = np.asarray(channel_kwargs["u_re_prev"])
    previous_im = np.asarray(channel_kwargs["u_im_prev"])
    names, sizes, penalties, designs_re, designs_im = [], [], [], [], []
    for previous in range(j):
        for part in ("re", "im"):
            basis = np.asarray(
                channel_kwargs[f"basis_theta_{part}_by_component"][previous]
            )
            penalty = np.asarray(
                channel_kwargs[f"penalty_theta_{part}_by_component"][previous],
                dtype=np.float64,
            )
            if not np.isfinite(penalty).all():
                raise ValueError("theta penalties must be finite")
            # Cholesky verifies positive definiteness without silently changing
            # the Gaussian prior with an extra numerical ridge.
            np.linalg.cholesky(penalty)
            names.append(f"theta_{part}_{j}_{previous}")
            sizes.append(penalty.shape[0])
            penalties.append(penalty)
            real = previous_re[:, previous, :]
            imag = previous_im[:, previous, :]
            designs_re.append(
                (real if part == "re" else -imag)[:, :, None]
                * basis[:, None, :]
            )
            designs_im.append(
                (imag if part == "re" else real)[:, :, None]
                * basis[:, None, :]
            )
    penalty = block_diag(*penalties)
    penalty_cholesky = np.linalg.cholesky(penalty)
    design_re = np.concatenate(designs_re, axis=-1)
    design_im = np.concatenate(designs_im, axis=-1)
    observed_re = np.asarray(channel_kwargs["u_re_channel"])
    observed_im = np.asarray(channel_kwargs["u_im_channel"])
    # Reserve most of the shared budget for the more expensive Gram map.
    linear_cache = _prepare_linear_cache(
        design_re,
        design_im,
        observed_re,
        observed_im,
        max_bytes=_DESIGN_CACHE_BYTES // 4,
    )
    gram_budget = _DESIGN_CACHE_BYTES - (
        0 if linear_cache is None else linear_cache.nbytes
    )
    backend = channel_kwargs.get("theta_backend", "auto")
    if backend not in ("auto", "dense", "banded"):
        raise ValueError("theta_backend must be auto, dense or banded")
    plan = (
        None
        if backend == "dense"
        else banded.prepare_plan(
            design_re, design_im, penalty, force=backend == "banded"
        )
    )
    gram_cache = _prepare_gram_cache(
        design_re, design_im, max_bytes=gram_budget, banded_plan=plan
    )
    if plan is not None and gram_cache is None:
        if backend == "banded":
            raise ValueError(
                "banded Gram cache exceeds the fixed memory budget; use theta_backend='auto' or 'dense'"
            )
        plan = (
            None  # The dense path retains the uncached contraction fallback.
        )
    return {
        "channel_index": j,
        "basis_delta": channel_kwargs["basis_delta"],
        "penalty_delta": channel_kwargs["penalty_delta"],
        "roughness_scale": channel_kwargs["roughness_scale"],
        "smoothing_parameterization": channel_kwargs.get(
            "smoothing_parameterization", "centered"
        ),
        "duration": channel_kwargs["duration"],
        "Nb": channel_kwargs["Nb"],
        "Nh": channel_kwargs["Nh"],
        "enbw": channel_kwargs.get("enbw", 1.0),
        "eta": channel_kwargs.get("eta", 1.0),
        "design_re": jnp.asarray(design_re),
        "design_im": jnp.asarray(design_im),
        "gram_cache": gram_cache,
        "banded_plan": plan,
        "linear_cache": linear_cache,
        "observed_re": jnp.asarray(observed_re),
        "observed_im": jnp.asarray(observed_im),
        "theta_names": tuple(names),
        "theta_sizes": tuple(sizes),
        "theta_penalty": jnp.asarray(penalty),
        "theta_penalty_cholesky": jnp.asarray(penalty_cholesky),
        "theta_logdet_penalty": float(
            2 * np.sum(np.log(np.diag(penalty_cholesky)))
        ),
    }


def gaussian_theta_conditional(
    log_delta_sq: jax.Array,
    sigmas: jax.Array,
    *,
    design_re: jax.Array,
    design_im: jax.Array,
    observed_re: jax.Array,
    observed_im: jax.Array,
    theta_sizes: tuple[int, ...],
    theta_penalty: jax.Array,
    theta_penalty_cholesky: jax.Array,
    theta_logdet_penalty: float,
    Nb: int,
    Nh: int,
    duration: float,
    enbw: float = 1.0,
    eta: float = 1.0,
    gram_cache: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    linear_cache: jax.Array | None = None,
    banded_plan: banded.BandedPlan | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return marginal log L, conditional mean, scaled Cholesky, and scales.

    With prior precision Q and weighted design Gram matrix A, the conditional
    precision is H=Q+2A. We factor D^-1 H D^-1 with D=sqrt(diag(H)) to improve
    conditioning. Its conditional covariance is D^-1 L^-T L^-1 D^-1.

    The marginal quadratic is evaluated at the conditional mean as residual
    power plus prior quadratic. This avoids subtracting two large quadratic
    forms when a LISA channel is almost determined by preceding channels.
    With a banded plan, the factor uses column-band storage and scales use
    the plan's ordering; the returned mean always uses original ordering.
    """
    log_variance = jnp.clip(
        log_delta_sq, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT
    )
    likelihood_scale = eta / enbw
    frequency_precision = likelihood_scale * jnp.exp(-log_variance) / duration
    expanded_sigma = jnp.repeat(
        sigmas,
        jnp.asarray(theta_sizes),
        total_repeat_length=theta_penalty.shape[0],
    )
    if banded_plan is not None:
        plan = banded_plan
        sigma = expanded_sigma[plan.permutation]
        prior_precision = plan.penalty / (sigma[:, None] * sigma[plan.rows])
        gram = _weighted_gram_cache(
            gram_cache, frequency_precision, plan.penalty.size
        ).reshape(plan.penalty.shape)
    elif gram_cache is None:
        gram = jnp.einsum(
            "frk,frl,f->kl", design_re, design_re, frequency_precision
        ) + jnp.einsum(
            "frk,frl,f->kl", design_im, design_im, frequency_precision
        )
    else:
        n_weights = theta_penalty.shape[0]
        upper = _weighted_gram_cache(
            gram_cache, frequency_precision, n_weights**2
        )
        upper = upper.reshape(n_weights, n_weights)
        gram = upper + upper.T - jnp.diag(jnp.diag(upper))
    if banded_plan is None:
        prior_precision = theta_penalty / (
            expanded_sigma[:, None] * expanded_sigma[None, :]
        )
    if linear_cache is None:
        linear = 2 * (
            jnp.einsum(
                "frk,fr,f->k", design_re, observed_re, frequency_precision
            )
            + jnp.einsum(
                "frk,fr,f->k", design_im, observed_im, frequency_precision
            )
        )
    else:
        linear = linear_cache @ frequency_precision
    precision = prior_precision + 2 * gram
    penalty_columns = None
    if banded_plan is None:
        scale = jnp.sqrt(jnp.diag(precision))
        balanced = precision / (scale[:, None] * scale[None, :])
        cholesky = jnp.linalg.cholesky(balanced)
        intermediate = solve_triangular(cholesky, linear / scale, lower=True)
        mean = solve_triangular(cholesky.T, intermediate, lower=False) / scale
        logdet_conditional = _logdet_at_precision(precision, cholesky, scale)
    else:
        scale = jnp.sqrt(precision[:, 0])
        balanced = precision / (scale[:, None] * scale[plan.rows])
        cholesky = banded.cholesky(balanced)
        intermediate = banded.solve_triangular(
            cholesky, linear[plan.permutation] / scale
        )
        mean = (
            banded.solve_triangular(cholesky, intermediate, transpose=True)
            / scale
        )[plan.inverse_permutation]
        logdet_conditional = banded.logdet_from_factor(
            precision, cholesky, scale
        )
        theta_penalty_cholesky = plan.penalty_factor
        penalty_columns = plan.permutation[plan.rows]
    quadratic = _quadratic_at_conditional_mean(
        mean,
        expanded_sigma,
        theta_penalty_cholesky,
        design_re,
        design_im,
        observed_re,
        observed_im,
        frequency_precision,
        penalty_columns,
    )
    logdet_prior = theta_logdet_penalty - 2 * jnp.sum(
        jnp.asarray(theta_sizes) * jnp.log(sigmas)
    )
    marginal = (
        -likelihood_scale * Nb * Nh * jnp.sum(log_variance)
        - quadratic
        + 0.5 * (logdet_prior - logdet_conditional)
    )
    return marginal, mean, cholesky, scale


def _conditional_kwargs(prepared: Mapping[str, Any]) -> dict:
    return {
        key: prepared[key]
        for key in (
            "design_re",
            "design_im",
            "gram_cache",
            "linear_cache",
            "banded_plan",
            "observed_re",
            "observed_im",
            "theta_sizes",
            "theta_penalty",
            "theta_penalty_cholesky",
            "theta_logdet_penalty",
            "Nb",
            "Nh",
            "duration",
            "enbw",
            "eta",
        )
    }


def collapsed_channel_model(
    channel_index: int,
    basis_delta: jax.Array,
    penalty_delta: jax.Array,
    roughness_scale: float,
    theta_names: tuple[str, ...],
    smoothing_parameterization: str = "centered",
    **conditional_kwargs: Any,
) -> None:
    """Sample delta coefficients and all scales, integrating theta weights."""
    weights = _sample_pspline_block(
        f"sigma_delta_{channel_index}",
        f"weights_delta_{channel_index}",
        penalty_delta,
        roughness_scale,
        smoothing_parameterization,
    )
    sigmas = jnp.stack(
        [
            numpyro.sample(f"sigma_{name}", dist.HalfNormal(roughness_scale))
            for name in theta_names
        ]
    )
    marginal, _, _, _ = gaussian_theta_conditional(
        basis_delta @ weights, sigmas, **conditional_kwargs
    )
    numpyro.factor(f"likelihood_channel_{channel_index}", marginal)


def sample_conditional_theta(
    posterior: Mapping[str, Any],
    prepared_kwargs: Mapping[str, Any],
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """Recover theta weights and full likelihood for every retained draw.

    Inputs have arbitrary leading sample dimensions, conventionally
    (chain, draw, coefficient). Each hyperparameter draw gets one independent
    conditional Gaussian draw. A sequential ``lax.map`` limits intermediate
    memory to one precision matrix rather than chains*draws such matrices.
    """
    j = int(prepared_kwargs["channel_index"])
    weights = jnp.asarray(posterior[f"weights_delta_{j}"])
    leading_shape = weights.shape[:-1]
    flat_weights = weights.reshape(-1, weights.shape[-1])
    names = prepared_kwargs["theta_names"]
    sigmas = jnp.stack(
        [
            jnp.asarray(posterior[f"sigma_{name}"]).reshape(-1)
            for name in names
        ],
        axis=-1,
    )
    log_delta = flat_weights @ jnp.asarray(prepared_kwargs["basis_delta"]).T
    keys = jax.random.split(rng_key, flat_weights.shape[0])
    conditional_kwargs = _conditional_kwargs(prepared_kwargs)

    def draw(inputs):
        logd, scales, key = inputs
        _, mean, cholesky, diagonal = gaussian_theta_conditional(
            logd, scales, **conditional_kwargs
        )
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        plan = prepared_kwargs.get("banded_plan")
        if plan is None:
            deviation = (
                solve_triangular(
                    cholesky.T,
                    noise,
                    lower=False,
                )
                / diagonal
            )
        else:
            deviation = (
                banded.solve_triangular(cholesky, noise, transpose=True)
                / diagonal
            )[plan.inverse_permutation]
        beta = mean + deviation
        residual_re = prepared_kwargs["observed_re"] - jnp.einsum(
            "frk,k->fr", prepared_kwargs["design_re"], beta
        )
        residual_im = prepared_kwargs["observed_im"] - jnp.einsum(
            "frk,k->fr", prepared_kwargs["design_im"], beta
        )
        logd = jnp.clip(logd, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT)
        full_log_likelihood = (
            (
                -prepared_kwargs["Nb"] * prepared_kwargs["Nh"] * jnp.sum(logd)
                - jnp.sum(
                    (residual_re**2 + residual_im**2) * jnp.exp(-logd)[:, None]
                )
                / prepared_kwargs["duration"]
            )
            * prepared_kwargs["eta"]
            / prepared_kwargs["enbw"]
        )
        return beta, full_log_likelihood

    beta, log_likelihood = jax.lax.map(draw, (log_delta, sigmas, keys))
    output = {}
    offset = 0
    for name, size in zip(names, prepared_kwargs["theta_sizes"], strict=True):
        output[f"weights_{name}"] = beta[:, offset : offset + size].reshape(
            *leading_shape, size
        )
        offset += size
    output[f"log_likelihood_block_{j}"] = log_likelihood.reshape(leading_shape)
    return output
