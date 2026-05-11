"""Helpers for MorphZ evidence estimation in the pipeline."""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Callable
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpyro.infer.util import log_density

from ..datatypes.multivar import MultivarFFT
from ..logger import logger
from .models import _blocked_channel_model
from .stages import _channel_model_kwargs

_NONCONVERGENCE_MESSAGE = "Convergence not reached within"


class _MorphZWarningCounter(logging.Handler):
    """Count MorphZ bridge-sampling non-convergence warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.nonconverged_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _NONCONVERGENCE_MESSAGE in record.getMessage():
            self.nonconverged_count += 1


@dataclass(frozen=True)
class MorphZEvidenceResult:
    """Summary of MorphZ evidence estimates."""

    lnz: float
    lnz_err: float
    is_valid: bool
    n_estimations: int
    nonconverged_count: int
    estimates: np.ndarray
    factor_results: tuple[MorphZEvidenceResult, ...] = ()


@dataclass(frozen=True)
class _ParamSpec:
    name: str
    shape: tuple[int, ...]
    width: int


def _as_jax_tree(tree: Any) -> Any:
    def _convert_leaf(value: Any) -> Any:
        if isinstance(value, jax.Array | jnp.ndarray | np.ndarray):
            return jnp.asarray(value)
        return value

    return jax.tree_util.tree_map(_convert_leaf, tree)


def _flatten_posterior_samples(
    posterior: xr.Dataset,
    param_names: list[str],
) -> tuple[np.ndarray, list[_ParamSpec]]:
    """Return flattened posterior samples with deterministic parameter order."""
    parts: list[np.ndarray] = []
    specs: list[_ParamSpec] = []
    n_rows: int | None = None

    for name in param_names:
        values = np.asarray(posterior[name].values)
        if values.ndim < 2:
            raise ValueError(
                f"Posterior variable {name} must have chain/draw dimensions."
            )
        sample_shape = tuple(int(v) for v in values.shape[2:])
        flat = values.reshape((-1, int(np.prod(sample_shape, dtype=int) or 1)))
        if n_rows is None:
            n_rows = int(flat.shape[0])
        elif int(flat.shape[0]) != n_rows:
            raise ValueError(
                f"Posterior variable {name} has inconsistent sample count."
            )
        parts.append(flat)
        specs.append(
            _ParamSpec(
                name=name,
                shape=sample_shape,
                width=int(flat.shape[1]),
            )
        )

    if not parts:
        raise ValueError("No posterior variables available for lnZ.")
    return np.concatenate(parts, axis=1), specs


def _unpack_params(
    sample_vec: jnp.ndarray,
    specs: list[_ParamSpec],
) -> dict[str, jnp.ndarray]:
    params: dict[str, jnp.ndarray] = {}
    start = 0
    for spec in specs:
        stop = start + spec.width
        value = sample_vec[start:stop]
        if spec.shape:
            value = jnp.reshape(value, spec.shape)
        else:
            value = value[0]
        params[spec.name] = value
        start = stop
    return params


def _param_belongs_to_channel(name: str, channel_index: int) -> bool:
    j = int(channel_index)
    prefixes = (
        f"delta_{j}",
        f"phi_delta_{j}",
        f"weights_delta_{j}",
        f"delta_theta_re_{j}_",
        f"phi_theta_re_{j}_",
        f"weights_theta_re_{j}_",
        f"delta_theta_im_{j}_",
        f"phi_theta_im_{j}_",
        f"weights_theta_im_{j}_",
    )
    return any(str(name).startswith(prefix) for prefix in prefixes)


def _posterior_param_names(
    posterior: xr.Dataset,
    *,
    channel_index: int | None = None,
) -> list[str]:
    """Return posterior variable names for the full model or one channel."""
    if channel_index is None:
        return [
            name
            for name in ("weights", "phi", "delta")
            if name in posterior.data_vars
        ]
    return sorted(
        str(name)
        for name in posterior.data_vars
        if _param_belongs_to_channel(str(name), channel_index)
    )


def _build_log_posterior(
    posterior: xr.Dataset,
    *,
    model_fn: Callable[..., Any],
    model_kwargs: dict[str, Any],
    param_names: list[str],
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], float]]:
    """Build posterior samples plus a callable log-posterior evaluator."""
    post_samples, specs = _flatten_posterior_samples(posterior, param_names)
    model_kwargs_jax = _as_jax_tree(model_kwargs)

    @jax.jit
    def _logpost(sample_vec: jnp.ndarray) -> jnp.ndarray:
        params = _unpack_params(sample_vec, specs)
        log_prob, _ = log_density(
            model_fn,
            (),
            model_kwargs_jax,
            params,
        )
        return log_prob

    log_posterior_values = np.asarray(
        jax.vmap(_logpost)(jnp.asarray(post_samples)),
        dtype=float,
    )

    def log_posterior_fn(theta: np.ndarray) -> float:
        return float(_logpost(jnp.asarray(theta)))

    return post_samples, log_posterior_values, log_posterior_fn


def _default_lnz_output_path(outdir: str | None) -> str:
    if outdir is not None:
        path = Path(outdir) / "diagnostics" / "morphz"
    else:
        path = Path(tempfile.gettempdir()) / "log_psplines_morphz"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _default_lnz_kwargs(
    *,
    n_draws: int,
    posterior_dim: int,
    data: MultivarFFT,
    outdir: str | None,
    extra_kwargs: dict[str, Any] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Build default MorphZ kwargs for one evidence estimation task."""
    lnz_kwargs = dict((extra_kwargs or {}).get("lnz_kwargs", {}))
    # After whitening the samples have identity covariance, so:
    # - thin=1: use all samples for the most accurate whitening transform
    # - morph_type="indep": optimal for whitened (approximately i.i.d.) samples
    lnz_kwargs.setdefault("thin", 1)
    lnz_kwargs.setdefault(
        "n_resamples",
        int(min(512, max(64, n_draws // 2))),
    )
    lnz_kwargs.setdefault("kde_fraction", 0.5)
    lnz_kwargs.setdefault("bridge_start_fraction", 0.5)
    lnz_kwargs.setdefault("max_iter", 5000)
    lnz_kwargs.setdefault("tol", 1e-2)
    lnz_kwargs.setdefault("morph_type", "indep")
    lnz_kwargs.setdefault("n_estimations", 1)
    lnz_kwargs.setdefault("kde_bw", "silverman")
    lnz_kwargs.setdefault("verbose", verbose)
    lnz_kwargs.setdefault("plot", False)
    lnz_kwargs.setdefault("show_progress", False)
    lnz_kwargs.setdefault("output_path", _default_lnz_output_path(outdir))
    return lnz_kwargs


def _factor_output_path(base_path: str, factor: int) -> str:
    path = Path(base_path) / f"factor_{int(factor)}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _combine_factor_estimates(
    factor_results: list[MorphZEvidenceResult],
) -> np.ndarray:
    """Combine row-wise factor estimates when all factors expose them."""
    if not factor_results:
        return np.empty((0, 2), dtype=float)
    if any(
        result.estimates.ndim != 2 or result.estimates.shape[1] < 2
        for result in factor_results
    ):
        return np.empty((0, 2), dtype=float)
    first_shape = factor_results[0].estimates.shape
    if any(result.estimates.shape != first_shape for result in factor_results):
        return np.empty((0, 2), dtype=float)

    estimates = np.stack(
        [result.estimates for result in factor_results], axis=0
    )
    lnz = np.sum(estimates[:, :, 0], axis=0)
    lnz_err = np.sqrt(np.sum(estimates[:, :, 1] ** 2, axis=0))
    return np.column_stack([lnz, lnz_err])


def _combine_factor_results(
    factor_results: list[MorphZEvidenceResult],
) -> MorphZEvidenceResult:
    """Aggregate independent factor evidence estimates in log space."""
    if not factor_results:
        return MorphZEvidenceResult(
            lnz=np.nan,
            lnz_err=np.nan,
            is_valid=False,
            n_estimations=0,
            nonconverged_count=0,
            estimates=np.empty((0, 2), dtype=float),
        )

    combined_estimates = _combine_factor_estimates(factor_results)
    is_valid = all(result.is_valid for result in factor_results)
    if not is_valid:
        return MorphZEvidenceResult(
            lnz=np.nan,
            lnz_err=np.nan,
            is_valid=False,
            n_estimations=0,
            nonconverged_count=int(
                sum(result.nonconverged_count for result in factor_results)
            ),
            estimates=combined_estimates,
            factor_results=tuple(factor_results),
        )

    return MorphZEvidenceResult(
        lnz=float(sum(result.lnz for result in factor_results)),
        lnz_err=float(
            np.sqrt(sum(result.lnz_err**2 for result in factor_results))
        ),
        is_valid=True,
        n_estimations=int(
            min(result.n_estimations for result in factor_results)
        ),
        nonconverged_count=int(
            sum(result.nonconverged_count for result in factor_results)
        ),
        estimates=combined_estimates,
        factor_results=tuple(factor_results),
    )


def run_morphz_evidence(
    *,
    post_samples: np.ndarray,
    log_posterior_values: np.ndarray,
    log_posterior_function: Callable[[np.ndarray], float],
    **kwargs: Any,
) -> MorphZEvidenceResult:
    """Run MorphZ and invalidate results when every bridge run fails to converge."""
    import morphZ

    bridge_logger = logging.getLogger("morphZ.bridge")
    warning_counter = _MorphZWarningCounter()
    bridge_logger.addHandler(warning_counter)
    quiet = not bool(kwargs.get("verbose", False)) and not bool(
        kwargs.get("show_progress", False)
    )
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    stdout_cm = redirect_stdout(stdout_buffer) if quiet else nullcontext()
    stderr_cm = redirect_stderr(stderr_buffer) if quiet else nullcontext()
    try:
        try:
            with stdout_cm, stderr_cm:
                estimates_raw = morphZ.evidence(
                    post_samples=post_samples,
                    log_posterior_values=log_posterior_values,
                    log_posterior_function=log_posterior_function,
                    **kwargs,
                )
        except Exception as exc:
            logger.warning(f"MorphZ evidence evaluation failed: {exc}")
            return MorphZEvidenceResult(
                lnz=np.nan,
                lnz_err=np.nan,
                is_valid=False,
                n_estimations=0,
                nonconverged_count=warning_counter.nonconverged_count,
                estimates=np.empty((0, 2), dtype=float),
            )
    finally:
        bridge_logger.removeHandler(warning_counter)

    estimates = np.asarray(estimates_raw, dtype=float)
    if estimates.ndim == 1:
        estimates = estimates[None, :]

    n_estimations = int(estimates.shape[0]) if estimates.size else 0
    finite_rows = (
        np.all(np.isfinite(estimates), axis=1)
        if estimates.ndim == 2 and estimates.shape[1] >= 2
        else np.zeros((n_estimations,), dtype=bool)
    )
    usable_estimates = estimates[finite_rows]
    all_nonconverged = (
        n_estimations > 0
        and warning_counter.nonconverged_count >= n_estimations
    )
    is_valid = (usable_estimates.size > 0) and not all_nonconverged
    if not is_valid:
        return MorphZEvidenceResult(
            lnz=np.nan,
            lnz_err=np.nan,
            is_valid=False,
            n_estimations=n_estimations,
            nonconverged_count=warning_counter.nonconverged_count,
            estimates=estimates,
        )

    return MorphZEvidenceResult(
        lnz=float(np.median(usable_estimates[:, 0])),
        lnz_err=float(np.median(usable_estimates[:, 1])),
        is_valid=True,
        n_estimations=n_estimations,
        nonconverged_count=warning_counter.nonconverged_count,
        estimates=estimates,
    )


def _whiten_samples(
    post_samples: np.ndarray,
    log_posterior_values: np.ndarray,
    log_posterior_function: Callable[[np.ndarray], float],
    *,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], float], float]:
    """Whiten posterior samples so the sample covariance is the identity.

    KDE-based bridge sampling (morphZ) fails when the posterior lives in a
    thin manifold — the KDE fills a volume exponentially larger than the true
    posterior, so the bridge weights collapse.  Whitening maps

        z = L⁻¹ (θ − μ)

    where μ = sample mean, L = chol(Cov).  In z-space the distribution is
    approximately spherical, so an independent-1D KDE is a near-perfect
    proposal and bridge sampling converges quickly.

    morphZ estimates lnZ in z-space, which equals the true lnZ plus
    log|det L| (the log-Jacobian of the transform).  The caller must subtract
    log_det_L from the returned lnZ.

    Returns
    -------
    z_samples          : whitened samples, shape (n, d)
    log_post_z         : log-posterior in z-space = log_post + log_det_L
    log_post_z_fn      : callable log-posterior in z-space
    log_det_L          : log|det(L)| — subtract from morphZ result to get lnZ
    """
    n, d = post_samples.shape
    mean = np.mean(post_samples, axis=0)
    cov = np.cov(post_samples, rowvar=False) if n > 1 else np.eye(d)

    reg = regularization * np.trace(cov) / max(d, 1)
    cov_reg = cov + reg * np.eye(d)

    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, reg)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    L_inv = np.linalg.inv(L)
    log_det_L = float(np.sum(np.log(np.abs(np.diag(L)))))

    z_samples = (post_samples - mean) @ L_inv.T
    log_post_z = log_posterior_values + log_det_L

    _L = L.copy()
    _mean = mean.copy()

    def log_post_z_fn(z: np.ndarray) -> float:
        return log_posterior_function(_L @ z + _mean) + log_det_L

    return z_samples, log_post_z, log_post_z_fn, log_det_L


def _correct_lnz_result(
    result: MorphZEvidenceResult,
    lnz_correction: float,
) -> MorphZEvidenceResult:
    """Subtract lnz_correction from all lnZ fields (whitening Jacobian)."""
    if lnz_correction == 0.0:
        return result
    corrected_estimates = result.estimates.copy()
    if corrected_estimates.ndim == 2 and corrected_estimates.shape[1] >= 1:
        corrected_estimates[:, 0] -= lnz_correction
    return MorphZEvidenceResult(
        lnz=result.lnz - lnz_correction,
        lnz_err=result.lnz_err,
        is_valid=result.is_valid,
        n_estimations=result.n_estimations,
        nonconverged_count=result.nonconverged_count,
        estimates=corrected_estimates,
        factor_results=tuple(
            _correct_lnz_result(fr, lnz_correction)
            for fr in result.factor_results
        ),
    )


def _evaluate_lnz_task(
    *,
    post_samples: np.ndarray,
    log_post: np.ndarray,
    log_post_fn: Callable[[np.ndarray], float],
    data: MultivarFFT,
    outdir: str | None,
    extra_kwargs: dict[str, Any] | None,
    verbose: bool,
    factor_index: int | None = None,
    lnz_correction: float = 0.0,
) -> MorphZEvidenceResult:
    """Run one MorphZ estimation task and log a concise summary."""
    n_draws = int(post_samples.shape[0])
    posterior_dim = int(post_samples.shape[1])
    lnz_kwargs = _default_lnz_kwargs(
        n_draws=n_draws,
        posterior_dim=posterior_dim,
        data=data,
        outdir=outdir,
        extra_kwargs=extra_kwargs,
        verbose=verbose,
    )
    factor_msg = (
        f" for factor {factor_index}" if factor_index is not None else ""
    )
    if factor_index is not None:
        lnz_kwargs["output_path"] = _factor_output_path(
            lnz_kwargs["output_path"], factor_index
        )
    logger.info(
        f"Computing MorphZ lnZ{factor_msg} with {n_draws} posterior samples "
        f"and {posterior_dim} parameters (whitened) using morph_type="
        f"{lnz_kwargs['morph_type']}."
    )
    result = run_morphz_evidence(
        post_samples=post_samples,
        log_posterior_values=log_post,
        log_posterior_function=log_post_fn,
        **lnz_kwargs,
    )
    result = _correct_lnz_result(result, lnz_correction)
    if factor_index is not None:
        logger.info(
            f"Factor {factor_index} lnZ valid={result.is_valid} "
            f"lnz={result.lnz} lnz_err={result.lnz_err}."
        )
    return result


def estimate_pipeline_lnz(
    *,
    idata: xr.DataTree,
    data: MultivarFFT,
    model_kwargs: dict[str, Any],
    outdir: str | None,
    extra_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> MorphZEvidenceResult:
    """Compute MorphZ lnZ from pipeline posterior samples."""
    posterior = idata["posterior"].dataset
    if posterior is None:
        raise ValueError("idata.posterior is required for lnZ computation.")

    n_channels = int(model_kwargs["n_channels"])
    factor_results: list[MorphZEvidenceResult] = []
    for channel_index in range(n_channels):
        post_samples, log_post, log_post_fn = _build_log_posterior(
            posterior,
            model_fn=_blocked_channel_model,
            model_kwargs=_channel_model_kwargs(model_kwargs, channel_index),
            param_names=_posterior_param_names(
                posterior, channel_index=channel_index
            ),
        )
        post_samples, log_post, log_post_fn, log_det_L = _whiten_samples(
            post_samples, log_post, log_post_fn
        )
        factor_results.append(
            _evaluate_lnz_task(
                post_samples=post_samples,
                log_post=log_post,
                log_post_fn=log_post_fn,
                data=data,
                outdir=outdir,
                extra_kwargs=extra_kwargs,
                verbose=verbose,
                factor_index=channel_index,
                lnz_correction=log_det_L,
            )
        )

    return _combine_factor_results(factor_results)


def _posterior_coords(posterior: xr.Dataset) -> dict[str, np.ndarray]:
    return {
        "chain": np.asarray(posterior.coords["chain"].values),
        "draw": np.asarray(posterior.coords["draw"].values),
    }


def _channel_theta_array(
    posterior: xr.Dataset,
    basis_list: list[Any],
    channel_index: int,
    param_type: str,
) -> np.ndarray:
    """Evaluate per-draw theta spline values for one multivariate channel."""
    if channel_index <= 0:
        weights_delta = np.asarray(
            posterior[f"weights_delta_{channel_index}"].values,
            dtype=np.float64,
        )
        n_chain, n_draw = weights_delta.shape[:2]
        n_freq = int(np.asarray(basis_list[0]).shape[0]) if basis_list else 0
        return np.zeros((n_chain, n_draw, n_freq, 0), dtype=np.float64)

    theta_parts: list[np.ndarray] = []
    for theta_idx, theta_basis in enumerate(basis_list):
        weights_name = (
            f"weights_theta_{param_type}_{channel_index}_{theta_idx}"
        )
        weights = np.asarray(posterior[weights_name].values, dtype=np.float64)
        basis = np.asarray(theta_basis, dtype=np.float64)
        theta_eval = np.einsum("fk,cdk->cdf", basis, weights)
        theta_parts.append(theta_eval)
    return np.stack(theta_parts, axis=-1)


def _pointwise_multivar_log_likelihood(
    posterior: xr.Dataset,
    data: MultivarFFT,
    model_kwargs: dict[str, Any],
) -> xr.Dataset:
    """Return blocked multivariate pointwise log-likelihood draws by frequency."""
    u_re = np.asarray(model_kwargs["u_re"], dtype=np.float64)
    u_im = np.asarray(model_kwargs["u_im"], dtype=np.float64)
    bases_delta = model_kwargs["bases_delta"]
    bases_theta_re = model_kwargs["bases_theta_re"]
    bases_theta_im = model_kwargs["bases_theta_im"]
    nb = float(model_kwargs["Nb"])
    nh = float(model_kwargs["Nh"])
    duration = float(model_kwargs["duration"])
    enbw = float(model_kwargs.get("enbw", 1.0))

    freq = np.asarray(data.freq, dtype=np.float64)
    p = int(model_kwargs["n_channels"])
    coords = _posterior_coords(posterior)
    coords["freq"] = freq

    channel_terms: dict[str, xr.DataArray] = {}
    total_pointwise: np.ndarray | None = None

    for channel_index in range(p):
        weights_delta = np.asarray(
            posterior[f"weights_delta_{channel_index}"].values,
            dtype=np.float64,
        )
        basis_delta = np.asarray(bases_delta[channel_index], dtype=np.float64)
        log_delta_sq = np.einsum("fk,cdk->cdf", basis_delta, weights_delta)
        log_delta_sq = np.clip(log_delta_sq, a_min=-80.0, a_max=80.0)
        delta_eff_sq = np.exp(log_delta_sq)

        pointwise = -nb * nh * np.log(delta_eff_sq)

        u_re_channel = u_re[:, channel_index, :]
        u_im_channel = u_im[:, channel_index, :]
        if channel_index > 0:
            theta_re = _channel_theta_array(
                posterior,
                list(bases_theta_re[channel_index]),
                channel_index,
                "re",
            )
            theta_im = _channel_theta_array(
                posterior,
                list(bases_theta_im[channel_index]),
                channel_index,
                "im",
            )
            u_re_prev = u_re[:, :channel_index, :]
            u_im_prev = u_im[:, :channel_index, :]

            contrib_re = np.einsum(
                "cdfl,flr->cdfr", theta_re, u_re_prev
            ) - np.einsum("cdfl,flr->cdfr", theta_im, u_im_prev)
            contrib_im = np.einsum(
                "cdfl,flr->cdfr", theta_re, u_im_prev
            ) + np.einsum("cdfl,flr->cdfr", theta_im, u_re_prev)
            u_re_resid = u_re_channel[None, None, :, :] - contrib_re
            u_im_resid = u_im_channel[None, None, :, :] - contrib_im
        else:
            u_re_resid = u_re_channel[None, None, :, :]
            u_im_resid = u_im_channel[None, None, :, :]

        residual_power_sum = np.sum(u_re_resid**2 + u_im_resid**2, axis=-1)
        pointwise = pointwise - residual_power_sum / (duration * delta_eff_sq)
        pointwise = pointwise / enbw

        channel_terms[f"log_likelihood_channel_{channel_index}"] = (
            xr.DataArray(
                pointwise,
                dims=("chain", "draw", "freq"),
                coords=coords,
            )
        )
        total_pointwise = (
            pointwise
            if total_pointwise is None
            else total_pointwise + pointwise
        )

    if total_pointwise is None:
        raise ValueError(
            "No multivariate channels available for pointwise log-likelihood."
        )

    return xr.Dataset(
        {
            "log_likelihood": xr.DataArray(
                total_pointwise,
                dims=("chain", "draw", "freq"),
                coords=coords,
            ),
            **channel_terms,
        }
    )


def compute_pointwise_lnl(
    *,
    idata: xr.DataTree,
    data: MultivarFFT,
    model_kwargs: dict[str, Any],
) -> xr.Dataset:
    """Compute pointwise log-likelihood contributions for PSIS-LOO.

    Each retained frequency bin is treated as one observation. For multivariate
    fits, the dataset includes a total ``log_likelihood`` variable plus
    per-channel diagnostics.
    """
    posterior = idata["posterior"].dataset
    if posterior is None:
        raise ValueError(
            "idata.posterior is required to compute pointwise log-likelihood."
        )

    return _pointwise_multivar_log_likelihood(posterior, data, model_kwargs)


__all__ = [
    "MorphZEvidenceResult",
    "compute_pointwise_lnl",
    "estimate_pipeline_lnz",
    "run_morphz_evidence",
]
