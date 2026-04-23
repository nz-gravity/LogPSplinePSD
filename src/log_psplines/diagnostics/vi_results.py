"""VI result diagnostics and PSD reconstruction helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..logger import logger
from .psd_compare import compute_multivar_riae_diagnostics


def _to_np(x) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _to_np_dict(d: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    return {} if not d else {name: _to_np(value) for name, value in d.items()}


def _get_scaling_factor(*sources) -> float:
    for src in sources:
        val = getattr(src, "scaling_factor", None) if src is not None else None
        if val is not None:
            return float(val)
    return 1.0


def _make_psd_rescaler(sampler) -> Callable[[np.ndarray], np.ndarray]:
    channel_stds = getattr(
        getattr(sampler, "fft_data", None), "channel_stds", None
    )
    if channel_stds is None:
        return lambda arr: arr
    stds = np.asarray(channel_stds, dtype=np.float64)
    factor_matrix = np.outer(stds, stds).astype(np.float64)
    return lambda arr: arr * factor_matrix


def _build_psd_quantiles_dict(
    psd_real_q: np.ndarray,
    psd_imag_q: np.ndarray,
    coh_percentiles: Optional[np.ndarray] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    psd_quantiles: Dict[str, Any] = {
        "posterior_psd": np.asarray(psd_real_q, dtype=np.float64)
        + 1j * np.asarray(psd_imag_q, dtype=np.float64)
    }
    coherence_quantiles = (
        None
        if coh_percentiles is None
        else {
            "q05": coh_percentiles[0],
            "q50": coh_percentiles[1],
            "q95": coh_percentiles[2],
        }
    )
    return psd_quantiles, coherence_quantiles


def _reconstruct_psd_quantiles_from_draws(
    *,
    spline_model,
    config,
    log_delta_samples: jnp.ndarray,
    theta_re_samples: jnp.ndarray,
    theta_im_samples: jnp.ndarray,
    p: int,
    rescale_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[
    Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[np.ndarray]
]:
    n_available = int(log_delta_samples.shape[0])
    max_cfg = int(getattr(config, "vi_psd_max_draws", 0) or 0)
    n_draws = max(1, min(n_available, max_cfg) if max_cfg > 0 else n_available)
    if n_draws < n_available:
        logger.debug(
            f"Capping VI PSD reconstruction to {n_draws} draws "
            f"(limit={getattr(config, 'vi_psd_max_draws', 0)})."
        )
    psd_real_q, psd_imag_q, coh_percentiles = (
        spline_model.compute_psd_quantiles(
            log_delta_samples[:n_draws],
            theta_re_samples[:n_draws],
            theta_im_samples[:n_draws],
            percentiles=[5.0, 50.0, 95.0],
            n_samples_max=n_draws,
            compute_coherence=p > 1,
        )
    )
    psd_quantiles, coherence_quantiles = _build_psd_quantiles_dict(
        rescale_fn(psd_real_q),
        rescale_fn(psd_imag_q),
        coh_percentiles,
    )
    return (
        psd_quantiles,
        coherence_quantiles,
        np.asarray(psd_quantiles["posterior_psd"][1], dtype=np.complex128),
    )


def _extract_true_psd(
    sampler,
    rescale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[np.ndarray]:
    if sampler.config.true_psd is None:
        return None
    psd = _to_np(sampler.config.true_psd)
    return rescale_fn(psd) if rescale_fn is not None else psd


def _extract_posterior_psd(psd_quantiles: Any) -> Optional[np.ndarray]:
    posterior_psd = (
        psd_quantiles.get("posterior_psd")
        if isinstance(psd_quantiles, dict)
        else None
    )
    if posterior_psd is None:
        return None
    posterior_psd = np.asarray(posterior_psd, dtype=np.complex128)
    return (
        None
        if posterior_psd.ndim < 1
        else posterior_psd[1] if posterior_psd.shape[0] >= 3 else posterior_psd
    )


def _extract_psd_q50(
    diagnostics: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    return (
        None
        if not diagnostics
        else _extract_posterior_psd(diagnostics.get("psd_quantiles"))
    )


def _extract_multivar_design_psd(diagnostics: Optional[Dict[str, Any]]):
    if not diagnostics:
        return None
    design = diagnostics.get("psd_matrix_complex")
    return (
        np.asarray(design, dtype=np.complex128)
        if design is not None
        else _extract_posterior_psd(diagnostics.get("psd_quantiles"))
    )


def _rescale_multivar_psd_for_diagnostics(
    sampler, psd: np.ndarray
) -> np.ndarray:
    return np.asarray(_make_psd_rescaler(sampler)(np.asarray(psd)))


def _ensure_positive_definite_psd(
    psd_matrix: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    arr = np.asarray(psd_matrix, dtype=np.complex128).copy()
    arr = 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))
    eye = np.eye(arr.shape[-1], dtype=np.complex128)
    for idx in range(arr.shape[0]):
        vals = np.linalg.eigvalsh(arr[idx]).real
        if not np.isfinite(vals).all():
            raise ValueError(
                "Encountered non-finite eigenvalues in PSD matrix"
            )
        min_eig = float(vals.min(initial=np.inf))
        if min_eig < eps:
            arr[idx] += (eps - min_eig + eps) * eye
    return arr


def _build_univar_vi_diagnostics(sampler, vi_result) -> Dict[str, Any]:
    scaling = _get_scaling_factor(
        getattr(sampler, "periodogram", None), sampler.config
    )
    diagnostics: Dict[str, Any] = {
        "weights": None,
        "psd": None,
        "true_psd": _extract_true_psd(sampler),
    }
    weights = vi_result.means.get("weights")
    if weights is not None:
        diagnostics["weights"] = _to_np(weights)
        diagnostics["psd"] = (
            _to_np(jnp.exp(sampler.spline_model(vi_result.means["weights"])))
            * scaling
        )
    weights_draws = (
        None if vi_result.samples is None else vi_result.samples.get("weights")
    )
    if weights_draws is not None and weights_draws.size:
        psd_draws = (
            np.asarray(
                jax.device_get(
                    jnp.exp(
                        jax.vmap(sampler.spline_model)(
                            jnp.asarray(weights_draws)
                        )
                    )
                )
            )
            * scaling
        )
        q05, q50, q95 = np.percentile(psd_draws, [5, 50, 95], axis=0)
        diagnostics["psd_quantiles"] = {"q05": q05, "q50": q50, "q95": q95}
    vi_samples = _to_np_dict(vi_result.samples)
    if vi_samples:
        diagnostics["vi_samples"] = vi_samples
    return diagnostics


def _vi_weights_to_log_delta_theta(
    values: Dict[str, Any],
    sampler,
    *,
    is_batch: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_delta_sq = jnp.stack(
        [
            (
                jnp.asarray(values[f"weights_delta_{ch}"])
                @ jnp.asarray(sampler.all_bases[ch]).T
                if is_batch
                else jnp.einsum(
                    "nk,k->n",
                    jnp.asarray(sampler.all_bases[ch]),
                    jnp.asarray(values[f"weights_delta_{ch}"]),
                )
            )
            for ch in range(sampler.p)
        ],
        axis=-1,
    )
    if not is_batch:
        log_delta_sq = log_delta_sq[None, ...]
    n_samples = log_delta_sq.shape[0]
    theta_re = jnp.zeros((n_samples, sampler.N, sampler.n_theta))
    theta_im = jnp.zeros_like(theta_re)
    for j in range(1, sampler.p):
        for l in range(j):
            idx = sampler.spline_model.theta_index(j, l)
            key_re, key_im = (
                f"weights_theta_re_{j}_{l}",
                f"weights_theta_im_{j}_{l}",
            )
            if key_re not in values or key_im not in values:
                continue
            basis_re = jnp.asarray(
                sampler.spline_model.get_theta_model("re", j, l).basis
            )
            basis_im = jnp.asarray(
                sampler.spline_model.get_theta_model("im", j, l).basis
            )
            w_re, w_im = jnp.asarray(values[key_re]), jnp.asarray(
                values[key_im]
            )
            re_eval = (
                w_re @ basis_re.T
                if is_batch
                else jnp.einsum("nk,k->n", basis_re, w_re)[None, :]
            )
            im_eval = (
                w_im @ basis_im.T
                if is_batch
                else jnp.einsum("nk,k->n", basis_im, w_im)[None, :]
            )
            theta_re = theta_re.at[:, :, idx].set(re_eval)
            theta_im = theta_im.at[:, :, idx].set(im_eval)
    return log_delta_sq, theta_re, theta_im


def _build_multivar_vi_diagnostics(sampler, vi_result) -> Dict[str, Any]:
    rescale_psd = _make_psd_rescaler(sampler)
    diagnostics: Dict[str, Any] = {
        "psd_matrix": None,
        "true_psd": _extract_true_psd(sampler, rescale_fn=rescale_psd),
    }
    try:
        log_delta_sq, theta_re, theta_im = _vi_weights_to_log_delta_theta(
            vi_result.means, sampler, is_batch=False
        )
        vi_psd = sampler.spline_model.reconstruct_psd_matrix(
            log_delta_sq, theta_re, theta_im, n_samples_max=1
        )[0]
        diagnostics["psd_matrix"] = rescale_psd(np.asarray(vi_psd))
        samples_tree = vi_result.samples or {}
        if samples_tree and all(
            f"weights_delta_{ch}" in samples_tree for ch in range(sampler.p)
        ):
            ld_s, tr_s, ti_s = _vi_weights_to_log_delta_theta(
                samples_tree, sampler, is_batch=True
            )
            psd_quantiles, coherence_quantiles, diagnostics["psd_matrix"] = (
                _reconstruct_psd_quantiles_from_draws(
                    spline_model=sampler.spline_model,
                    config=sampler.config,
                    log_delta_samples=ld_s,
                    theta_re_samples=tr_s,
                    theta_im_samples=ti_s,
                    p=sampler.p,
                    rescale_fn=rescale_psd,
                )
            )
            diagnostics["psd_quantiles"] = psd_quantiles
            if coherence_quantiles is not None:
                diagnostics["coherence_quantiles"] = coherence_quantiles
    except (OverflowError, KeyError) as err:  # pragma: no cover
        if sampler.config.verbose:
            logger.warning(f"Could not build VI PSD diagnostics: {err}")
    vi_samples = _to_np_dict(vi_result.samples)
    if vi_samples:
        diagnostics["vi_samples"] = vi_samples
    true_psd = diagnostics["true_psd"]
    vi_psd_np = diagnostics["psd_matrix"]
    if true_psd is not None and vi_psd_np is not None:
        diagnostics.update(
            compute_multivar_riae_diagnostics(
                vi_psd_np,
                np.real(true_psd),
                np.asarray(sampler.freq, dtype=np.float64),
                psd_quantiles=diagnostics.get("psd_quantiles"),
            )
        )
    return diagnostics
