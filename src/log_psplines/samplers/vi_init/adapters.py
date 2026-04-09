"""Adapters that encapsulate VI initialisation for samplers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import init_to_value

from ...datatypes.multivar_utils import _interp_complex_matrix
from ...diagnostics.psd_compare import compute_multivar_riae_diagnostics
from ...logger import logger
from ..pspline_block import (
    build_log_density_fn,
    pspline_hyperparameter_initials,
)
from .bridge import transfer_block_init_values, transfer_univar_weights
from .core import fit_vi
from .defaults import (
    default_init_values_multivar,
    default_init_values_univar,
)
from .guide import (
    suggest_guide_block,
    suggest_guide_multivar,
    suggest_guide_univar,
)
from .mixin import (
    VIInitialisationArtifacts,
    _compute_psis_khat,
    _interpret_khat,
)

# ---------------------------------------------------------------------------
# Shared helpers (Phase 1 – small pure utilities)
# ---------------------------------------------------------------------------


def _get_scaling_factor(*sources) -> float:
    """Return the first positive ``scaling_factor`` found on *sources*."""
    for src in sources:
        if src is None:
            continue
        val = getattr(src, "scaling_factor", None)
        if val is not None:
            return float(val)
    return 1.0


def _make_psd_rescaler(sampler) -> Callable[[np.ndarray], np.ndarray]:
    """Build a PSD rescaling closure from channel standard deviations.

    Returns the identity function when ``channel_stds`` is unavailable.
    """
    channel_stds = getattr(
        getattr(sampler, "fft_data", None), "channel_stds", None
    )
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float32)
        factor_matrix = np.outer(channel_stds, channel_stds).astype(np.float32)
        return lambda arr: arr * factor_matrix
    return lambda arr: arr


def _build_psd_quantiles_dict(
    psd_real_q: np.ndarray,
    psd_imag_q: np.ndarray,
    coh_percentiles: Optional[np.ndarray] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Build ``psd_quantiles`` and ``coherence_quantiles`` dicts."""
    psd_quantiles: Dict[str, Any] = {
        "real": {
            "q05": psd_real_q[0],
            "q50": psd_real_q[1],
            "q95": psd_real_q[2],
        },
        "imag": {
            "q05": psd_imag_q[0],
            "q50": psd_imag_q[1],
            "q95": psd_imag_q[2],
        },
    }
    coherence_quantiles = None
    if coh_percentiles is not None:
        coh_percentiles = coh_percentiles * 1.0
        coherence_quantiles = {
            "q05": coh_percentiles[0],
            "q50": coh_percentiles[1],
            "q95": coh_percentiles[2],
        }
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
    """Cap draws, compute PSD quantiles, rescale, and build quantile dicts.

    Returns ``(psd_quantiles, coherence_quantiles, median_psd_real)``.
    """
    n_available = int(log_delta_samples.shape[0])
    max_cfg = int(getattr(config, "vi_psd_max_draws", 0) or 0)
    n_draws = min(n_available, max_cfg) if max_cfg > 0 else n_available
    n_draws = max(1, n_draws)
    if n_draws < n_available:
        logger.debug(
            f"Capping VI PSD reconstruction to {n_draws} draws "
            f"(limit={getattr(config, 'vi_psd_max_draws', 0)})."
        )
    log_delta_samples = log_delta_samples[:n_draws]
    theta_re_samples = theta_re_samples[:n_draws]
    theta_im_samples = theta_im_samples[:n_draws]

    psd_real_q, psd_imag_q, coh_percentiles = (
        spline_model.compute_psd_quantiles(
            log_delta_samples,
            theta_re_samples,
            theta_im_samples,
            percentiles=[5.0, 50.0, 95.0],
            n_samples_max=n_draws,
            compute_coherence=p > 1,
        )
    )
    psd_real_q = rescale_fn(psd_real_q)
    psd_imag_q = rescale_fn(psd_imag_q)
    psd_quantiles, coherence_quantiles = _build_psd_quantiles_dict(
        psd_real_q, psd_imag_q, coh_percentiles
    )
    return psd_quantiles, coherence_quantiles, psd_quantiles["real"]["q50"]


def _to_np(x) -> np.ndarray:
    """Move a JAX array to host as a NumPy array."""
    return np.asarray(jax.device_get(x))


def _to_np_dict(d: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert a dict of JAX arrays to NumPy arrays on host."""
    if not d:
        return {}
    return {name: _to_np(value) for name, value in d.items()}


def _extract_true_psd(
    sampler,
    rescale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Return the true PSD from config, optionally rescaled."""
    if sampler.config.true_psd is None:
        return None
    psd = _to_np(sampler.config.true_psd)
    if rescale_fn is not None:
        psd = rescale_fn(psd)
    return psd


def _coarse_vi_metadata(sampler) -> Dict[str, Any]:
    metadata = dict(getattr(sampler, "_coarse_vi_metadata", {}) or {})
    metadata.setdefault("coarse_vi_attempted", 0)
    metadata.setdefault("coarse_vi_success", 0)
    return metadata


def _mark_coarse_vi(
    diagnostics: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    *,
    attempted: bool,
    success: bool,
) -> Dict[str, Any]:
    out = dict(diagnostics or {})
    out.update(metadata)
    out["coarse_vi_attempted"] = int(bool(attempted))
    out["coarse_vi_success"] = int(bool(success))
    return out


def _strip_coarse_vi_plot_arrays(
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(diagnostics)
    for key in (
        "psd",
        "weights",
        "psd_quantiles",
        "psd_matrix",
        "psd_matrix_complex",
        "coherence_quantiles",
    ):
        out.pop(key, None)
    return out


def _median_vi_values(
    draws: Dict[str, jnp.ndarray],
    site_names: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, jnp.ndarray]]:
    """Return elementwise median across VI posterior draws."""
    if not draws:
        return None
    out: Dict[str, jnp.ndarray] = {}
    for name, value in draws.items():
        if site_names is not None and name not in site_names:
            continue
        arr = jnp.asarray(value)
        out[name] = jnp.median(arr, axis=0) if arr.ndim >= 1 else arr
    return out or None


def select_vi_or_default_init(
    *,
    vi_values: Optional[Dict[str, jnp.ndarray]],
    default_values: Dict[str, jnp.ndarray],
    log_posterior_fn: Callable[[Dict[str, jnp.ndarray]], float],
    log_prefix: str = "VI init",
):
    """Compare median VI init vs deterministic default; return the better one."""
    if vi_values is None:
        return init_to_value(values=default_values)

    vi_lp = float(log_posterior_fn(vi_values))
    det_lp = float(log_posterior_fn(default_values))
    delta_lp = vi_lp - det_lp
    delta_label = (
        f"improved by {delta_lp:.3f}"
        if np.isfinite(delta_lp) and delta_lp >= 0.0
        else f"worse by {abs(delta_lp):.3f}"
    )
    logger.info(
        f"{log_prefix}: VI log-post={vi_lp:.3f}, default log-post={det_lp:.3f} "
        f"({delta_label})"
    )
    if np.isfinite(vi_lp) and vi_lp > det_lp:
        return init_to_value(values=vi_values)
    return init_to_value(values=default_values)


def _validate_positive_finite_psd(psd: np.ndarray) -> bool:
    arr = np.asarray(psd, dtype=np.float64)
    return bool(arr.size and np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def _extract_multivar_design_psd(diagnostics: Optional[Dict[str, Any]]):
    if not diagnostics:
        return None
    design = diagnostics.get("psd_matrix_complex")
    if design is not None:
        return np.asarray(design, dtype=np.complex128)

    psd_quantiles = diagnostics.get("psd_quantiles")
    if psd_quantiles:
        real_q50 = (
            psd_quantiles.get("real", {}).get("q50")
            if isinstance(psd_quantiles, dict)
            else None
        )
        imag_q50 = (
            psd_quantiles.get("imag", {}).get("q50")
            if isinstance(psd_quantiles, dict)
            else None
        )
        if real_q50 is not None and imag_q50 is not None:
            return np.asarray(real_q50) + 1j * np.asarray(imag_q50)
    return None


def _ensure_positive_definite_psd(
    psd_matrix: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    arr = np.asarray(psd_matrix, dtype=np.complex128).copy()
    arr = 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))
    n_freq, n_chan, _ = arr.shape
    eye = np.eye(n_chan, dtype=np.complex128)

    for idx in range(n_freq):
        vals = np.linalg.eigvalsh(arr[idx]).real
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                "Encountered non-finite eigenvalues in PSD matrix"
            )
        min_eig = float(vals.min(initial=np.inf))
        if min_eig < eps:
            arr[idx] += (eps - min_eig + eps) * eye

    return arr


def _univar_model_args(sampler) -> tuple:
    """Build the positional args tuple for ``bayesian_model``."""
    return (
        sampler.log_pdgrm,
        sampler.basis_matrix,
        sampler.penalty_matrix,
        sampler.log_parametric,
        sampler.Nh,
        sampler.config.alpha_phi,
        sampler.config.beta_phi,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
    )


def _univar_default_init(sampler) -> Dict[str, jnp.ndarray]:
    """Build default init values for univariate VI/NUTS."""
    return default_init_values_univar(
        sampler.spline_model,
        alpha_phi=sampler.config.alpha_phi,
        beta_phi=sampler.config.beta_phi,
        alpha_delta=sampler.config.alpha_delta,
        beta_delta=sampler.config.beta_delta,
    )


def _build_univar_vi_diagnostics(sampler, vi_result) -> Dict[str, Any]:
    """Build PSD diagnostic dict from a univariate VI result."""
    scaling = _get_scaling_factor(
        getattr(sampler, "periodogram", None), sampler.config
    )
    weights = vi_result.means.get("weights")
    weights_np = None
    vi_psd = None
    psd_quantiles = None

    if weights is not None:
        weights_np = _to_np(weights)
        ln_psd = sampler.spline_model(vi_result.means["weights"])
        vi_psd = _to_np(jnp.exp(ln_psd)) * scaling

    if vi_result.samples is not None:
        weights_draws = vi_result.samples.get("weights")
        if weights_draws is not None and weights_draws.size:
            ln_psd_draws = jax.vmap(sampler.spline_model)(
                jnp.asarray(weights_draws)
            )
            psd_draws_np = (
                np.asarray(jax.device_get(jnp.exp(ln_psd_draws))) * scaling
            )
            q05, q50, q95 = np.percentile(psd_draws_np, [5, 50, 95], axis=0)
            psd_quantiles = {"q05": q05, "q50": q50, "q95": q95}

    diagnostics: Dict[str, Any] = {
        "weights": weights_np,
        "psd": vi_psd,
        "true_psd": _extract_true_psd(sampler),
    }
    if psd_quantiles is not None:
        diagnostics["psd_quantiles"] = psd_quantiles
    vi_samples = _to_np_dict(vi_result.samples)
    if vi_samples:
        diagnostics["vi_samples"] = vi_samples
    return diagnostics


def compute_vi_artifacts_univar(
    sampler,
    *,
    model: Callable[..., Any],
    init_values: Optional[Dict[str, jnp.ndarray]] = None,
) -> VIInitialisationArtifacts:
    """Run VI for univariate samplers and return initialisation artifacts."""

    guide_spec = sampler.config.vi_guide or suggest_guide_univar(
        sampler.n_weights + 2
    )

    def _postprocess(vi_result):
        means = {name: jnp.asarray(v) for name, v in vi_result.means.items()}
        diagnostics = _build_univar_vi_diagnostics(sampler, vi_result)
        return means, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=_univar_model_args(sampler),
        guide=guide_spec,
        init_values=init_values or _univar_default_init(sampler),
        postprocess=_postprocess,
    )


def compute_coarse_vi_artifacts_univar(
    sampler,
    *,
    coarse_sampler,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """VI-coarse → transfer → VI-fine for univariate models.

    1. Run standard VI on the coarse grid.
    2. Transfer the coarse PSD shape to fine-grid spline weights.
    3. Run standard VI on the fine grid, warm-started from the transfer.
    """
    metadata = _coarse_vi_metadata(sampler)
    coarse_artifacts = compute_vi_artifacts_univar(coarse_sampler, model=model)
    coarse_diag = coarse_artifacts.diagnostics or {}

    coarse_psd = coarse_diag.get("psd")
    if coarse_psd is None:
        coarse_psd = (coarse_diag.get("psd_quantiles") or {}).get("q50")

    if coarse_psd is None or not _validate_positive_finite_psd(coarse_psd):
        logger.warning(
            "Coarse-grid VI did not produce a valid PSD; "
            "falling back to standard fine-grid VI."
        )
        fine_artifacts = compute_vi_artifacts_univar(sampler, model=model)
        diagnostics = _mark_coarse_vi(
            fine_artifacts.diagnostics,
            metadata,
            attempted=True,
            success=False,
        )
        return VIInitialisationArtifacts(
            fine_artifacts.init_strategy,
            fine_artifacts.rng_key,
            _strip_coarse_vi_plot_arrays(diagnostics),
            means=fine_artifacts.means,
            posterior_draws=fine_artifacts.posterior_draws,
        )

    try:
        coarse_freq = np.asarray(coarse_sampler.periodogram.freqs, dtype=float)
        fine_freq = np.asarray(sampler.periodogram.freqs, dtype=float)

        coarse_means = coarse_artifacts.means or {}
        if "weights" not in coarse_means:
            raise ValueError("Coarse VI did not produce mean weights")

        fine_weights = transfer_univar_weights(
            coarse_weights=np.asarray(jax.device_get(coarse_means["weights"])),
            coarse_spline_model=coarse_sampler.spline_model,
            coarse_freq=coarse_freq,
            coarse_scaling=_get_scaling_factor(
                coarse_sampler.periodogram, coarse_sampler.config
            ),
            fine_spline_model=sampler.spline_model,
            fine_freq=fine_freq,
            fine_scaling=_get_scaling_factor(
                sampler.periodogram, sampler.config
            ),
        )
        transferred_init = dict(_univar_default_init(sampler))
        transferred_init["weights"] = jnp.asarray(fine_weights)

        coarse_only = bool(getattr(sampler.config, "vi_coarse_only", False))

        if coarse_only:
            # Skip fine-grid VI — use transferred weights directly.
            diagnostics = _mark_coarse_vi(
                coarse_diag,
                metadata,
                attempted=True,
                success=True,
            )
            diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
            diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
            diagnostics["coarse_vi_psd"] = np.asarray(coarse_psd)
            coarse_losses = coarse_diag.get("losses")
            if coarse_losses is not None:
                diagnostics["coarse_losses"] = np.asarray(coarse_losses)
            return VIInitialisationArtifacts(
                init_strategy=None,
                rng_key=coarse_artifacts.rng_key,
                diagnostics=diagnostics,
                means=coarse_artifacts.means,
                posterior_draws=coarse_artifacts.posterior_draws,
            )

        # Run fine-grid VI warm-started from transferred weights.
        fine_artifacts = compute_vi_artifacts_univar(
            sampler,
            model=model,
            init_values=transferred_init,
        )

        diagnostics = _mark_coarse_vi(
            fine_artifacts.diagnostics,
            metadata,
            attempted=True,
            success=True,
        )
        diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
        diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
        diagnostics["coarse_vi_psd"] = np.asarray(coarse_psd)
        coarse_losses = coarse_diag.get("losses")
        if coarse_losses is not None:
            diagnostics["coarse_losses"] = np.asarray(coarse_losses)

        return VIInitialisationArtifacts(
            fine_artifacts.init_strategy,
            fine_artifacts.rng_key,
            diagnostics,
            means=fine_artifacts.means,
            posterior_draws=fine_artifacts.posterior_draws,
        )
    except Exception as exc:
        logger.warning(
            f"Coarse-to-fine transfer failed ({exc}); "
            f"falling back to standard fine-grid VI."
        )
        fine_artifacts = compute_vi_artifacts_univar(sampler, model=model)
        diagnostics = _mark_coarse_vi(
            fine_artifacts.diagnostics,
            metadata,
            attempted=True,
            success=False,
        )
        return VIInitialisationArtifacts(
            fine_artifacts.init_strategy,
            fine_artifacts.rng_key,
            _strip_coarse_vi_plot_arrays(diagnostics),
            means=fine_artifacts.means,
            posterior_draws=fine_artifacts.posterior_draws,
        )


def _add_theta_init_values(
    *,
    init_values: Dict[str, jnp.ndarray],
    channel_index: int,
    theta_count: int,
    delta_init: jnp.ndarray,
    phi_theta_init: jnp.ndarray,
    theta_weights_re: jnp.ndarray,
    theta_weights_im: jnp.ndarray,
) -> None:
    """Populate per-theta initial values for blocked VI."""
    if theta_count <= 0:
        return
    delta_val = jnp.asarray(delta_init)
    log_phi_val = jnp.log(jnp.asarray(phi_theta_init))
    for part, weights in [("re", theta_weights_re), ("im", theta_weights_im)]:
        for theta_idx in range(theta_count):
            pfx = f"{channel_index}_{theta_idx}"
            init_values[f"delta_theta_{part}_{pfx}"] = delta_val
            init_values[f"phi_theta_{part}_{pfx}"] = log_phi_val
            init_values[f"weights_theta_{part}_{pfx}"] = weights


def _vi_weights_to_log_delta_theta(
    values: Dict[str, Any],
    sampler,
    *,
    is_batch: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert VI weight dicts to log_delta_sq / theta_re / theta_im tensors.

    Parameters
    ----------
    values : dict
        VI means (single vectors) or draws (batched). Keys include
        ``weights_delta_0`` and either legacy shared theta keys
        (``weights_theta_re``, ``weights_theta_im``) or per-component keys
        (``weights_theta_re_{j}_{l}``, ``weights_theta_im_{j}_{l}``).
    sampler :
        Multivariate sampler (provides ``all_bases``, ``p``, ``n_theta``, ``N``).
    is_batch : bool
        If *True*, weight arrays have shape ``(n_draws, n_basis)``; otherwise
        ``(n_basis,)`` and a leading dim is added.

    Returns
    -------
    log_delta_sq, theta_re, theta_im
        Tensors of shape ``(n_samples, N, p)`` / ``(n_samples, N, n_theta)``.
    """
    log_delta_terms: List[jnp.ndarray] = []
    for ch in range(sampler.p):
        w = jnp.asarray(values[f"weights_delta_{ch}"])
        basis = jnp.asarray(sampler.all_bases[ch])
        if is_batch:
            log_delta_terms.append(w @ basis.T)
        else:
            log_delta_terms.append(jnp.einsum("nk,k->n", basis, w))
    log_delta_sq = jnp.stack(log_delta_terms, axis=-1)
    if not is_batch:
        log_delta_sq = log_delta_sq[None, ...]

    n_samples = log_delta_sq.shape[0]
    if sampler.n_theta > 0:
        theta_re = jnp.zeros((n_samples, sampler.N, sampler.n_theta))
        theta_im = jnp.zeros_like(theta_re)

        # Backward compatibility: shared theta keys.
        if "weights_theta_re" in values and "weights_theta_im" in values:
            first_pair = sampler.spline_model.theta_pair_from_index(0)
            basis_theta = jnp.asarray(
                sampler.spline_model.get_theta_model(
                    "re", first_pair[0], first_pair[1]
                ).basis
            )
            w_re = jnp.asarray(values["weights_theta_re"])
            w_im = jnp.asarray(values["weights_theta_im"])
            if is_batch:
                tr_base = w_re @ basis_theta.T
                ti_base = w_im @ basis_theta.T
            else:
                tr_base = (jnp.einsum("nk,k->n", basis_theta, w_re))[None, :]
                ti_base = (jnp.einsum("nk,k->n", basis_theta, w_im))[None, :]
            theta_re = jnp.repeat(tr_base[:, :, None], sampler.n_theta, axis=2)
            theta_im = jnp.repeat(ti_base[:, :, None], sampler.n_theta, axis=2)
        else:
            for j in range(1, sampler.p):
                for l in range(j):
                    theta_idx = sampler.spline_model.theta_index(j, l)
                    key_re = f"weights_theta_re_{j}_{l}"
                    key_im = f"weights_theta_im_{j}_{l}"
                    if key_re not in values or key_im not in values:
                        continue

                    basis_re = jnp.asarray(
                        sampler.spline_model.get_theta_model("re", j, l).basis
                    )
                    basis_im = jnp.asarray(
                        sampler.spline_model.get_theta_model("im", j, l).basis
                    )
                    w_re = jnp.asarray(values[key_re])
                    w_im = jnp.asarray(values[key_im])
                    if is_batch:
                        theta_re_eval = w_re @ basis_re.T
                        theta_im_eval = w_im @ basis_im.T
                    else:
                        theta_re_eval = jnp.einsum("nk,k->n", basis_re, w_re)[
                            None, :
                        ]
                        theta_im_eval = jnp.einsum("nk,k->n", basis_im, w_im)[
                            None, :
                        ]
                    theta_re = theta_re.at[:, :, theta_idx].set(theta_re_eval)
                    theta_im = theta_im.at[:, :, theta_idx].set(theta_im_eval)
    else:
        theta_re = jnp.zeros((n_samples, sampler.N, 0))
        theta_im = jnp.zeros((n_samples, sampler.N, 0))

    return log_delta_sq, theta_re, theta_im


def _build_multivar_vi_diagnostics(sampler, vi_result) -> Dict[str, Any]:
    """Build PSD diagnostic dict from a multivariate VI result."""
    _rescale_psd = _make_psd_rescaler(sampler)
    vi_psd_np = None
    psd_quantiles = None
    coherence_quantiles = None

    try:
        log_delta_sq, theta_re, theta_im = _vi_weights_to_log_delta_theta(
            vi_result.means,
            sampler,
            is_batch=False,
        )
        vi_psd = sampler.spline_model.reconstruct_psd_matrix(
            log_delta_sq,
            theta_re,
            theta_im,
            n_samples_max=1,
        )[0]
        vi_psd_np = _rescale_psd(np.asarray(vi_psd))

        samples_tree = vi_result.samples or {}
        has_delta_draws = all(
            f"weights_delta_{ch}" in samples_tree for ch in range(sampler.p)
        )
        if samples_tree and has_delta_draws:
            ld_s, tr_s, ti_s = _vi_weights_to_log_delta_theta(
                samples_tree,
                sampler,
                is_batch=True,
            )
            psd_quantiles, coherence_quantiles, vi_psd_np = (
                _reconstruct_psd_quantiles_from_draws(
                    spline_model=sampler.spline_model,
                    config=sampler.config,
                    log_delta_samples=ld_s,
                    theta_re_samples=tr_s,
                    theta_im_samples=ti_s,
                    p=sampler.p,
                    rescale_fn=_rescale_psd,
                )
            )
    except (OverflowError, KeyError) as err:  # pragma: no cover
        if sampler.config.verbose:
            logger.warning(f"Could not build VI PSD diagnostics: {err}")

    true_psd = _extract_true_psd(sampler, rescale_fn=_rescale_psd)
    diagnostics: Dict[str, Any] = {
        "psd_matrix": vi_psd_np,
        "true_psd": true_psd,
    }
    if psd_quantiles is not None:
        diagnostics["psd_quantiles"] = psd_quantiles
    if coherence_quantiles is not None:
        diagnostics["coherence_quantiles"] = coherence_quantiles
    vi_samples = _to_np_dict(vi_result.samples)
    if vi_samples:
        diagnostics["vi_samples"] = vi_samples
    if true_psd is not None and vi_psd_np is not None:
        diagnostics.update(
            compute_multivar_riae_diagnostics(
                vi_psd_np,
                np.real(true_psd),
                np.asarray(sampler.freq, dtype=np.float64),
                psd_quantiles=psd_quantiles,
            )
        )
    return diagnostics


def compute_vi_artifacts_multivar(
    sampler,
    *,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI for fully coupled multivariate samplers."""

    total_latents = sum(
        m.n_basis + 2 for m in sampler.spline_model.diagonal_models
    )
    if sampler.n_theta > 0:
        for pair in sampler.spline_model.theta_pairs:
            total_latents += (
                sampler.spline_model.offdiag_re_models[pair].n_basis + 2
            )
            total_latents += (
                sampler.spline_model.offdiag_im_models[pair].n_basis + 2
            )
    guide_spec = sampler.config.vi_guide or suggest_guide_multivar(
        total_latents
    )

    def _postprocess(vi_result):
        init_values = {
            name: jnp.asarray(v) for name, v in vi_result.means.items()
        }
        diagnostics = _build_multivar_vi_diagnostics(sampler, vi_result)
        return init_values, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=(
            sampler.u_re,
            sampler.u_im,
            sampler.duration,
            sampler.Nb,
            sampler.all_bases,
            sampler.all_penalties,
            sampler.Nh,
            sampler.config.alpha_phi,
            sampler.config.beta_phi,
            sampler.config.alpha_delta,
            sampler.config.beta_delta,
        ),
        guide=guide_spec,
        init_values=default_init_values_multivar(
            sampler.spline_model,
            alpha_phi=sampler.config.alpha_phi,
            beta_phi=sampler.config.beta_phi,
            alpha_delta=sampler.config.alpha_delta,
            beta_delta=sampler.config.beta_delta,
        ),
        postprocess=_postprocess,
    )


@dataclass
class BlockVIArtifacts:
    """Container holding per-block VI outputs for blocked samplers."""

    init_strategies: List[Optional[Callable[[Any], Any]]]
    mcmc_keys: List[jax.Array]
    rng_key: jax.Array
    diagnostics: Optional[Dict[str, Any]]


def _prepare_block_accum(sampler) -> Dict[str, Any]:
    """Prepare mutable accumulator state for blocked VI initialisation.

    Combines draw buffers and diagnostic accumulators into a single dict
    consumed by :func:`_accumulate_block_vi_diagnostics` and
    :func:`_assemble_blocked_vi_diagnostics`.
    """
    posterior_draws = (
        sampler.config.vi_posterior_draws
        if getattr(sampler.config, "vi_posterior_draws", 0) > 0
        else 0
    )
    store_draws = sampler.config.init_from_vi and posterior_draws > 0
    log_delta_draws = None
    theta_re_draws = None
    theta_im_draws = None
    if store_draws:
        log_delta_draws = np.zeros(
            (posterior_draws, sampler.N, sampler.p),
            dtype=np.float32,
        )
        if sampler.n_theta > 0:
            theta_re_draws = np.zeros(
                (posterior_draws, sampler.N, sampler.n_theta),
                dtype=np.float32,
            )
            theta_im_draws = np.zeros_like(theta_re_draws)

    return {
        # Draw buffers
        "posterior_draws": posterior_draws,
        "store_draws": store_draws,
        "log_delta_draws": log_delta_draws,
        "theta_re_draws": theta_re_draws,
        "theta_im_draws": theta_im_draws,
        "draws_missing": False,
        "draws_recorded": 0,
        # Diagnostic accumulators
        "vi_samples": {},
        "vi_losses_blocks": [],
        "vi_guides": [],
        "vi_log_delta_means": [],
        "vi_theta_re_mean": None,
        "vi_theta_im_mean": None,
        "psis_khat_values": [],
        "psis_hyper_entries": [],
        "psis_weight_blocks": [],
        "psis_corr_summary": {},
        "psis_thresholds": None,
    }


def _aggregate_psis_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    psis_khat_values: List[float],
    psis_hyper_entries: List[Dict[str, Any]],
    psis_weight_blocks: List[Dict[str, Any]],
    psis_corr_summary: Dict[str, Any],
    psis_thresholds: Optional[Dict[str, float]],
    verbose: bool,
) -> None:
    """Merge per-block PSIS diagnostics into *diagnostics* in-place."""
    if psis_khat_values:
        khat_array = np.asarray(psis_khat_values, dtype=float)
        khat_max = float(np.nanmax(khat_array))
        status, status_msg = _interpret_khat(khat_max)
        diagnostics.update(
            {
                "psis_khat_per_block": khat_array,
                "psis_khat_max": khat_max,
                "psis_khat_status": status,
                "psis_status_message": status_msg,
                "psis_khat_threshold": 0.7,
                "psis_flag_warn": status in ("warn", "fail"),
                "psis_flag_critical": status == "fail",
            }
        )
        if verbose or status in ("warn", "fail"):
            logger.info(
                f"VI PSIS k-hat max (blocked) = {khat_max:.3f} ({status_msg})"
            )
        if status == "fail":
            logger.warning(
                "VI PSIS diagnostic (blocked) indicates poor posterior fit. "
                "Consider adjusting the guide or VI settings."
            )

    if psis_hyper_entries or psis_weight_blocks or psis_corr_summary:
        psis_moment_summary: Dict[str, Any] = {}
        if psis_hyper_entries:
            psis_moment_summary["hyperparameters"] = psis_hyper_entries
        if psis_weight_blocks:
            psis_moment_summary["weights_by_block"] = psis_weight_blocks
            _AGG_SPECS = {
                "var_ratio_min": np.nanmin,
                "var_ratio_median": np.nanmedian,
                "var_ratio_max": np.nanmax,
                "frac_outside": np.nanmax,
                "bias_median_abs": np.nanmedian,
                "bias_max_abs": np.nanmax,
                "bias_abs_median": np.nanmedian,
                "bias_abs_max": np.nanmax,
            }
            weights_agg = {
                k: float(fn([e.get(k, np.nan) for e in psis_weight_blocks]))
                for k, fn in _AGG_SPECS.items()
            }
            weights_agg["n_weights"] = int(
                np.sum([e.get("n_weights", 0) for e in psis_weight_blocks])
            )
            psis_moment_summary["weights"] = weights_agg
        if psis_thresholds:
            psis_moment_summary["thresholds"] = psis_thresholds
        if psis_moment_summary:
            diagnostics["psis_moment_summary"] = psis_moment_summary
        if psis_corr_summary:
            diagnostics["psis_correlation_summary"] = psis_corr_summary


def _build_block_model_args(
    sampler,
    channel_index: int,
    alpha_phi_theta: float,
    beta_phi_theta: float,
) -> tuple:
    """Build the positional args tuple for ``_blocked_channel_model``."""
    theta_re_basis, theta_re_penalty = (
        sampler._theta_component_arrays_for_channel(channel_index, part="re")
    )
    theta_im_basis, theta_im_penalty = (
        sampler._theta_component_arrays_for_channel(channel_index, part="im")
    )
    return (
        channel_index,
        sampler.u_re[:, channel_index, :],
        sampler.u_im[:, channel_index, :],
        sampler.u_re[:, :channel_index, :],
        sampler.u_im[:, :channel_index, :],
        sampler.all_bases[channel_index],
        sampler.all_penalties[channel_index],
        theta_re_basis,
        theta_re_penalty,
        theta_im_basis,
        theta_im_penalty,
        sampler.config.alpha_phi,
        sampler.config.beta_phi,
        alpha_phi_theta,
        beta_phi_theta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        sampler.duration,
        sampler.Nb,
        sampler.Nh,
    )


def _build_block_init_values(
    *,
    sampler,
    channel_index: int,
    theta_count: int,
    delta_weights_init: jnp.ndarray,
    alpha_phi_theta: float,
    beta_phi_theta: float,
) -> Dict[str, jnp.ndarray]:
    """Build initial values for a single blocked channel model."""
    alpha_phi_delta = sampler.config.alpha_phi
    beta_phi_delta = sampler.config.beta_phi
    delta_init, phi_delta_init = pspline_hyperparameter_initials(
        alpha_phi_delta,
        beta_phi_delta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        divide_phi_by_delta=True,
    )
    _, phi_theta_init = pspline_hyperparameter_initials(
        alpha_phi_theta,
        beta_phi_theta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        divide_phi_by_delta=True,
    )
    init_values = {
        f"delta_{channel_index}": jnp.asarray(delta_init),
        f"phi_delta_{channel_index}": jnp.log(jnp.asarray(phi_delta_init)),
        f"weights_delta_{channel_index}": delta_weights_init,
    }
    if theta_count > 0:
        for theta_idx in range(theta_count):
            re_model = sampler.spline_model.get_theta_model(
                "re", channel_index, theta_idx
            )
            im_model = sampler.spline_model.get_theta_model(
                "im", channel_index, theta_idx
            )
            pfx = f"{channel_index}_{theta_idx}"
            init_values[f"delta_theta_re_{pfx}"] = jnp.asarray(delta_init)
            init_values[f"phi_theta_re_{pfx}"] = jnp.log(
                jnp.asarray(phi_theta_init)
            )
            init_values[f"weights_theta_re_{pfx}"] = jnp.asarray(
                re_model.weights
            )
            init_values[f"delta_theta_im_{pfx}"] = jnp.asarray(delta_init)
            init_values[f"phi_theta_im_{pfx}"] = jnp.log(
                jnp.asarray(phi_theta_init)
            )
            init_values[f"weights_theta_im_{pfx}"] = jnp.asarray(
                im_model.weights
            )
    return init_values


def _run_single_block_vi(
    *,
    sampler,
    block_model: Callable[..., Any],
    vi_key: jax.Array,
    model_args: tuple[Any, ...],
    guide_spec: str,
    progress_bar: bool,
    init_values: Dict[str, jnp.ndarray],
):
    """Execute one VI run with a guarded retry for unstable ELBOs."""
    vi_result = fit_vi(
        model=block_model,
        rng_key=vi_key,
        vi_steps=sampler.config.vi_steps,
        optimizer_lr=sampler.config.vi_lr,
        model_args=model_args,
        guide=guide_spec,
        posterior_draws=sampler.config.vi_posterior_draws,
        progress_bar=progress_bar,
        init_values=init_values,
    )
    losses_arr = _to_np(vi_result.losses)
    if not (losses_arr.size and not np.isfinite(losses_arr[-1])):
        return vi_result, losses_arr

    logger.warning(
        f"VI returned a non-finite ELBO (guide={vi_result.guide_name}); retrying with diag guide."
    )
    vi_result = fit_vi(
        model=block_model,
        rng_key=vi_key,
        vi_steps=min(int(sampler.config.vi_steps), 2000),
        optimizer_lr=min(float(sampler.config.vi_lr), 1e-3),
        model_args=model_args,
        guide="diag",
        posterior_draws=sampler.config.vi_posterior_draws,
        progress_bar=progress_bar,
        init_values=init_values,
    )
    losses_arr = _to_np(vi_result.losses)
    return vi_result, losses_arr


def _accumulate_block_vi_diagnostics(
    *,
    sampler,
    vi_result,
    block_model,
    model_args: tuple,
    channel_index: int,
    theta_start: int,
    theta_count: int,
    delta_basis,
    accum: Dict[str, Any],
) -> None:
    """Accumulate per-block VI diagnostic data into *accum* (mutated in place).

    Collects: losses, guide name, VI samples, PSIS k-hat, log-delta means,
    theta means, draw buffers.
    """
    losses_arr = _to_np(vi_result.losses)
    accum["vi_losses_blocks"].append(losses_arr)
    accum["vi_guides"].append(vi_result.guide_name)
    accum["vi_samples"].update(_to_np_dict(vi_result.samples))

    # PSIS diagnostics
    psis_diag = _compute_psis_khat(
        model=block_model,
        model_args=model_args,
        model_kwargs={},
        guide=vi_result.guide,
        guide_params=vi_result.params,
        vi_samples=vi_result.samples,
        latent_samples=vi_result.latent_samples,
    )
    if psis_diag is not None:
        accum["psis_khat_values"].append(psis_diag["psis_khat_max"])
        moment_summary = psis_diag.get("psis_moment_summary") or {}
        if moment_summary and accum["psis_thresholds"] is None:
            accum["psis_thresholds"] = moment_summary.get("thresholds")
        for entry in moment_summary.get("hyperparameters") or []:
            new_entry = dict(entry)
            name = new_entry.get("param", "param")
            new_entry["param"] = f"block_{channel_index}:{name}"
            accum["psis_hyper_entries"].append(new_entry)
        weight_stats = moment_summary.get("weights")
        if weight_stats:
            accum["psis_weight_blocks"].append(
                {"block": channel_index, **weight_stats}
            )
        corr_summary = psis_diag.get("psis_correlation_summary") or {}
        for label, stats in corr_summary.items():
            if stats:
                accum["psis_corr_summary"][
                    f"{label}_block_{channel_index}"
                ] = stats

    # Log-delta mean from VI
    weights_delta_name = f"weights_delta_{channel_index}"
    weights_delta = vi_result.means.get(weights_delta_name)
    if weights_delta is not None:
        log_delta_vi = jnp.einsum("nk,k->n", delta_basis, weights_delta)
        accum["vi_log_delta_means"].append(_to_np(log_delta_vi))

    # Theta means
    if sampler.n_theta > 0 and theta_count > 0:
        for key in ("vi_theta_re_mean", "vi_theta_im_mean"):
            if accum[key] is None:
                accum[key] = np.zeros(
                    (sampler.N, sampler.n_theta), dtype=np.float32
                )

        theta_slice = slice(theta_start, theta_start + theta_count)
        for part, accum_key in (
            ("re", "vi_theta_re_mean"),
            ("im", "vi_theta_im_mean"),
        ):
            components = []
            for theta_idx in range(theta_count):
                basis = jnp.asarray(
                    sampler.spline_model.get_theta_model(
                        part, channel_index, theta_idx
                    ).basis
                )
                w = vi_result.means.get(
                    f"weights_theta_{part}_{channel_index}_{theta_idx}"
                )
                if w is None:
                    raise KeyError(
                        f"theta weights {part} {channel_index}_{theta_idx}"
                    )
                components.append(_to_np(jnp.einsum("nk,k->n", basis, w)))
            if components:
                accum[accum_key][:, theta_slice] = np.stack(components, axis=1)

    # Fill draw buffers
    if not accum["store_draws"] or vi_result.samples is None:
        return

    max_draws = accum["posterior_draws"]

    def _record_draws(weights_samples, basis, target_buf, col):
        """Project weight samples through basis and store in target buffer."""
        if weights_samples is None:
            accum["draws_missing"] = True
            return
        weights_samples = jnp.asarray(weights_samples)
        n = min(max_draws, weights_samples.shape[0])
        accum["draws_recorded"] = (
            n
            if accum["draws_recorded"] == 0
            else min(accum["draws_recorded"], n)
        )
        basis_jnp = jnp.asarray(basis, dtype=weights_samples.dtype)
        target_buf[:n, :, col] = _to_np(weights_samples[:n] @ basis_jnp.T)

    _record_draws(
        vi_result.samples.get(weights_delta_name),
        delta_basis,
        accum["log_delta_draws"],
        channel_index,
    )

    if (
        sampler.n_theta > 0
        and theta_count > 0
        and accum["theta_re_draws"] is not None
    ):
        for theta_idx in range(theta_count):
            prefix = f"{channel_index}_{theta_idx}"
            column = theta_start + theta_idx
            for part, buf_key in (
                ("re", "theta_re_draws"),
                ("im", "theta_im_draws"),
            ):
                basis = jnp.asarray(
                    sampler.spline_model.get_theta_model(
                        part, channel_index, theta_idx
                    ).basis
                )
                _record_draws(
                    vi_result.samples.get(f"weights_theta_{part}_{prefix}"),
                    basis,
                    accum[buf_key],
                    column,
                )


def _assemble_blocked_vi_diagnostics(
    *,
    sampler,
    accum: Dict[str, Any],
    _rescale_psd: Callable[[np.ndarray], np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Assemble final blocked-VI diagnostics from accumulated per-block data."""
    p = sampler.p
    diagnostics = None

    if sampler.config.init_from_vi and accum["vi_log_delta_means"]:
        vi_log_delta_means = accum["vi_log_delta_means"]
        log_delta_vi_np = (
            np.stack(vi_log_delta_means, axis=1)
            if len(vi_log_delta_means) == p
            else None
        )

        vi_psd_np = None
        vi_psd = None
        if log_delta_vi_np is not None:
            theta_re_vi = accum["vi_theta_re_mean"]
            theta_im_vi = accum["vi_theta_im_mean"]
            if sampler.n_theta > 0:
                assert theta_re_vi is not None and theta_im_vi is not None
            else:
                theta_re_vi = np.zeros((sampler.N, 0), dtype=np.float32)
                theta_im_vi = np.zeros((sampler.N, 0), dtype=np.float32)
            vi_psd = sampler.spline_model.reconstruct_psd_matrix(
                jnp.asarray(log_delta_vi_np)[None, ...],
                jnp.asarray(theta_re_vi)[None, ...],
                jnp.asarray(theta_im_vi)[None, ...],
                n_samples_max=1,
            )[0]
            vi_psd_np = _rescale_psd(np.asarray(vi_psd))

        psd_quantiles = None
        coherence_quantiles = None
        log_delta_draws = accum["log_delta_draws"]
        if (
            accum["store_draws"]
            and not accum["draws_missing"]
            and log_delta_draws is not None
        ):
            desired = accum["draws_recorded"] or accum["posterior_draws"]
            available = min(desired, log_delta_draws.shape[0])
            if available > 0:
                ld_s = jnp.asarray(
                    log_delta_draws[:available], dtype=jnp.float32
                )
                theta_re_draws = accum["theta_re_draws"]
                if sampler.n_theta > 0 and theta_re_draws is not None:
                    tr_s = jnp.asarray(
                        theta_re_draws[:available], dtype=jnp.float32
                    )
                    ti_s = jnp.asarray(
                        accum["theta_im_draws"][:available], dtype=jnp.float32
                    )
                else:
                    tr_s = jnp.zeros(
                        (available, sampler.N, 0), dtype=jnp.float32
                    )
                    ti_s = jnp.zeros_like(tr_s)
                logger.debug("Reconstructing PSD samples from VI draws...")
                psd_quantiles, coherence_quantiles, vi_psd_np = (
                    _reconstruct_psd_quantiles_from_draws(
                        spline_model=sampler.spline_model,
                        config=sampler.config,
                        log_delta_samples=ld_s,
                        theta_re_samples=tr_s,
                        theta_im_samples=ti_s,
                        p=p,
                        rescale_fn=_rescale_psd,
                    )
                )

        valid_losses = [
            arr
            for arr in accum["vi_losses_blocks"]
            if arr.size and np.all(np.isfinite(arr))
        ]
        losses_mean = None
        losses_stack = None
        if valid_losses:
            min_len = min(arr.shape[0] for arr in valid_losses)
            if min_len > 0:
                losses_stack = np.stack(
                    [arr[-min_len:] for arr in valid_losses], axis=0
                )
                losses_mean = losses_stack.mean(axis=0)

        vi_guides = accum["vi_guides"]
        guide_label = ",".join(sorted(set(vi_guides))) if vi_guides else "vi"
        diagnostics = {
            "losses": (
                losses_mean if losses_mean is not None else np.asarray([])
            ),
            "losses_per_block": losses_stack,
            "guide": guide_label,
            "psd_matrix": vi_psd_np,
            "psd_matrix_complex": (
                np.asarray(vi_psd, dtype=np.complex128)
                if log_delta_vi_np is not None
                else None
            ),
            "true_psd": _extract_true_psd(sampler),
        }
        if psd_quantiles is not None:
            diagnostics["psd_quantiles"] = psd_quantiles
        if coherence_quantiles is not None:
            diagnostics["coherence_quantiles"] = coherence_quantiles
        true_psd = diagnostics.get("true_psd")
        if true_psd is not None and vi_psd_np is not None:
            diagnostics.update(
                compute_multivar_riae_diagnostics(
                    vi_psd_np,
                    np.real(np.asarray(true_psd)),
                    np.asarray(sampler.freq, dtype=np.float64),
                    psd_quantiles=psd_quantiles,
                )
            )

    if accum["vi_samples"]:
        diagnostics = diagnostics or {}
        diagnostics["vi_samples"] = accum["vi_samples"]

    if diagnostics is not None and (
        accum["psis_khat_values"]
        or accum["psis_hyper_entries"]
        or accum["psis_weight_blocks"]
        or accum["psis_corr_summary"]
    ):
        _aggregate_psis_diagnostics(
            diagnostics,
            psis_khat_values=accum["psis_khat_values"],
            psis_hyper_entries=accum["psis_hyper_entries"],
            psis_weight_blocks=accum["psis_weight_blocks"],
            psis_corr_summary=accum["psis_corr_summary"],
            psis_thresholds=accum["psis_thresholds"],
            verbose=sampler.config.verbose,
        )

    return diagnostics


def _vi_means_are_usable(means: Dict[str, Any]) -> bool:
    """Return True if VI means are safe to use as init_to_value."""
    if not means:
        return False
    for name, value in means.items():
        try:
            arr = _to_np(value)
        except Exception:
            return False
        if arr.size and not np.all(np.isfinite(arr)):
            return False
        if str(name).startswith("delta_") and arr.size and not np.all(arr > 0):
            return False
    return True


def _block_site_names(channel_index: int) -> list[str]:
    """Return the VI sample site names for a single blocked channel."""
    names = [f"weights_delta_{channel_index}"]
    for theta_idx in range(channel_index):
        for prefix in ("theta_re", "theta_im"):
            names.append(f"weights_{prefix}_{channel_index}_{theta_idx}")
    return names


def prepare_block_vi(
    sampler,
    *,
    rng_key: jax.Array,
    block_model: Callable[..., Any],
    init_overrides: Optional[Dict[int, Dict[str, jnp.ndarray]]] = None,
) -> BlockVIArtifacts:
    """Run VI per block for blocked multivariate samplers.

    Parameters
    ----------
    init_overrides:
        Optional mapping ``{channel_index: {site_name: value}}`` providing
        warm-start values for VI.  When supplied, these are merged into the
        default init values so that VI starts from the transferred position
        (e.g. from a coarse-grid run).
    """

    guide_cfg = getattr(sampler.config, "vi_guide", None) or "auto(block)"
    steps = int(getattr(sampler.config, "vi_steps", 0) or 0)
    draws = int(getattr(sampler.config, "vi_posterior_draws", 0) or 0)
    if sampler.config.init_from_vi:
        logger.info(
            "Running VI initialisation per block "
            f"(p={sampler.p}, guide={guide_cfg}, steps={steps}, Lr={sampler.config.vi_lr}, posterior_draws={draws})..."
        )
    else:
        logger.info(
            f"Skipping VI initialisation per block (p={sampler.p}); using default NUTS initialisation."
        )
    p = sampler.p
    init_strategies: List[Optional[Callable[[Any], Any]]] = [None] * p
    mcmc_keys: List[jax.Array] = [jax.random.PRNGKey(0)] * p

    _rescale_psd = _make_psd_rescaler(sampler)
    accum = _prepare_block_accum(sampler)

    current_key = rng_key

    for channel_index in range(p):
        current_key, block_key = jax.random.split(current_key)

        if sampler.config.init_from_vi:
            vi_key, mcmc_key = jax.random.split(block_key)
        else:
            vi_key = None
            mcmc_key = block_key

        mcmc_keys[channel_index] = mcmc_key

        if not sampler.config.init_from_vi or vi_key is None:
            continue

        delta_basis = sampler.all_bases[channel_index]
        delta_penalty = sampler.all_penalties[channel_index]
        delta_model = sampler.spline_model.diagonal_models[channel_index]
        delta_weights_init = jnp.asarray(delta_model.weights)

        theta_start = channel_index * (channel_index - 1) // 2
        theta_count = channel_index
        theta_basis_cols = 0
        if theta_count > 0:
            theta_basis_cols = max(
                int(
                    sampler.spline_model.get_theta_model(
                        "re", channel_index, theta_idx
                    ).basis.shape[1]
                )
                for theta_idx in range(theta_count)
            )

        guide_spec = sampler.config.vi_guide or suggest_guide_block(
            delta_basis.shape[1], theta_count, theta_basis_cols
        )
        progress_bar = (
            sampler.config.vi_progress_bar
            if sampler.config.vi_progress_bar is not None
            else sampler.config.verbose
        )

        try:
            alpha_phi_theta = getattr(
                sampler.config, "alpha_phi_theta", sampler.config.alpha_phi
            )
            beta_phi_theta = getattr(
                sampler.config, "beta_phi_theta", sampler.config.beta_phi
            )

            init_values = _build_block_init_values(
                sampler=sampler,
                channel_index=channel_index,
                theta_count=theta_count,
                delta_weights_init=delta_weights_init,
                alpha_phi_theta=alpha_phi_theta,
                beta_phi_theta=beta_phi_theta,
            )
            if init_overrides and channel_index in init_overrides:
                init_values.update(init_overrides[channel_index])

            model_args = _build_block_model_args(
                sampler, channel_index, alpha_phi_theta, beta_phi_theta
            )
            block_log_posterior_fn = build_log_density_fn(
                partial(block_model, *model_args),
                {},
            )

            vi_result, losses_arr = _run_single_block_vi(
                sampler=sampler,
                block_model=block_model,
                vi_key=vi_key,
                model_args=model_args,
                guide_spec=guide_spec,
                progress_bar=progress_bar,
                init_values=init_values,
            )

            vi_means = {
                name: jnp.asarray(value)
                for name, value in (vi_result.means or {}).items()
            }
            default_init_values = dict(init_values)
            if losses_arr.size and not np.isfinite(losses_arr[-1]):
                logger.warning(
                    f"VI returned a non-finite ELBO for block {channel_index} "
                    f"(guide={vi_result.guide_name}); skipping VI-based init."
                )
            elif not _vi_means_are_usable(vi_means):
                logger.warning(
                    f"VI produced invalid mean parameters for block {channel_index} "
                    f"(guide={vi_result.guide_name}); skipping VI-based init."
                )
            else:
                vi_median = _median_vi_values(
                    draws={
                        name: jnp.asarray(value)
                        for name, value in (vi_result.samples or {}).items()
                    },
                )
                init_strategies[channel_index] = select_vi_or_default_init(
                    vi_values=vi_median,
                    default_values=default_init_values,
                    log_posterior_fn=lambda vals, fn=block_log_posterior_fn: float(
                        fn({k: jnp.asarray(v) for k, v in vals.items()})
                    ),
                    log_prefix=f"Blocked VI init channel {channel_index}",
                )

            _accumulate_block_vi_diagnostics(
                sampler=sampler,
                vi_result=vi_result,
                block_model=block_model,
                model_args=model_args,
                channel_index=channel_index,
                theta_start=theta_start,
                theta_count=theta_count,
                delta_basis=delta_basis,
                accum=accum,
            )

        except Exception as exc:  # pragma: no cover - defensive fallback
            if sampler.config.verbose:
                logger.warning(
                    f"VI block initialisation failed [channel {channel_index}]: {exc}"
                )
            if accum["store_draws"]:
                accum["draws_missing"] = True

    diagnostics = _assemble_blocked_vi_diagnostics(
        sampler=sampler,
        accum=accum,
        _rescale_psd=_rescale_psd,
    )

    return BlockVIArtifacts(
        init_strategies=init_strategies,
        mcmc_keys=mcmc_keys,
        rng_key=current_key,
        diagnostics=diagnostics,
    )


def prepare_coarse_block_vi(
    sampler,
    *,
    coarse_sampler,
    block_model: Callable[..., Any],
) -> BlockVIArtifacts:
    """Run blocked VI on a coarse grid, transfer to fine, then run standard VI.

    1. Standard ``prepare_block_vi`` on the coarse sampler.
    2. Transfer coarse VI means per channel to the fine grid via the bridge.
    3. Standard ``prepare_block_vi`` on the fine sampler, warm-started from
       the transferred values.
    """
    metadata = _coarse_vi_metadata(sampler)
    coarse_setup = prepare_block_vi(
        coarse_sampler,
        rng_key=sampler.rng_key,
        block_model=block_model,
    )
    diagnostics = _mark_coarse_vi(
        coarse_setup.diagnostics,
        metadata,
        attempted=True,
        success=False,
    )

    coarse_design = _extract_multivar_design_psd(diagnostics)
    if coarse_design is None:
        logger.warning(
            "Coarse blocked VI did not produce a valid PSD matrix warm "
            "start; using default init."
        )
        return BlockVIArtifacts(
            init_strategies=[None] * sampler.p,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=_strip_coarse_vi_plot_arrays(diagnostics),
        )

    try:
        fine_freq = np.asarray(sampler.freq, dtype=np.float64)
        coarse_freq = np.asarray(coarse_sampler.freq, dtype=np.float64)

        # Interpolate the full design PSD for diagnostics.
        fine_design = _interp_complex_matrix(
            coarse_freq,
            fine_freq,
            np.asarray(coarse_design, dtype=np.complex128),
        )
        fine_design = _ensure_positive_definite_psd(fine_design)

        # Transfer coarse VI means per channel to fine-grid init values.
        coarse_vi_samples = (coarse_setup.diagnostics or {}).get(
            "vi_samples"
        ) or {}
        init_overrides: Dict[int, Dict[str, jnp.ndarray]] = {}
        for channel_index in range(sampler.p):
            key = f"weights_delta_{channel_index}"
            if key not in coarse_vi_samples:
                continue
            coarse_means = _median_vi_values(
                draws={
                    name: jnp.asarray(value)
                    for name, value in coarse_vi_samples.items()
                },
                site_names=_block_site_names(channel_index),
            )
            if not coarse_means:
                continue
            default_vals = _build_block_init_values(
                sampler=sampler,
                channel_index=channel_index,
                theta_count=channel_index,
                delta_weights_init=jnp.asarray(
                    sampler.spline_model.diagonal_models[channel_index].weights
                ),
                alpha_phi_theta=getattr(
                    sampler.config,
                    "alpha_phi_theta",
                    sampler.config.alpha_phi,
                ),
                beta_phi_theta=getattr(
                    sampler.config,
                    "beta_phi_theta",
                    sampler.config.beta_phi,
                ),
            )
            transferred = transfer_block_init_values(
                draw_values=coarse_means,
                channel_index=channel_index,
                coarse_sampler=coarse_sampler,
                fine_sampler=sampler,
                coarse_freq=coarse_freq,
                fine_freq=fine_freq,
                default_init_values=default_vals,
            )
            init_overrides[channel_index] = transferred

        coarse_only = bool(getattr(sampler.config, "vi_coarse_only", False))

        if coarse_only:
            # Skip fine-grid VI — use coarse diagnostics directly.
            coarse_diag = coarse_setup.diagnostics or {}
            merged_diagnostics = dict(coarse_diag)
            merged_diagnostics.update(metadata)
            merged_diagnostics["coarse_vi_attempted"] = 1
            merged_diagnostics["coarse_vi_success"] = 1
            merged_diagnostics["psd_matrix_complex"] = fine_design
            merged_diagnostics["psd_matrix"] = np.real(fine_design)
            merged_diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
            merged_diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
            merged_diagnostics["coarse_vi_psd"] = np.real(
                np.asarray(coarse_design, dtype=np.complex128)
            )
            coarse_losses = coarse_diag.get("losses")
            if coarse_losses is not None:
                merged_diagnostics["coarse_losses"] = np.asarray(coarse_losses)
            return BlockVIArtifacts(
                init_strategies=[None] * sampler.p,
                mcmc_keys=coarse_setup.mcmc_keys,
                rng_key=coarse_setup.rng_key,
                diagnostics=merged_diagnostics,
            )

        # Run standard blocked VI on the fine sampler, seeded from the
        # transferred coarse values.
        fine_setup = prepare_block_vi(
            sampler,
            rng_key=coarse_setup.rng_key,
            block_model=block_model,
            init_overrides=init_overrides or None,
        )

        # Merge coarse diagnostics into the fine result.
        merged_diagnostics = dict(fine_setup.diagnostics or {})
        merged_diagnostics.update(metadata)
        merged_diagnostics["coarse_vi_attempted"] = 1
        merged_diagnostics["coarse_vi_success"] = 1
        merged_diagnostics["psd_matrix_complex"] = fine_design
        merged_diagnostics["psd_matrix"] = np.real(fine_design)
        merged_diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
        merged_diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
        merged_diagnostics["coarse_vi_psd"] = np.real(
            np.asarray(coarse_design, dtype=np.complex128)
        )
        coarse_diag = coarse_setup.diagnostics or {}
        coarse_losses = coarse_diag.get("losses")
        if coarse_losses is not None:
            merged_diagnostics["coarse_losses"] = np.asarray(coarse_losses)
        coarse_per_block = coarse_diag.get("losses_per_block")
        if coarse_per_block is not None:
            merged_diagnostics["coarse_losses_per_block"] = np.asarray(
                coarse_per_block
            )

        return BlockVIArtifacts(
            init_strategies=fine_setup.init_strategies,
            mcmc_keys=fine_setup.mcmc_keys,
            rng_key=fine_setup.rng_key,
            diagnostics=merged_diagnostics,
        )
    except Exception as exc:
        logger.warning(
            f"Could not transfer coarse blocked VI warm start to full "
            f"grid: {exc}"
        )
        return BlockVIArtifacts(
            init_strategies=[None] * sampler.p,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=_strip_coarse_vi_plot_arrays(diagnostics),
        )
