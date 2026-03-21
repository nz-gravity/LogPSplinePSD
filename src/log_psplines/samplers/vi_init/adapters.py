"""Adapters that encapsulate VI initialisation for samplers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer.util import init_to_uniform, init_to_value

from ...datatypes.multivar_utils import _interp_complex_matrix
from ...diagnostics.psd_compare import compute_multivar_riae_diagnostics
from ...logger import logger
from ...psplines.initialisation import init_weights
from ..pspline_block import (
    build_log_density_fn,
    pspline_hyperparameter_initials,
)
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



def _coarse_vi_refine_steps(config) -> int:
    return max(0, int(getattr(config, "coarse_vi_fine_refine_steps", 0) or 0))


def _coarse_vi_refine_guide(config) -> str:
    return str(getattr(config, "coarse_vi_fine_refine_guide", None) or "diag")


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
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[np.ndarray]]:
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


def _extract_vi_samples_np(vi_result) -> Dict[str, np.ndarray]:
    """Convert a ``VIResult.samples`` dict to NumPy arrays on host."""
    if vi_result.samples is None:
        return {}
    return {
        name: np.asarray(jax.device_get(value))
        for name, value in vi_result.samples.items()
    }


def _extract_true_psd(
    sampler,
    rescale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Return the true PSD from config, optionally rescaled."""
    if sampler.config.true_psd is None:
        return None
    psd = np.asarray(jax.device_get(sampler.config.true_psd))
    if rescale_fn is not None:
        psd = rescale_fn(psd)
    return psd


def _maybe_refine_vi_locally(
    *,
    sampler,
    model: Callable[..., Any],
    model_args: tuple[Any, ...],
    init_values: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    log_prefix: str,
) -> tuple[Optional[Any], jax.Array]:
    """Run a short local VI polish from a transferred coarse init."""
    refine_steps = _coarse_vi_refine_steps(sampler.config)
    if refine_steps <= 0:
        return None, rng_key

    refine_guide = _coarse_vi_refine_guide(sampler.config)
    progress_cfg = getattr(sampler.config, "vi_progress_bar", None)
    progress_bar = (
        bool(getattr(sampler.config, "verbose", False))
        if progress_cfg is None
        else bool(progress_cfg)
    )
    key_refine, key_out = jax.random.split(rng_key)
    try:
        logger.info(
            f"{log_prefix}: refining transferred init on fine model "
            f"(steps={refine_steps}, guide={refine_guide})."
        )
        vi_result = fit_vi(
            model=model,
            rng_key=key_refine,
            vi_steps=refine_steps,
            optimizer_lr=float(getattr(sampler.config, "vi_lr", 1e-2)),
            model_args=model_args,
            guide=refine_guide,
            posterior_draws=int(
                getattr(sampler.config, "vi_posterior_draws", 0) or 0
            ),
            progress_bar=progress_bar,
            init_values=init_values,
        )
        return vi_result, key_out
    except Exception as exc:
        logger.warning(
            f"{log_prefix}: fine-grid VI refinement failed ({exc}); using transferred coarse init."
        )
        return None, key_out


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


def _strip_coarse_vi_plot_arrays(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
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


def _median_sample_tree(
    sample_tree: Dict[str, Any],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, jnp.ndarray]:
    """Return elementwise medians for the requested VI sample sites."""
    if not sample_tree:
        return {}
    if names is None:
        names = sample_tree.keys()
    out: Dict[str, jnp.ndarray] = {}
    for name in names:
        value = sample_tree.get(name)
        if value is None:
            continue
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            out[name] = arr
        else:
            out[name] = jnp.median(arr, axis=0)
    return out


def _pick_from_draws_or_means(
    rng_key: Optional[jax.Array],
    *,
    draws: Dict[str, jnp.ndarray],
    means: Dict[str, jnp.ndarray],
    num_chains: int,
    anchor_name: str,
    site_names: Optional[Sequence[str]] = None,
) -> tuple[Optional[Dict[str, jnp.ndarray]], str]:
    """Pick a single candidate init from VI draws or means.

    For multi-chain: selects a random draw.  For single-chain: elementwise
    median.  Falls back to *means* when draws are unavailable.
    """
    if draws and anchor_name in draws:
        n_draws = int(jnp.asarray(draws[anchor_name]).shape[0])
        if num_chains > 1:
            idx = _select_vi_sample_index(rng_key, n_draws)
            selected = {
                name: jnp.asarray(value)[idx]
                for name, value in draws.items()
                if site_names is None or name in site_names
            }
            return selected, f"draw {idx}"
        return (
            _median_sample_tree(draws, names=site_names),
            "median",
        )
    if means:
        return means, "mean"
    return None, "unavailable"


def _select_vi_sample_index(rng_key: Optional[jax.Array], n_draws: int) -> int:
    if rng_key is None or n_draws <= 1:
        return 0
    idx = jax.random.randint(rng_key, (), 0, n_draws)
    return int(jax.device_get(idx))


def _extract_refined_artifacts(
    refined_vi_result: Optional[Any],
    site_filter: Optional[set[str]] = None,
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Extract draws and means as JAX arrays from a refined VI result."""
    if refined_vi_result is None:
        return {}, {}

    def _filter(items):
        return {
            name: jnp.asarray(value)
            for name, value in (items or {}).items()
            if site_filter is None or name in site_filter
        }

    return _filter(refined_vi_result.samples), _filter(refined_vi_result.means)


def _make_chainwise_init_strategy(
    *,
    first_site_name: str,
    default_values: Dict[str, jnp.ndarray],
    build_candidate: Callable[[Optional[jax.Array]], tuple[Optional[Dict[str, jnp.ndarray]], str]],
    log_posterior_fn: Optional[Callable[[Dict[str, jnp.ndarray]], float]] = None,
    log_prefix: str,
) -> Callable[[Any], Any]:
    """Create a per-chain init strategy with VI-vs-deterministic gating."""

    active_values: Optional[Dict[str, jnp.ndarray]] = None

    def _strategy(site=None):
        nonlocal active_values

        if site is None:
            return partial(_strategy)

        if site["type"] != "sample" or site["is_observed"]:
            return None

        if active_values is None or site["name"] == first_site_name:
            rng_key = site["kwargs"].get("rng_key")
            candidate_values, label = build_candidate(rng_key)
            selected_values = default_values

            if candidate_values is not None:
                use_candidate = True
                if log_posterior_fn is not None:
                    with numpyro.handlers.block():
                        det_log_post = float(log_posterior_fn(default_values))
                        vi_log_post = float(log_posterior_fn(candidate_values))
                    logger.info(
                        f"{log_prefix}: deterministic init log posterior={det_log_post:.3f}, "
                        f"VI init log posterior={vi_log_post:.3f} [{label}]"
                    )
                    use_candidate = (
                        np.isfinite(vi_log_post) and vi_log_post > det_log_post
                    )
                selected_values = candidate_values if use_candidate else default_values
            active_values = selected_values

        if site["name"] in active_values:
            return active_values[site["name"]]
        return init_to_uniform(site)

    return _strategy


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
            raise ValueError("Encountered non-finite eigenvalues in PSD matrix")
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
        weights_np = np.asarray(jax.device_get(weights))
        ln_psd = sampler.spline_model(vi_result.means["weights"])
        vi_psd = np.asarray(jax.device_get(jnp.exp(ln_psd))) * scaling

    if vi_result.samples is not None:
        weights_draws = vi_result.samples.get("weights")
        if weights_draws is not None and weights_draws.size:
            ln_psd_draws = jax.vmap(sampler.spline_model)(
                jnp.asarray(weights_draws)
            )
            psd_draws_np = np.asarray(
                jax.device_get(jnp.exp(ln_psd_draws))
            ) * scaling
            q05, q50, q95 = np.percentile(
                psd_draws_np, [5, 50, 95], axis=0
            )
            psd_quantiles = {"q05": q05, "q50": q50, "q95": q95}

    diagnostics: Dict[str, Any] = {
        "weights": weights_np,
        "psd": vi_psd,
        "true_psd": _extract_true_psd(sampler),
    }
    if psd_quantiles is not None:
        diagnostics["psd_quantiles"] = psd_quantiles
    vi_samples = _extract_vi_samples_np(vi_result)
    if vi_samples:
        diagnostics["vi_samples"] = vi_samples
    return diagnostics


def compute_vi_artifacts_univar(
    sampler,
    *,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI for univariate samplers and return initialisation artifacts."""

    guide_spec = sampler.config.vi_guide or suggest_guide_univar(
        sampler.n_weights + 2
    )

    def _postprocess(vi_result):
        init_values = {
            name: jnp.asarray(v) for name, v in vi_result.means.items()
        }
        diagnostics = _build_univar_vi_diagnostics(sampler, vi_result)
        return init_values, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=_univar_model_args(sampler),
        guide=guide_spec,
        init_values=_univar_default_init(sampler),
        postprocess=_postprocess,
    )


def compute_coarse_vi_artifacts_univar(
    sampler,
    *,
    coarse_sampler,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI on a coarse grid and transfer the PSD shape to fine-grid init."""

    metadata = _coarse_vi_metadata(sampler)
    coarse_artifacts = compute_vi_artifacts_univar(coarse_sampler, model=model)
    diagnostics = _mark_coarse_vi(
        coarse_artifacts.diagnostics,
        metadata,
        attempted=True,
        success=False,
    )

    coarse_psd = diagnostics.get("psd")
    if coarse_psd is None:
        psd_quantiles = diagnostics.get("psd_quantiles") or {}
        coarse_psd = psd_quantiles.get("q50")
    if coarse_psd is None or not _validate_positive_finite_psd(coarse_psd):
        logger.warning(
            "Coarse-grid VI did not produce a valid PSD warm start; using default init."
        )
        return VIInitialisationArtifacts(
            None,
            coarse_artifacts.rng_key,
            _strip_coarse_vi_plot_arrays(diagnostics),
        )

    try:
        coarse_freq = np.asarray(coarse_sampler.periodogram.freqs, dtype=float)
        fine_freq = np.asarray(sampler.periodogram.freqs, dtype=float)
        fine_scaling = _get_scaling_factor(sampler.periodogram, sampler.config)
        default_values = _univar_default_init(sampler)

        raw_draws = coarse_artifacts.posterior_draws or {}
        transformed_cache: Dict[str, Dict[str, jnp.ndarray]] = {}

        def _transform_candidate(draw_values: Dict[str, jnp.ndarray]):
            coarse_weights = np.asarray(
                jax.device_get(jnp.asarray(draw_values["weights"]))
            )
            coarse_psd_draw = np.exp(
                np.asarray(
                    jax.device_get(coarse_sampler.spline_model(jnp.asarray(coarse_weights)))
                )
            )
            coarse_psd_draw *= _get_scaling_factor(
                coarse_sampler.periodogram, coarse_sampler.config
            )
            interp_psd = np.interp(fine_freq, coarse_freq, coarse_psd_draw)
            interp_model_psd = np.maximum(interp_psd / fine_scaling, 1e-12)
            fine_weights = init_weights(
                jnp.asarray(np.array(np.log(interp_model_psd), copy=True)),
                sampler.spline_model,
            )
            candidate = dict(default_values)
            candidate["weights"] = jnp.asarray(fine_weights)
            return candidate

        def _build_transferred_candidate(rng_key: Optional[jax.Array]):
            if raw_draws and "weights" in raw_draws:
                n_draws = int(jnp.asarray(raw_draws["weights"]).shape[0])
                if int(sampler.config.num_chains) > 1:
                    idx = _select_vi_sample_index(rng_key, n_draws)
                    label = f"draw {idx}"
                else:
                    idx = -1
                    label = "median"
                cache_key = str(idx)
                if cache_key not in transformed_cache:
                    if idx >= 0:
                        draw_values = {
                            name: jnp.asarray(value)[idx]
                            for name, value in raw_draws.items()
                            if name == "weights"
                        }
                    else:
                        draw_values = _median_sample_tree(raw_draws, names=("weights",))
                    transformed_cache[cache_key] = _transform_candidate(
                        draw_values
                    )
                return transformed_cache[cache_key], label

            means = coarse_artifacts.means or {}
            if "weights" not in means:
                return None, "unavailable"
            return (
                _transform_candidate(
                    {
                        "weights": jnp.asarray(means["weights"]),
                    }
                ),
                "mean",
            )

        seed_candidate, seed_label = _build_transferred_candidate(None)
        refined_vi_result = None
        next_rng_key = coarse_artifacts.rng_key
        if seed_candidate is not None:
            refined_vi_result, next_rng_key = _maybe_refine_vi_locally(
                sampler=sampler,
                model=model,
                model_args=_univar_model_args(sampler),
                init_values=seed_candidate,
                rng_key=coarse_artifacts.rng_key,
                log_prefix="Univariate coarse-VI init",
            )

        _univar_sites = {"weights", "phi", "delta"}
        refined_draws, refined_means = _extract_refined_artifacts(
            refined_vi_result, site_filter=_univar_sites
        )

        def _build_candidate(rng_key: Optional[jax.Array]):
            result, label = _pick_from_draws_or_means(
                rng_key,
                draws=refined_draws,
                means=refined_means,
                num_chains=int(sampler.config.num_chains),
                anchor_name="weights",
                site_names=_univar_sites,
            )
            if result is not None:
                return result, f"refined {label}"
            return _build_transferred_candidate(rng_key)

        def _log_posterior(values: Dict[str, jnp.ndarray]) -> float:
            params = {
                "weights": jnp.asarray(values["weights"]),
                "phi": jnp.asarray(values["phi"]),
                "delta": jnp.asarray(values["delta"]),
            }
            return float(sampler._logpost_fn(params))

        init_strategy = _make_chainwise_init_strategy(
            first_site_name="delta",
            default_values=default_values,
            build_candidate=_build_candidate,
            log_posterior_fn=_log_posterior,
            log_prefix="Univariate coarse-VI init",
        )
        candidate_preview, _ = _build_candidate(None)
        fine_psd = None
        if candidate_preview is not None:
            fine_psd = np.array(
                jax.device_get(
                    jnp.exp(sampler.spline_model(candidate_preview["weights"]))
                ),
                copy=True,
            )
            fine_psd *= fine_scaling

        diagnostics = dict(diagnostics)
        if candidate_preview is not None:
            diagnostics["weights"] = np.asarray(
                jax.device_get(candidate_preview["weights"])
            )
            diagnostics["psd"] = fine_psd
            diagnostics["coarse_vi_success"] = 1
        if refined_vi_result is not None:
            diagnostics["coarse_vi_fine_refine_steps"] = _coarse_vi_refine_steps(
                sampler.config
            )
            diagnostics["coarse_vi_fine_refine_guide"] = (
                refined_vi_result.guide_name
            )
        diagnostics.pop("psd_quantiles", None)
        diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
        return VIInitialisationArtifacts(
            init_strategy,
            next_rng_key,
            diagnostics,
            means=candidate_preview,
        )
    except Exception as exc:
        logger.warning(
            f"Could not transfer coarse-grid VI warm start to full grid: {exc}"
        )
        return VIInitialisationArtifacts(
            None,
            coarse_artifacts.rng_key,
            _strip_coarse_vi_plot_arrays(diagnostics),
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
        VI means (single vectors) or draws (batched).  Keys like
        ``weights_delta_0``, ``weights_theta_re``, ``weights_theta_im``.
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
        basis_theta = jnp.asarray(sampler.all_bases[sampler.p])
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
            vi_result.means, sampler, is_batch=False,
        )
        vi_psd = sampler.spline_model.reconstruct_psd_matrix(
            log_delta_sq, theta_re, theta_im, n_samples_max=1,
        )[0]
        vi_psd_np = _rescale_psd(np.asarray(vi_psd))

        samples_tree = vi_result.samples or {}
        has_delta_draws = all(
            f"weights_delta_{ch}" in samples_tree for ch in range(sampler.p)
        )
        if samples_tree and has_delta_draws:
            ld_s, tr_s, ti_s = _vi_weights_to_log_delta_theta(
                samples_tree, sampler, is_batch=True,
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
    vi_samples = _extract_vi_samples_np(vi_result)
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

    total_latents = sum(m.n_basis + 2 for m in sampler.spline_model.diagonal_models)
    if sampler.n_theta > 0:
        total_latents += sampler.spline_model.offdiag_re_model.n_basis + 2
        total_latents += sampler.spline_model.offdiag_im_model.n_basis + 2
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
            (posterior_draws, sampler.N, sampler.p), dtype=np.float32,
        )
        if sampler.n_theta > 0:
            theta_re_draws = np.zeros(
                (posterior_draws, sampler.N, sampler.n_theta), dtype=np.float32,
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


def _blocked_channel_log_posterior_from_values(
    *,
    log_posterior_fn: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
    values: Dict[str, jnp.ndarray],
) -> float:
    """Evaluate the blocked-channel NumPyro log posterior at a candidate init point."""
    params = {name: jnp.asarray(value) for name, value in values.items()}
    return float(log_posterior_fn(params))


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

            def _agg(key, fn=np.nanmedian):
                return float(fn([e.get(key, np.nan) for e in psis_weight_blocks]))

            psis_moment_summary["weights"] = {
                "var_ratio_min": _agg("var_ratio_min", np.nanmin),
                "var_ratio_median": _agg("var_ratio_median"),
                "var_ratio_max": _agg("var_ratio_max", np.nanmax),
                "frac_outside": _agg("frac_outside", np.nanmax),
                "bias_median_abs": _agg("bias_median_abs"),
                "bias_max_abs": _agg("bias_max_abs", np.nanmax),
                "bias_abs_median": _agg("bias_abs_median"),
                "bias_abs_max": _agg("bias_abs_max", np.nanmax),
                "n_weights": int(np.sum([e.get("n_weights", 0) for e in psis_weight_blocks])),
            }
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
    return (
        channel_index,
        sampler.u_re[:, channel_index, :],
        sampler.u_im[:, channel_index, :],
        sampler.u_re[:, :channel_index, :],
        sampler.u_im[:, :channel_index, :],
        sampler.all_bases[channel_index],
        sampler.all_penalties[channel_index],
        sampler._theta_basis,
        sampler._theta_penalty,
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
        theta_weights_re = jnp.asarray(
            sampler.spline_model.offdiag_re_model.weights
        )
        theta_weights_im = jnp.asarray(
            sampler.spline_model.offdiag_im_model.weights
        )
        _add_theta_init_values(
            init_values=init_values,
            channel_index=channel_index,
            theta_count=theta_count,
            delta_init=jnp.asarray(delta_init),
            phi_theta_init=jnp.asarray(phi_theta_init),
            theta_weights_re=theta_weights_re,
            theta_weights_im=theta_weights_im,
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
    losses_arr = np.asarray(jax.device_get(vi_result.losses))
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
    losses_arr = np.asarray(jax.device_get(vi_result.losses))
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
    losses_arr = np.asarray(jax.device_get(vi_result.losses))
    accum["vi_losses_blocks"].append(losses_arr)
    accum["vi_guides"].append(vi_result.guide_name)
    accum["vi_samples"].update(_extract_vi_samples_np(vi_result))

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
        accum["vi_log_delta_means"].append(
            np.asarray(jax.device_get(log_delta_vi))
        )

    # Theta means
    if sampler.n_theta > 0 and theta_count > 0:
        if accum["vi_theta_re_mean"] is None:
            accum["vi_theta_re_mean"] = np.zeros(
                (sampler.N, sampler.n_theta), dtype=np.float32
            )
            accum["vi_theta_im_mean"] = np.zeros(
                (sampler.N, sampler.n_theta), dtype=np.float32
            )

        theta_re_components: List[np.ndarray] = []
        theta_im_components: List[np.ndarray] = []
        for theta_idx in range(theta_count):
            prefix = f"{channel_index}_{theta_idx}"
            w_re = vi_result.means.get(f"weights_theta_re_{prefix}")
            w_im = vi_result.means.get(f"weights_theta_im_{prefix}")
            if w_re is None or w_im is None:
                raise KeyError(f"theta weights {prefix}")
            theta_re_components.append(
                np.asarray(
                    jax.device_get(
                        jnp.einsum("nk,k->n", sampler._theta_basis, w_re)
                    )
                )
            )
            theta_im_components.append(
                np.asarray(
                    jax.device_get(
                        jnp.einsum("nk,k->n", sampler._theta_basis, w_im)
                    )
                )
            )
        if theta_re_components:
            theta_re_block = np.stack(theta_re_components, axis=1)
            theta_im_block = np.stack(theta_im_components, axis=1)
        else:
            theta_re_block = np.zeros((sampler.N, theta_count))
            theta_im_block = np.zeros((sampler.N, theta_count))
        theta_slice = slice(theta_start, theta_start + theta_count)
        accum["vi_theta_re_mean"][:, theta_slice] = theta_re_block
        accum["vi_theta_im_mean"][:, theta_slice] = theta_im_block

    # Fill draw buffers
    store_draws = accum["store_draws"]
    posterior_draws = accum["posterior_draws"]
    if store_draws and vi_result.samples is not None:
        weights_delta_samples = vi_result.samples.get(weights_delta_name)
        log_delta_draws = accum["log_delta_draws"]
        if weights_delta_samples is not None:
            weights_delta_samples = jnp.asarray(weights_delta_samples)
            draw_count = min(posterior_draws, weights_delta_samples.shape[0])
            accum["draws_recorded"] = (
                draw_count
                if accum["draws_recorded"] == 0
                else min(accum["draws_recorded"], draw_count)
            )
            delta_basis_jnp = jnp.asarray(
                delta_basis, dtype=weights_delta_samples.dtype
            )
            log_delta_samples = (
                weights_delta_samples[:draw_count] @ delta_basis_jnp.T
            )
            log_delta_draws[:draw_count, :, channel_index] = np.asarray(
                jax.device_get(log_delta_samples)
            )
        else:
            accum["draws_missing"] = True

        theta_re_draws = accum["theta_re_draws"]
        if (
            sampler.n_theta > 0
            and theta_count > 0
            and theta_re_draws is not None
        ):
            theta_basis_jnp = jnp.asarray(
                sampler._theta_basis, dtype=jnp.float32
            )
            theta_im_draws = accum["theta_im_draws"]
            for theta_idx in range(theta_count):
                prefix = f"{channel_index}_{theta_idx}"
                w_re_s = vi_result.samples.get(f"weights_theta_re_{prefix}")
                w_im_s = vi_result.samples.get(f"weights_theta_im_{prefix}")
                if w_re_s is None or w_im_s is None:
                    accum["draws_missing"] = True
                    continue
                w_re_s = jnp.asarray(w_re_s)
                w_im_s = jnp.asarray(w_im_s)
                theta_basis_jnp = jnp.asarray(
                    sampler._theta_basis, dtype=w_re_s.dtype
                )
                draw_count = min(posterior_draws, w_re_s.shape[0])
                accum["draws_recorded"] = (
                    draw_count
                    if accum["draws_recorded"] == 0
                    else min(accum["draws_recorded"], draw_count)
                )
                column = theta_start + theta_idx
                theta_re_draws[:draw_count, :, column] = np.asarray(
                    jax.device_get(w_re_s[:draw_count] @ theta_basis_jnp.T)
                )
                theta_im_draws[:draw_count, :, column] = np.asarray(
                    jax.device_get(w_im_s[:draw_count] @ theta_basis_jnp.T)
                )


def _assemble_blocked_vi_diagnostics(
    *,
    sampler,
    accum: Dict[str, Any],
    _rescale_psd: Callable[[np.ndarray], np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Assemble final blocked-VI diagnostics from accumulated per-block data."""
    p = sampler.p
    vi_log_delta_means = accum["vi_log_delta_means"]
    vi_theta_re_mean = accum["vi_theta_re_mean"]
    vi_theta_im_mean = accum["vi_theta_im_mean"]
    vi_losses_blocks = accum["vi_losses_blocks"]
    vi_guides = accum["vi_guides"]
    vi_samples = accum["vi_samples"]
    store_draws = accum["store_draws"]
    draws_missing = accum["draws_missing"]
    draws_recorded = accum["draws_recorded"]
    posterior_draws = accum["posterior_draws"]
    log_delta_draws = accum["log_delta_draws"]
    theta_re_draws = accum["theta_re_draws"]
    theta_im_draws = accum["theta_im_draws"]
    psis_khat_values = accum["psis_khat_values"]
    psis_hyper_entries = accum["psis_hyper_entries"]
    psis_weight_blocks = accum["psis_weight_blocks"]
    psis_corr_summary = accum["psis_corr_summary"]
    psis_thresholds = accum["psis_thresholds"]
    diagnostics = None

    if sampler.config.init_from_vi and vi_log_delta_means:
        log_delta_vi_np = (
            np.stack(vi_log_delta_means, axis=1)
            if len(vi_log_delta_means) == p
            else None
        )

        vi_psd_np = None
        vi_psd = None
        if log_delta_vi_np is not None:
            if sampler.n_theta > 0:
                assert vi_theta_re_mean is not None and vi_theta_im_mean is not None
                theta_re_vi_np = vi_theta_re_mean
                theta_im_vi_np = vi_theta_im_mean
            else:
                theta_re_vi_np = np.zeros((sampler.N, 0), dtype=np.float32)
                theta_im_vi_np = np.zeros((sampler.N, 0), dtype=np.float32)
            vi_psd = sampler.spline_model.reconstruct_psd_matrix(
                jnp.asarray(log_delta_vi_np)[None, ...],
                jnp.asarray(theta_re_vi_np)[None, ...],
                jnp.asarray(theta_im_vi_np)[None, ...],
                n_samples_max=1,
            )[0]
            vi_psd_np = _rescale_psd(np.asarray(vi_psd))

        psd_quantiles = None
        coherence_quantiles = None
        if store_draws and not draws_missing and log_delta_draws is not None:
            desired = draws_recorded or posterior_draws
            available = min(desired, log_delta_draws.shape[0])
            if available > 0:
                ld_s = jnp.asarray(
                    log_delta_draws[:available], dtype=jnp.float32
                )
                if sampler.n_theta > 0 and theta_re_draws is not None:
                    tr_s = jnp.asarray(
                        theta_re_draws[:available], dtype=jnp.float32
                    )
                    ti_s = jnp.asarray(
                        theta_im_draws[:available], dtype=jnp.float32
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
            for arr in vi_losses_blocks
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

        guide_label = ",".join(sorted(set(vi_guides))) if vi_guides else "vi"
        diagnostics = {
            "losses": losses_mean if losses_mean is not None else np.asarray([]),
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

    if vi_samples:
        diagnostics = diagnostics or {}
        diagnostics["vi_samples"] = vi_samples

    if diagnostics is not None and (
        psis_khat_values or psis_hyper_entries or psis_weight_blocks or psis_corr_summary
    ):
        _aggregate_psis_diagnostics(
            diagnostics,
            psis_khat_values=psis_khat_values,
            psis_hyper_entries=psis_hyper_entries,
            psis_weight_blocks=psis_weight_blocks,
            psis_corr_summary=psis_corr_summary,
            psis_thresholds=psis_thresholds,
            verbose=sampler.config.verbose,
        )

    return diagnostics


def _vi_means_are_usable(means: Dict[str, Any]) -> bool:
    """Return True if VI means are safe to use as init_to_value."""
    if not means:
        return False
    for name, value in means.items():
        try:
            arr = np.asarray(jax.device_get(value))
        except Exception:
            return False
        if arr.size and not np.all(np.isfinite(arr)):
            return False
        if str(name).startswith("delta_") and arr.size and not np.all(arr > 0):
            return False
    return True


def prepare_block_vi(
    sampler,
    *,
    rng_key: jax.Array,
    block_model: Callable[..., Any],
) -> BlockVIArtifacts:
    """Run VI per block for blocked multivariate samplers."""

    guide_cfg = getattr(sampler.config, "vi_guide", None) or "auto(block)"
    steps = int(getattr(sampler.config, "vi_steps", 0) or 0)
    draws = int(getattr(sampler.config, "vi_posterior_draws", 0) or 0)
    logger.info(
        "Running VI initialisation per block "
        f"(p={sampler.p}, guide={guide_cfg}, steps={steps}, Lr={sampler.config.vi_lr}, posterior_draws={draws})..."
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

        guide_spec = sampler.config.vi_guide or suggest_guide_block(
            delta_basis.shape[1], theta_count, sampler._theta_basis.shape[1]
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
                draw_tree = {
                    name: jnp.asarray(value)
                    for name, value in (vi_result.samples or {}).items()
                }

                def _build_candidate(
                    rng_key: Optional[jax.Array],
                    *,
                    draw_values: Dict[str, jax.Array] = draw_tree,
                    means_values: Dict[str, jax.Array] = vi_means,
                    anchor: str = f"weights_delta_{channel_index}",
                ):
                    return _pick_from_draws_or_means(
                        rng_key,
                        draws=draw_values,
                        means=means_values,
                        num_chains=int(sampler.config.num_chains),
                        anchor_name=anchor,
                    )

                init_strategies[channel_index] = _make_chainwise_init_strategy(
                    first_site_name=f"delta_{channel_index}",
                    default_values=default_init_values,
                    build_candidate=_build_candidate,
                    log_posterior_fn=lambda vals, logpost_fn=block_log_posterior_fn: _blocked_channel_log_posterior_from_values(
                        log_posterior_fn=logpost_fn,
                        values=vals,
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
        sampler=sampler, accum=accum, _rescale_psd=_rescale_psd,
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
    """Run blocked VI on a coarse grid and transfer design weights to fine init."""

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
            "Coarse blocked VI did not produce a valid PSD matrix warm start; using default init."
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
        fine_design = _interp_complex_matrix(
            coarse_freq,
            fine_freq,
            np.asarray(coarse_design, dtype=np.complex128),
        )
        fine_design = _ensure_positive_definite_psd(fine_design)

        diagnostics = dict(diagnostics)
        diagnostics["psd_matrix_complex"] = fine_design
        diagnostics["psd_matrix"] = np.real(fine_design)
        diagnostics.pop("psd_quantiles", None)
        diagnostics.pop("coherence_quantiles", None)
        diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
        diagnostics["coarse_vi_success"] = 1

        coarse_draws = {
            name: jnp.asarray(value)
            for name, value in ((coarse_setup.diagnostics or {}).get("vi_samples") or {}).items()
        }
        init_strategies: List[Optional[Callable[[Any], Any]]] = [None] * sampler.p
        refine_rng_key = coarse_setup.rng_key

        for channel_index in range(sampler.p):
            theta_count = channel_index
            alpha_phi_theta = getattr(
                sampler.config,
                "alpha_phi_theta",
                sampler.config.alpha_phi,
            )
            beta_phi_theta = getattr(
                sampler.config,
                "beta_phi_theta",
                sampler.config.beta_phi,
            )
            default_init_values = _build_block_init_values(
                sampler=sampler,
                channel_index=channel_index,
                theta_count=theta_count,
                delta_weights_init=jnp.asarray(
                    sampler.spline_model.diagonal_models[channel_index].weights
                ),
                alpha_phi_theta=alpha_phi_theta,
                beta_phi_theta=beta_phi_theta,
            )
            block_model_args = _build_block_model_args(
                sampler, channel_index, alpha_phi_theta, beta_phi_theta
            )
            block_log_posterior_fn = build_log_density_fn(
                partial(block_model, *block_model_args),
                {},
            )

            transform_cache: Dict[str, Dict[str, jnp.ndarray]] = {}
            site_names = [
                f"weights_delta_{channel_index}",
            ]
            for theta_idx in range(channel_index):
                for prefix in ("theta_re", "theta_im"):
                    site_names.extend(
                        [
                            f"weights_{prefix}_{channel_index}_{theta_idx}",
                        ]
                    )

            def _transform_candidate(
                draw_values: Dict[str, jnp.ndarray],
                *,
                ch: int = channel_index,
                default_vals: Dict[str, jnp.ndarray] = default_init_values,
            ) -> Dict[str, jnp.ndarray]:
                candidate = dict(default_vals)
                fine_diag_model = sampler.spline_model.diagonal_models[ch]
                log_delta_coarse = np.asarray(
                    jax.device_get(
                        jnp.einsum(
                            "nk,k->n",
                            coarse_sampler.all_bases[ch],
                            jnp.asarray(draw_values[f"weights_delta_{ch}"]),
                        )
                    )
                )
                log_delta_fine = np.interp(fine_freq, coarse_freq, log_delta_coarse)
                candidate[f"weights_delta_{ch}"] = init_weights(
                    jnp.asarray(np.array(log_delta_fine, copy=True)),
                    fine_diag_model,
                )

                if ch > 0:
                    for theta_idx in range(ch):
                        for prefix, fine_model in (
                            ("theta_re", sampler.spline_model.offdiag_re_model),
                            ("theta_im", sampler.spline_model.offdiag_im_model),
                        ):
                            w_key = f"weights_{prefix}_{ch}_{theta_idx}"
                            eval_coarse = np.asarray(
                                jax.device_get(
                                    jnp.einsum(
                                        "nk,k->n",
                                        coarse_sampler._theta_basis,
                                        jnp.asarray(draw_values[w_key]),
                                    )
                                )
                            )
                            eval_fine = np.interp(
                                fine_freq, coarse_freq, eval_coarse
                            )
                            candidate[w_key] = init_weights(
                                jnp.asarray(np.array(eval_fine, copy=True)),
                                fine_model,
                            )
                return candidate

            def _build_candidate(
                rng_key: Optional[jax.Array],
                *,
                ch: int = channel_index,
                names: list[str] = list(site_names),
                cache: Dict[str, Dict[str, jnp.ndarray]] = transform_cache,
                default_vals: Dict[str, jnp.ndarray] = default_init_values,
            ):
                if coarse_draws and f"weights_delta_{ch}" in coarse_draws:
                    n_draws = int(coarse_draws[f"weights_delta_{ch}"].shape[0])
                    if int(sampler.config.num_chains) > 1:
                        idx = _select_vi_sample_index(rng_key, n_draws)
                        label = f"draw {idx}"
                    else:
                        idx = -1
                        label = "median"
                    cache_key = f"{ch}:{idx}"
                    if cache_key not in cache:
                        if idx >= 0:
                            raw = {
                                name: coarse_draws[name][idx]
                                for name in names
                                if name in coarse_draws
                            }
                        else:
                            raw = _median_sample_tree(coarse_draws, names=names)
                        cache[cache_key] = _transform_candidate(raw, ch=ch, default_vals=default_vals)
                    return cache[cache_key], label
                return None, "unavailable"

            transferred_seed, transferred_label = _build_candidate(None)
            refined_vi_result = None
            if transferred_seed is not None:
                refine_rng_key = jax.random.fold_in(refine_rng_key, channel_index)
                refined_vi_result, _ = _maybe_refine_vi_locally(
                    sampler=sampler,
                    model=block_model,
                    model_args=block_model_args,
                    init_values=transferred_seed,
                    rng_key=refine_rng_key,
                    log_prefix=f"Blocked coarse-VI init channel {channel_index}",
                )
            refined_draws, refined_means = _extract_refined_artifacts(
                refined_vi_result
            )

            _anchor = f"weights_delta_{channel_index}"

            def _build_effective_candidate(
                rng_key: Optional[jax.Array],
                *,
                refined_draw_values: Dict[str, jnp.ndarray] = refined_draws,
                refined_mean_values: Dict[str, jnp.ndarray] = refined_means,
            ):
                result, label = _pick_from_draws_or_means(
                    rng_key,
                    draws=refined_draw_values,
                    means=refined_mean_values,
                    num_chains=int(sampler.config.num_chains),
                    anchor_name=_anchor,
                )
                if result is not None:
                    return result, f"refined {label}"
                return _build_candidate(rng_key)

            init_strategies[channel_index] = _make_chainwise_init_strategy(
                first_site_name=f"delta_{channel_index}",
                default_values=default_init_values,
                build_candidate=_build_effective_candidate,
                log_posterior_fn=lambda vals, logpost_fn=block_log_posterior_fn: _blocked_channel_log_posterior_from_values(
                    log_posterior_fn=logpost_fn,
                    values=vals,
                ),
                log_prefix=f"Blocked coarse-VI init channel {channel_index}",
            )
            if refined_vi_result is not None:
                diagnostics[
                    f"coarse_vi_fine_refine_guide_channel_{channel_index}"
                ] = refined_vi_result.guide_name
                diagnostics[
                    f"coarse_vi_fine_refine_steps_channel_{channel_index}"
                ] = _coarse_vi_refine_steps(sampler.config)

        return BlockVIArtifacts(
            init_strategies=init_strategies,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        logger.warning(
            f"Could not transfer coarse blocked VI warm start to full grid: {exc}"
        )
        return BlockVIArtifacts(
            init_strategies=[None] * sampler.p,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=_strip_coarse_vi_plot_arrays(diagnostics),
        )
