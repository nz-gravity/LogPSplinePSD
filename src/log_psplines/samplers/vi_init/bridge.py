"""Coarse-to-fine transfer bridge for VI warm starts.

The bridge interpolates a PSD (or spline evaluation) from a coarse frequency
grid onto a fine frequency grid and re-fits spline weights on the fine basis.
All interpolation reuses the existing ``_interp_psd_array`` /
``_interp_frequency_indexed_array`` utilities used for true-PSD alignment.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
import numpy as np

from ...datatypes.multivar_utils import _interp_frequency_indexed_array
from ...psplines.initialisation import init_weights


def transfer_univar_weights(
    *,
    coarse_weights: np.ndarray,
    coarse_spline_model,
    coarse_freq: np.ndarray,
    coarse_scaling: float,
    fine_spline_model,
    fine_freq: np.ndarray,
    fine_scaling: float,
) -> jnp.ndarray:
    """Transfer univariate spline weights from a coarse grid to a fine grid.

    Steps:
        1. Evaluate the coarse spline to get log-PSD on the coarse grid.
        2. Exponentiate and apply coarse scaling to get PSD in physical units.
        3. Interpolate PSD onto the fine frequency grid.
        4. Undo fine scaling and re-fit spline weights on the fine basis.
    """
    coarse_log_psd = np.asarray(
        coarse_spline_model(jnp.asarray(coarse_weights))
    )
    coarse_psd = np.exp(coarse_log_psd) * coarse_scaling

    fine_psd = _interp_psd_1d(coarse_psd, coarse_freq, fine_freq)
    fine_model_psd = np.maximum(fine_psd / fine_scaling, 1e-12)

    return init_weights(
        jnp.asarray(np.log(fine_model_psd)),
        fine_spline_model,
    )


def transfer_multivar_log_spline(
    *,
    coarse_weights: np.ndarray,
    coarse_basis: np.ndarray,
    coarse_freq: np.ndarray,
    fine_spline_model,
    fine_freq: np.ndarray,
) -> jnp.ndarray:
    """Transfer a single spline component (log-delta or theta) across grids.

    Unlike the univariate case, multivariate blocked VI operates on individual
    spline components (log-diagonal, theta_re, theta_im) rather than full PSD.
    The transfer evaluates the coarse spline, interpolates the curve, and
    re-fits weights on the fine basis.
    """
    eval_coarse = np.asarray(
        np.einsum("nk,k->n", coarse_basis, coarse_weights)
    )
    eval_fine = _interp_psd_1d(eval_coarse, coarse_freq, fine_freq)
    return init_weights(
        jnp.asarray(np.array(eval_fine, copy=True)),
        fine_spline_model,
    )


def transfer_block_init_values(
    *,
    draw_values: Dict[str, jnp.ndarray],
    channel_index: int,
    coarse_sampler,
    fine_sampler,
    coarse_freq: np.ndarray,
    fine_freq: np.ndarray,
    default_init_values: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Transfer a full set of blocked-channel init values from coarse to fine.

    Handles the diagonal (log-delta) weights and all off-diagonal theta
    components for one channel.
    """
    candidate = dict(default_init_values)

    fine_diag_model = fine_sampler.spline_model.diagonal_models[channel_index]
    candidate[f"weights_delta_{channel_index}"] = transfer_multivar_log_spline(
        coarse_weights=np.asarray(
            draw_values[f"weights_delta_{channel_index}"]
        ),
        coarse_basis=np.asarray(coarse_sampler.all_bases[channel_index]),
        coarse_freq=coarse_freq,
        fine_spline_model=fine_diag_model,
        fine_freq=fine_freq,
    )

    if channel_index > 0:
        for theta_idx in range(channel_index):
            for prefix, fine_model in (
                ("theta_re", fine_sampler.spline_model.offdiag_re_model),
                ("theta_im", fine_sampler.spline_model.offdiag_im_model),
            ):
                w_key = f"weights_{prefix}_{channel_index}_{theta_idx}"
                candidate[w_key] = transfer_multivar_log_spline(
                    coarse_weights=np.asarray(draw_values[w_key]),
                    coarse_basis=np.asarray(coarse_sampler._theta_basis),
                    coarse_freq=coarse_freq,
                    fine_spline_model=fine_model,
                    fine_freq=fine_freq,
                )

    return candidate


def _interp_psd_1d(
    values: np.ndarray,
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
) -> np.ndarray:
    """Interpolate a 1-D real array along the frequency axis.

    Delegates to ``_interp_frequency_indexed_array`` — the same utility used
    for true-PSD alignment — ensuring consistent interpolation behaviour
    (sorting, deduplication) across the codebase.
    """
    values = np.asarray(values).ravel()
    return _interp_frequency_indexed_array(
        np.asarray(freq_src, dtype=float),
        np.asarray(freq_tgt, dtype=float),
        values,
        sort_and_dedup=True,
    ).ravel()
