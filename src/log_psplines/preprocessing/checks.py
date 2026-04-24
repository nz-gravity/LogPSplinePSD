from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..logger import logger
from ..pipeline.config import PipelineConfig
from .data_prep import _normalize_excluded_frequency_bands


def _run_preprocessing_checks(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: PipelineConfig,
) -> None:
    """Run eigenvalue separation warnings (lightweight, no plotting)."""
    if not isinstance(processed_data, MultivarFFT):
        return
    if processed_data.raw_psd is None:
        logger.warning(
            "Skipping eigenvalue separation check: processed_data.raw_psd is missing."
        )
        return

    try:
        from ..diagnostics.preprocessing import (
            eigenvalue_separation_diagnostics,
        )

        min_lambda1_quantile = 0.05
        warn_threshold = 0.8
        warn_frac = 0.25

        diag = eigenvalue_separation_diagnostics(
            freq=np.asarray(processed_data.freq, dtype=float),
            matrix=np.asarray(processed_data.raw_psd),
            min_lambda1_quantile=min_lambda1_quantile,
        )
        p = int(diag.eigvals_desc.shape[1])
        if p < 2:
            return
        summaries = None
        if config.verbose:
            summaries = diag.ratio_summary(warn_threshold=warn_threshold)
            if diag.lambda1_cutoff is not None:
                kept = int(np.count_nonzero(diag.mask))
                logger.info(
                    f"Eigenvalue separation mask: keep λ1 > {diag.lambda1_cutoff:.3e} ({kept}/{diag.mask.size} bins)."
                )

        for key, ratio in diag.ratios.items():
            ratio_m = np.asarray(ratio)[diag.mask]
            ratio_m = ratio_m[np.isfinite(ratio_m)]
            frac = (
                float(np.mean(ratio_m > warn_threshold))
                if ratio_m.size
                else 0.0
            )
            if frac >= warn_frac:
                if summaries is None:
                    msg = (
                        f"Eigenvalue separation {key}: "
                        f"frac(>{warn_threshold:.2f})={frac*100:.1f}%"
                    )
                else:
                    msg = f"Eigenvalue separation {summaries[key]}"
                logger.warning(msg)
                worst = diag.worst_frequencies(top_k=10).get(key, [])
                if worst:
                    joined = ", ".join([f"{f:.4g}:{v:.3f}" for f, v in worst])
                    logger.warning(
                        f"Worst-separated frequencies for {key}: {joined}"
                    )
                continue
            if config.verbose and summaries is not None:
                logger.info(f"Eigenvalue separation {summaries[key]}")

    except Exception as exc:
        logger.warning(f"Eigenvalue separation check failed: {exc}")


def _save_preprocessing_plot(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: PipelineConfig,
    spline_model: object | None = None,
) -> None:
    """Save the preprocessing diagnostic plot, optionally with knot locations.

    This should be called *after* the spline model is built so that the knot
    locations shown on the plot are the exact same ones used by the sampler.

    Args:
        processed_data: The processed FFT data.
        config: Run configuration.
        spline_model: A ``MultivariateLogPSplines`` instance.  When provided,
            knot positions are extracted and drawn on each component panel.
    """
    if not isinstance(processed_data, MultivarFFT):
        return
    if processed_data.raw_psd is None:
        return

    try:
        from ..diagnostics.preprocessing import (
            eigenvalue_separation_diagnostics,
            extract_component_knots,
            save_eigenvalue_separation_plot,
        )

        if config.outdir is None:
            logger.warning("Skipping preprocessing plot save: outdir is None.")
            return

        freq = np.asarray(processed_data.freq, dtype=float)
        diag = eigenvalue_separation_diagnostics(
            freq=freq,
            matrix=np.asarray(processed_data.raw_psd),
            min_lambda1_quantile=0.05,
        )
        p = int(diag.eigvals_desc.shape[1])
        if p < 2:
            return

        out_path = (
            Path(config.outdir)
            / "diagnostics"
            / "preprocessing_eigenvalue_ratios.png"
        )

        nb = int(processed_data.Nb)
        nh = int(processed_data.Nh)
        n_coarse_or_retained = int(processed_data.N)
        n_ell = n_coarse_or_retained * nh
        fs = float(processed_data.fs)
        dt = 1.0 / fs
        lb = int(round(float(processed_data.duration) * fs))
        n_time = lb * nb
        total_duration = float(processed_data.duration) * nb
        n_fft = lb // 2 + 1
        nc = n_coarse_or_retained if nh > 1 else None

        info_parts = [
            f"n={n_time}",
            f"dt={dt:.6g}",
            f"T={total_duration:.6g}",
            f"N={n_fft}",
            f"N_ell={n_ell}",
            f"p={int(processed_data.p)}",
            f"Nb={nb}",
            f"Lb={lb}",
            f"Nc={nc if nc is not None else 'NA'}",
            f"Nh={nh}",
        ]
        window_name = (
            "rect"
            if config.wishart_window is None
            else str(config.wishart_window)
        )
        info_parts.append(f"window={window_name}")
        info_text = ", ".join(info_parts)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_bands = _normalize_excluded_frequency_bands(
            config.exclude_freq_bands
        )

        # Extract knot positions from the built model.
        comp_knots = None
        if spline_model is not None:
            try:
                comp_knots = extract_component_knots(spline_model, freq)
            except Exception as knot_exc:
                logger.debug(
                    f"Could not extract knot locations for plot: {knot_exc}"
                )

        save_eigenvalue_separation_plot(
            diag,
            str(out_path),
            warn_threshold=0.8,
            info_text=info_text,
            excluded_bands=excluded_bands,
            cholesky_matrix=np.asarray(processed_data.raw_psd),
            component_knots=comp_knots,
        )
        if config.verbose:
            logger.info(f"Saved preprocessing eigenvalue plot to {out_path}")
    except Exception as exc:
        logger.warning(f"Preprocessing plot save failed: {exc}")
