from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..logger import logger
from .configs import RunMCMCConfig


def _run_preprocessing_checks(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: RunMCMCConfig,
) -> None:
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
            save_eigenvalue_separation_plot,
        )

        # Use hardcoded defaults
        min_lambda1_quantile = 0.0
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
        if config.diagnostics.verbose:
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
            if config.diagnostics.verbose and summaries is not None:
                logger.info(f"Eigenvalue separation {summaries[key]}")

        # Always save preprocessing plots
        if config.diagnostics.outdir is None:
            logger.warning("Skipping preprocessing plot save: outdir is None.")
            return

        out_path = (
            Path(config.diagnostics.outdir)
            / "preprocessing_eigenvalue_ratios.png"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_eigenvalue_separation_plot(
            diag,
            str(out_path),
            warn_threshold=warn_threshold,
        )
        if config.diagnostics.verbose:
            logger.info(f"Saved preprocessing eigenvalue plot to {out_path}")
    except Exception as exc:
        logger.warning(f"Eigenvalue separation check failed: {exc}")
