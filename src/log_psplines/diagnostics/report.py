"""Write fit summaries without coupling the result model to ArviZ."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .summary_tables import (
    _truth_metrics_from_result,
    build_nuts_summary_table,
    build_vi_summary_table,
)

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def _median_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def save_summary_tables(
    result: "PSDResult",
    outdir: str | Path,
    *,
    true_psd: np.ndarray | None = None,
) -> None:
    """Persist available VI/NUTS diagnostics."""
    directory = Path(outdir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    summary_row = {}

    if result.vi is not None:
        vi_input = {"losses": np.asarray(result.vi.losses)}
        if result.vi.losses_per_block is not None:
            vi_input["losses_per_block"] = result.vi.losses_per_block
        table = build_vi_summary_table(vi_input)
        truth = _truth_metrics_from_result(result, true_psd=true_psd)
        for name, value in truth.items():
            table[name] = value
        table.to_csv(directory / "vi_summary.csv", index=False)
        for column in ("pareto_k_max", "riae", "l2", "coverage", "final_elbo"):
            if column in table:
                summary_row[f"vi_{column}"] = _median_numeric(table[column])

    if result.sample_stats is not None:
        table = build_nuts_summary_table(result, true_psd=true_psd)
        table.to_csv(directory / "nuts_summary.csv", index=False)
        for column in (
            "divergences",
            "max_treedepth_hits",
            "rhat_max",
            "riae",
            "l2",
            "coverage",
            "step_size",
            "ess_bulk_min",
            "ess_tail_min",
        ):
            if column in table:
                summary_row[f"nuts_{column}"] = _median_numeric(table[column])

    if summary_row:
        pd.DataFrame([summary_row]).to_csv(
            directory / "diagnostics.csv", index=False
        )
