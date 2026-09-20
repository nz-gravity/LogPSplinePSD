"""Write available fit summaries without fabricating missing diagnostics."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .summary_tables import build_nuts_summary_table, build_vi_summary_table

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
    """Persist computed metrics; absent likelihoods leave LOO unavailable."""
    directory = Path(outdir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    if result.time is not None:
        stats = result.idata["sample_stats"]
        pd.DataFrame(
            [
                {
                    "divergences": int(stats["diverging"].sum()),
                    "mean_accept_prob": float(stats["accept_prob"].mean()),
                    "max_tree_depth_hits": int(
                        (
                            stats["num_steps"]
                            >= 2 ** int(result.metadata["max_tree_depth"]) - 1
                        ).sum()
                    ),
                }
            ]
        ).to_csv(directory / "nuts_summary.csv", index=False)
        return
    summary_row = {}
    if result.vi is not None:
        table = build_vi_summary_table(result.idata, true_psd=true_psd)
        table.to_csv(directory / "vi_summary.csv", index=False)
        for column in ("pareto_k_max", "riae", "l2", "coverage", "final_elbo"):
            if column in table:
                value = _median_numeric(table[column])
                summary_row[f"vi_{column}"] = value
                result.idata["vi_sample_stats"].attrs[column] = value
    if "sample_stats" in result.idata.children:
        table = build_nuts_summary_table(result.idata, true_psd=true_psd)
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
                value = _median_numeric(table[column])
                summary_row[f"nuts_{column}"] = value
                result.idata["sample_stats"].attrs[column] = value
    if summary_row:
        pd.DataFrame([summary_row]).to_csv(
            directory / "diagnostics.csv", index=False
        )
