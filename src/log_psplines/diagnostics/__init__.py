"""Minimal diagnostics helpers."""

from log_psplines.diagnostics.plot_nuts import plot_energy
from log_psplines.diagnostics.summary_tables import build_nuts_summary_table, build_vi_summary_table

__all__ = [
    "build_nuts_summary_table",
    "build_vi_summary_table",
    "plot_energy",
]
