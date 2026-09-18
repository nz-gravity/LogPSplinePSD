from log_psplines.plotting.base import (
    COLORS,
    PlotConfig,
    compute_confidence_intervals,
    extract_plotting_data,
    setup_plot_style,
)
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.plotting.vi import plot_vi_loss

__all__ = [
    # Base utilities
    "COLORS",
    "PlotConfig",
    "extract_plotting_data",
    "compute_confidence_intervals",
    "setup_plot_style",
    # Main plotting functions
    "plot_psd_matrix",
    "PSDMatrixPlotSpec",
    "plot_vi_loss",
]
