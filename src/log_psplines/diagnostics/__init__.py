"""Post-fit diagnostics for sampling and fitted spectra."""

from log_psplines.diagnostics.sampling import (
    plot_energy,
    sampling_diagnostics,
    save_diagnostics,
)
from log_psplines.diagnostics.spectrum import spectrum_diagnostics

__all__ = [
    "plot_energy",
    "sampling_diagnostics",
    "save_diagnostics",
    "spectrum_diagnostics",
]
