"""Fitted spectra, posterior access and persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from log_psplines.arviz_utils._datatree import (
    save_inference_data as _save_inference_data,
)
from log_psplines.arviz_utils.to_arviz import _losses_per_block_array
from log_psplines.inference.vi import StageResult


@dataclass
class PSDResult:
    """Outputs from InferencePipeline.run()."""

    idata: xr.DataTree
    vi: StageResult | None = None
    time: np.ndarray | None = None

    def __post_init__(self) -> None:
        if "power_basis" in self.idata.children:
            from log_psplines.arviz_utils._datatree import require_dataset

            self.time = require_dataset(self.idata, "power_basis")[
                "grid_time"
            ].values

    @property
    def posterior(self) -> xr.Dataset:
        from log_psplines.arviz_utils.from_arviz import get_sample_dataset

        return get_sample_dataset(self.idata)

    @property
    def metadata(self) -> dict:
        return dict(self.idata.attrs)

    @property
    def frequency(self) -> np.ndarray:
        from log_psplines.arviz_utils.from_arviz import (
            _get_multivar_frequency_grid,
        )

        if "power_basis" in self.idata.children:
            from log_psplines.arviz_utils._datatree import require_dataset

            return require_dataset(self.idata, "power_basis")[
                "grid_frequency"
            ].values
        return _get_multivar_frequency_grid(self.idata)

    @property
    def spectral_density(self) -> np.ndarray:
        """Posterior draws (chain, draw, F, C, C), in original data units.

        A time grid inserts T before F. TV scalar results use C=1.
        """
        from log_psplines.arviz_utils.from_arviz import get_psd_dataset

        dataset = get_psd_dataset(self.idata)
        axes = ("chain", "draw")
        if "time" in dataset.dims:
            axes += ("time",)
        return dataset.spectral_density.transpose(
            *axes, "frequency", "channel", "channel_aux"
        ).values

    @property
    def psd(self) -> np.ndarray:
        """Auto spectra (chain, draw, F) for C=1, (..., F, C) otherwise."""
        diagonal = np.diagonal(self.spectral_density, axis1=-2, axis2=-1).real
        return diagonal[..., 0] if diagonal.shape[-1] == 1 else diagonal

    @property
    def coherence(self) -> np.ndarray:
        from log_psplines.models.matrix import SpectralMatrix

        return SpectralMatrix.coherence(self.spectral_density)

    def to_netcdf(self, path: str | Path) -> None:
        """Save posterior, basis and metadata without rendering plots."""
        _save_inference_data(self.idata, path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> PSDResult:
        from log_psplines.arviz_utils._datatree import open_inference_data

        return cls(idata=open_inference_data(path))

    def save(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        os.makedirs(outdir, exist_ok=True)
        from log_psplines.diagnostics.report import save_summary_tables
        from log_psplines.plotting.results import (
            plot_posterior_spectrum,
            plot_result_diagnostics,
        )

        # Preserve fitted data even if a requested diagnostic or plot fails.
        self.to_netcdf(Path(outdir) / "inference_data.nc")
        save_summary_tables(self, outdir, true_psd=true_psd)
        plot_posterior_spectrum(self, outdir, true_psd=true_psd)
        plot_result_diagnostics(self, outdir)
        _save_inference_data(
            self.idata,
            os.path.join(outdir, "inference_data.nc"),
            engine="h5netcdf",
        )
        if self.vi is not None and self.vi.losses is not None:
            np.save(
                os.path.join(outdir, "vi_losses.npy"),
                np.asarray(self.vi.losses),
            )
            losses_per_block = _losses_per_block_array(
                self.vi.losses_per_block
            )
            if losses_per_block.size:
                np.save(
                    os.path.join(outdir, "vi_losses_per_block.npy"),
                    losses_per_block,
                )
