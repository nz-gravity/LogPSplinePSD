"""Fitted spectra, posterior samples, diagnostics and persistence.

``PSDResult`` is the public result object returned by :func:`log_psplines.fit`.
It owns the posterior samples, sampler statistics and reconstructed spectrum.
ArviZ is deliberately kept at the diagnostics boundary: call
:meth:`PSDResult.to_arviz` when an ArviZ object is useful for R-hat, ESS,
trace or energy diagnostics.

The private DataTree payload is retained temporarily for backwards
compatibility while the old result-packing code is removed.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from log_psplines.arviz_utils._datatree import (
    open_inference_data as _open_inference_data,
    require_dataset as _require_dataset,
    save_inference_data as _save_inference_data,
)
from log_psplines.arviz_utils.from_arviz import get_psd_dataset
from log_psplines.arviz_utils.to_arviz import _losses_per_block_array
from log_psplines.inference.vi import StageResult


@dataclass(init=False)
class PSDResult:
    """Result of a LogPSplinePSD fit.

    The stable public interface is ``posterior``, ``sample_stats``,
    ``spectrum``, ``spectral_density``, ``psd``, ``coherence``,
    ``frequency`` and optional ``time``.

    Parameters
    ----------
    idata:
        Legacy internal DataTree produced by the current inference packers.
        It is immediately converted to native xarray result fields. This
        argument is transitional and will disappear once the packers are
        replaced.
    vi:
        Optional VI diagnostics.
    """

    _tree: xr.DataTree = field(repr=False)
    vi: StageResult | None = None
    posterior: xr.Dataset = field(init=False)
    sample_stats: xr.Dataset | None = field(init=False)
    spectrum: xr.DataArray = field(init=False)
    metadata: dict = field(init=False)

    def __init__(
        self,
        idata: xr.DataTree,
        vi: StageResult | None = None,
    ) -> None:
        self._tree = idata
        self.vi = vi
        self.posterior = self._load_posterior()
        self.sample_stats = self._load_optional_dataset("sample_stats")
        self.spectrum = self._load_spectrum()
        self.metadata = dict(idata.attrs)

    def _load_posterior(self) -> xr.Dataset:
        """Return the fitted posterior as a native xarray Dataset."""
        for group in ("posterior", "vi_posterior"):
            try:
                return _require_dataset(self._tree, group)
            except (KeyError, TypeError):
                continue
        raise KeyError("Fit result does not contain posterior samples.")

    def _load_optional_dataset(self, group: str) -> xr.Dataset | None:
        try:
            return _require_dataset(self._tree, group)
        except (KeyError, TypeError):
            return None

    def _load_spectrum(self) -> xr.DataArray:
        """Reconstruct the spectrum once at result construction time."""
        dataset = get_psd_dataset(self._tree)
        axes = ["chain", "draw"]
        if "time" in dataset["spectral_density"].dims:
            axes.append("time")
        axes.extend(["frequency", "channel", "channel_aux"])
        return dataset["spectral_density"].transpose(*axes)

    @property
    def idata(self) -> xr.DataTree:
        """Legacy DataTree view.

        Deprecated
        ----------
        Use the native ``PSDResult`` fields instead. ArviZ-specific work should
        use :meth:`to_arviz`.
        """
        warnings.warn(
            "PSDResult.idata is deprecated; use result.posterior, "
            "result.sample_stats, result.spectrum, or result.to_arviz().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._tree

    @property
    def frequency(self) -> np.ndarray:
        """Reconstruction frequency grid."""
        return np.asarray(self.spectrum.coords["frequency"].values, dtype=float)

    @property
    def time(self) -> np.ndarray | None:
        """Reconstruction time grid for time-varying fits."""
        if "time" not in self.spectrum.coords:
            return None
        return np.asarray(self.spectrum.coords["time"].values, dtype=float)

    @property
    def spectral_density(self) -> np.ndarray:
        """Posterior spectral matrices with channel axes last."""
        return np.asarray(self.spectrum.values)

    @property
    def psd(self) -> np.ndarray:
        """Auto spectra; scalar fits drop the singleton channel axis."""
        diagonal = np.diagonal(self.spectral_density, axis1=-2, axis2=-1).real
        return diagonal[..., 0] if diagonal.shape[-1] == 1 else diagonal

    @property
    def coherence(self) -> np.ndarray:
        """Magnitude-squared coherence for the reconstructed spectrum."""
        from log_psplines.models.matrix import SpectralMatrix

        return SpectralMatrix.coherence(self.spectral_density)

    def to_arviz(self):
        """Return an ArviZ InferenceData view for sampling diagnostics."""
        from log_psplines.diagnostics.arviz import to_arviz

        return to_arviz(self)

    def to_netcdf(self, path: str | Path) -> None:
        """Save the fit payload to NetCDF.

        Storage still uses the transitional DataTree layout so existing result
        files remain readable during the cleanup.
        """
        _save_inference_data(self._tree, path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "PSDResult":
        """Load a result written by :meth:`to_netcdf`."""
        return cls(_open_inference_data(path))

    def save(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        """Save the fit plus standard plots and diagnostic tables."""
        os.makedirs(outdir, exist_ok=True)
        from log_psplines.diagnostics.report import save_summary_tables
        from log_psplines.plotting.results import (
            plot_posterior_spectrum,
            plot_result_diagnostics,
        )

        # Write once before rendering so fitted samples survive plot failures.
        self.to_netcdf(Path(outdir) / "inference_data.nc")
        save_summary_tables(self, outdir, true_psd=true_psd)
        plot_posterior_spectrum(self, outdir, true_psd=true_psd)
        plot_result_diagnostics(self, outdir)

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
