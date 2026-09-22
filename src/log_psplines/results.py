"""Fitted spectra, posterior access and persistence.


Understanding ``PSDResult``
============================

``fit(data, config)`` returns a ``PSDResult``. It keeps the posterior samples,
the frequency grid, and convenient reconstructed spectral quantities together.

Core properties
---------------

``result.frequency``
   The retained positive-frequency grid in Hz. DC is removed by preprocessing.

``result.psd``
   Auto-spectral posterior draws. For a univariate stationary fit, the shape is
   ``(chain, draw, frequency)``. For a multivariate fit, channel dimensions are
   appended after frequency.

``result.spectral_density``
   Reconstructed spectral-matrix draws with shape
   ``(chain, draw, frequency, channel, channel_aux)``. Time-varying scalar fits
   add a ``time`` axis before ``frequency``.

``result.coherence``
   Coherence reconstructed from the spectral matrix. It is available for
   multivariate fits.

``result.posterior``
   The posterior as an ``xarray.Dataset``. Use this when you need named
   coordinates or model parameters rather than reconstructed spectra.

Save and reload
---------------

Use ``to_netcdf`` when you only need the inference data:

.. code-block:: python

   result.to_netcdf("runs/example/inference_data.nc")

Use ``save`` to write inference data, summary tables, plots, and diagnostics:

.. code-block:: python

   result.save("runs/example")

Reload a saved inference result with:

.. code-block:: python

   from log_psplines import PSDResult

   result = PSDResult.from_netcdf("runs/example/inference_data.nc")

For the files written by ``save`` and the checks to perform before interpreting
an analysis, see below



Outputs and Diagnostics
=======================

Return Value
------------

``fit(...).idata`` returns an ``xarray.DataTree``. Important groups include:

``posterior``
   NUTS posterior samples for spline weights and model parameters.

``sample_stats``
   Per-channel sampler diagnostics such as acceptance rate, step size, tree
   depth, and log probability.

``observed_data``
   Frequency grid and empirical PSD-like data derived from the Wishart
   statistics.

``vi_posterior`` and ``vi_sample_stats``
   VI draws, losses, and warm-start diagnostics when VI is enabled.

``prior_predictive`` and ``posterior_predictive``
   Reconstructed spectral quantities used by plotting and diagnostics when
   available.

Saved Files
-----------

When ``PipelineConfig(outdir=...)`` is set, the pipeline writes:

``inference_data.nc``
   NetCDF serialisation of the returned ``DataTree``.

``posterior_spectrum.png``
   PSD matrix summary or scalar time-frequency surface.

``diagnostics/vi_summary.csv``
   VI convergence and loss summary.

``diagnostics/nuts_summary.csv``
   NUTS diagnostics and, when a truth PSD is supplied, error metrics.

``diagnostics/vi_loss.png``
   VI loss trace.

``diagnostics/traces.png`` and ``diagnostics/energy.png``
   Standard MCMC trace and energy diagnostics.

Some files are conditional. For example, preprocessing eigenvalue plots are
written for multivariate frequency-domain inputs when an output directory is
available.

Loading Results
---------------

.. code-block:: python

   from log_psplines.arviz_utils import open_inference_data

   idata = open_inference_data("runs/example/inference_data.nc")

Extracting PSD Summaries
------------------------

.. code-block:: python

   from log_psplines.arviz_utils import (
       get_multivar_posterior_psd_quantiles,
       get_psd_dataset,
   )

   psd_draws = get_psd_dataset(idata, source="posterior")
   q = get_multivar_posterior_psd_quantiles(idata)

``get_psd_dataset`` returns posterior draws when available. Quantile helpers
return compact arrays suitable for plotting and reporting.

Diagnostics Checklist
---------------------

- Check that posterior PSD diagonals are positive.
- For multivariate runs, check Hermitian symmetry and positive definiteness of
  reconstructed spectral matrices.
- Check coherence lies in ``[0, 1]``.
- Inspect NUTS divergences, tree-depth hits, and effective sample size.
- Compare VI and NUTS posterior summaries when using VI warm starts.
- If ``true_psd`` was supplied, review RIAE, L2, and coverage metrics in the
  saved summaries.

Reporting contracts
-------------------

``get_psd_dataset(result.idata)`` reconstructs stationary and time-frequency
results using named dimensions. Time-frequency results add a ``time`` axis
before ``frequency``. ``PSDResult.spectral_density`` places channel axes last.

VI pointwise log likelihoods are currently unavailable. No zero-filled
``vi_log_likelihood`` group is written, and derived LOO metrics are unavailable.

Saving writes ``inference_data.nc`` before rendering. Unexpected plotting or
reporting failures raise an error; no substitute figure is saved under the
requested figure's filename.



"""

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
