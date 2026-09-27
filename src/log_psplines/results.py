"""Native fitted-spectrum result object.

ArviZ is intentionally kept out of the core result model. PSDResult owns
posterior samples, sampler statistics and reconstructed spectra directly as
xarray objects. Convert to ArviZ only when sampling diagnostics are required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from log_psplines.data.spectral import PowerData, WishartData
    from log_psplines.inference.vi import VIResult


def observed_wishart_data(data: WishartData) -> xr.Dataset:
    variables = {}
    coords = {
        "frequency": np.asarray(data.freq, dtype=float),
        "channel": np.arange(int(data.p)),
        "channel_aux": np.arange(int(data.p)),
    }
    if data.raw_psd is not None:
        variables["periodogram"] = (
            ("frequency", "channel", "channel_aux"),
            np.asarray(data.raw_psd, dtype=np.complex128),
        )
    return xr.Dataset(variables, coords=coords)


def observed_power_data(data: PowerData) -> xr.Dataset:
    """Store grid or scattered powers with their native coordinates."""
    if data.is_grid:
        dims = ("frequency",) if data.time is None else ("time", "frequency")
        coords = {"frequency": np.asarray(data.frequency)}
        if data.time is not None:
            coords["time"] = np.asarray(data.time)
        return xr.Dataset(
            {
                "power": (dims, np.asarray(data.power)),
                "counts": (dims, np.asarray(data.counts)),
            },
            coords=coords,
            attrs={"units": data.units},
        )
    return xr.Dataset(
        {
            "power": (("ordinate",), np.asarray(data.power)),
            "counts": (("ordinate",), np.asarray(data.counts)),
            "time": (("ordinate",), np.asarray(data.time)),
            "frequency": (("ordinate",), np.asarray(data.frequency)),
        },
        coords={"ordinate": np.arange(data.power.size)},
        attrs={"units": data.units},
    )


def _netcdf_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value.item() if hasattr(value, "item") else value
    arr = np.asarray(value)
    if arr.dtype == object or np.iscomplexobj(arr):
        return str(value)
    if arr.dtype == bool:
        return arr.astype(np.int8)
    return value


@dataclass
class PSDResult:
    """Posterior samples and reconstructed spectrum returned by fit."""

    posterior: xr.Dataset
    spectrum: xr.DataArray
    sample_stats: xr.Dataset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    vi: VIResult | None = None
    vi_posterior: xr.Dataset | None = None
    vi_spectrum: xr.DataArray | None = None
    log_likelihood: xr.Dataset | None = None
    observed_data: xr.Dataset | None = None

    @property
    def frequency(self) -> np.ndarray:
        return np.asarray(
            self.spectrum.coords["frequency"].values, dtype=float
        )

    @property
    def time(self) -> np.ndarray | None:
        if "time" not in self.spectrum.coords:
            return None
        return np.asarray(self.spectrum.coords["time"].values, dtype=float)

    @property
    def spectral_density(self) -> np.ndarray:
        return np.asarray(self.spectrum.values)

    @property
    def psd(self) -> np.ndarray:
        diagonal = np.diagonal(self.spectral_density, axis1=-2, axis2=-1).real
        return diagonal[..., 0] if diagonal.shape[-1] == 1 else diagonal

    @property
    def coherence(self) -> np.ndarray:
        from log_psplines.models.matrix import SpectralMatrix

        return SpectralMatrix.coherence(self.spectral_density)

    def quantiles(
        self, percentiles: tuple[float, ...] = (5.0, 50.0, 95.0)
    ) -> xr.DataArray:
        """Posterior spectral quantiles over chain and draw."""
        values = np.asarray(self.spectrum)
        flat = values.reshape(-1, *values.shape[2:])
        q = np.percentile(flat.real, percentiles, axis=0) + 1j * np.percentile(
            flat.imag, percentiles, axis=0
        )
        dims = ("percentile", *self.spectrum.dims[2:])
        coords = {
            name: self.spectrum.coords[name]
            for name in self.spectrum.dims[2:]
            if name in self.spectrum.coords
        }
        coords["percentile"] = np.asarray(percentiles, dtype=float)
        return xr.DataArray(q, dims=dims, coords=coords)

    def to_arviz(self):
        """Return a minimal ArviZ view for sampling diagnostics."""
        from log_psplines.diagnostics.arviz import to_arviz

        return to_arviz(self)

    def _storage_dataset(self) -> xr.Dataset:
        data_vars: dict[str, xr.DataArray] = {
            "spectral_density": self.spectrum,
        }
        groups = (
            ("posterior", self.posterior),
            ("sample_stats", self.sample_stats),
            ("vi_posterior", self.vi_posterior),
            ("log_likelihood", self.log_likelihood),
            ("observed", self.observed_data),
        )
        for prefix, dataset in groups:
            if dataset is None:
                continue
            stored_group = dataset
            if prefix == "observed":
                rename = {
                    name: f"observed_{name}"
                    for name in set(dataset.dims) | set(dataset.coords)
                }
                stored_group = dataset.rename(rename)
            for name, var in stored_group.data_vars.items():
                data_vars[f"{prefix}__{name}"] = var
        if self.vi_spectrum is not None:
            data_vars["vi_spectral_density"] = self.vi_spectrum
        attrs = {
            key: safe
            for key, value in self.metadata.items()
            if (safe := _netcdf_safe(value)) is not None
        }
        return xr.Dataset(data_vars, attrs=attrs)

    def to_netcdf(self, path: str | Path) -> None:
        """Save the native result representation to one NetCDF file."""
        self._storage_dataset().to_netcdf(
            Path(path), engine="h5netcdf", invalid_netcdf=True
        )

    @classmethod
    def from_netcdf(cls, path: str | Path) -> PSDResult:
        """Load a native result written by to_netcdf."""
        stored = xr.load_dataset(Path(path), engine="h5netcdf")

        def group(prefix: str) -> xr.Dataset | None:
            marker = f"{prefix}__"
            names = [
                name for name in stored.data_vars if name.startswith(marker)
            ]
            if not names:
                return None
            dataset = xr.Dataset(
                {name[len(marker) :]: stored[name] for name in names}
            )
            if prefix == "observed":
                rename = {
                    name: name.removeprefix("observed_")
                    for name in set(dataset.dims) | set(dataset.coords)
                    if name.startswith("observed_")
                }
                dataset = dataset.rename(rename)
            return dataset

        posterior = group("posterior")
        if posterior is None:
            raise ValueError("Stored result is missing posterior samples")
        return cls(
            posterior=posterior,
            sample_stats=group("sample_stats"),
            spectrum=stored["spectral_density"],
            metadata=dict(stored.attrs),
            vi_posterior=group("vi_posterior"),
            vi_spectrum=stored.get("vi_spectral_density"),
            log_likelihood=group("log_likelihood"),
            observed_data=group("observed"),
        )

    def save(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        """Save fitted samples, plots and diagnostics."""
        os.makedirs(outdir, exist_ok=True)
        from log_psplines.diagnostics.report import save_summary_tables
        from log_psplines.plotting.results import (
            plot_posterior_spectrum,
            plot_result_diagnostics,
        )

        self.to_netcdf(Path(outdir) / "inference_data.nc")
        save_summary_tables(self, outdir, true_psd=true_psd)
        plot_posterior_spectrum(self, outdir, true_psd=true_psd)
        plot_result_diagnostics(self, outdir)

        if self.vi is not None and self.vi.losses is not None:
            np.save(Path(outdir) / "vi_losses.npy", np.asarray(self.vi.losses))


__all__ = ["PSDResult"]
