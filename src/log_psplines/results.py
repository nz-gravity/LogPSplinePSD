"""Native fitted-spectrum result object.

ArviZ is intentionally kept out of the core result model. PSDResult owns
posterior samples, sampler statistics and reconstructed spectra directly as
xarray objects. Convert to ArviZ only when sampling diagnostics are required.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from log_psplines.models.reconstruction import reconstruct_psd_matrix

if TYPE_CHECKING:
    from log_psplines.config import PipelineConfig, PowerSplineConfig
    from log_psplines.data.spectral import PowerSpectrum, WishartData
    from log_psplines.inference.components import SpectralComponents
    from log_psplines.inference.vi import StageResult
    from log_psplines.models.spectrum import LogPSpline


def _values_to_dataset(
    values: dict[str, Any] | None,
    *,
    values_are_draws: bool = True,
) -> xr.Dataset | None:
    if not values:
        return None
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)
        if values_are_draws:
            if array.ndim == 0:
                raise ValueError(f"Samples for '{name}' require a draw axis")
            array = array[None, ...]
        else:
            array = array[None, None, ...]
        tail = tuple(f"{name}_dim_{i}" for i in range(array.ndim - 2))
        data_vars[name] = xr.DataArray(
            array, dims=("chain", "draw", *tail)
        )
    return xr.Dataset(data_vars)


def _flatten(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    return arr.reshape((-1,) + arr.shape[2:])


def _batch_spline_eval(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("fk,sk->sf", np.asarray(basis), np.asarray(weights))


def _stationary_spectrum(
    posterior: xr.Dataset,
    spline_model: "SpectralComponents",
    data: "WishartData",
) -> xr.DataArray:
    """Reconstruct stationary spectral matrices directly from posterior draws."""
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    n_sample = n_chain * n_draw

    log_delta = []
    for j in range(int(data.p)):
        weights = _flatten(posterior[f"weights_delta_{j}"].values)
        log_delta.append(
            _batch_spline_eval(spline_model.diagonal_models[j].basis, weights)
        )
    log_delta_sq = np.stack(log_delta, axis=-1)

    n_theta = int(spline_model.n_theta)
    theta_re = np.zeros((n_sample, int(data.N), n_theta))
    theta_im = np.zeros_like(theta_re)
    for theta_idx, (j, l) in enumerate(spline_model.theta_pairs):
        for part, target in (("re", theta_re), ("im", theta_im)):
            name = f"weights_theta_{part}_{j}_{l}"
            if name not in posterior:
                continue
            weights = _flatten(posterior[name].values)
            model = spline_model.get_theta_model(part, j, l)
            target[..., theta_idx] = _batch_spline_eval(model.basis, weights)

    spectrum = reconstruct_psd_matrix(
        jnp.asarray(log_delta_sq),
        jnp.asarray(theta_re),
        jnp.asarray(theta_im),
        n_samples_max=n_sample,
    ).reshape(n_chain, n_draw, int(data.N), int(data.p), int(data.p))

    if data.channel_stds is not None:
        scale = np.outer(data.channel_stds, data.channel_stds)
        spectrum = spectrum * scale[None, None, None, :, :]

    return xr.DataArray(
        np.asarray(spectrum, dtype=np.complex128),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "frequency": np.asarray(data.freq, dtype=float),
            "channel": np.arange(int(data.p)),
            "channel_aux": np.arange(int(data.p)),
        },
        name="spectral_density",
    )


def _power_spectrum(
    posterior: xr.Dataset,
    spline: "LogPSpline",
) -> xr.DataArray:
    """Reconstruct a scalar time-frequency spectrum from coefficient draws."""
    if spline.time is None:
        raise ValueError("Power results require a time basis")
    log_psd = np.einsum(
        "ti,cdij,fj->cdtf",
        np.asarray(spline.time.basis),
        np.asarray(posterior["weights"].values),
        np.asarray(spline.basis),
        optimize=True,
    )
    spectrum = np.exp(log_psd)[..., None, None]
    return xr.DataArray(
        spectrum.astype(np.complex128),
        dims=("chain", "draw", "time", "frequency", "channel", "channel_aux"),
        coords={
            "chain": posterior.coords["chain"],
            "draw": posterior.coords["draw"],
            "time": np.asarray(spline.time.grid, dtype=float),
            "frequency": np.asarray(spline.frequency.grid, dtype=float),
            "channel": [0],
            "channel_aux": [0],
        },
        name="spectral_density",
    )


def _observed_wishart(data: "WishartData") -> xr.Dataset:
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


def _observed_power(data: "PowerSpectrum") -> xr.Dataset:
    return xr.Dataset(
        {
            "power": (("time", "frequency"), np.asarray(data.power)),
            "counts": (("time", "frequency"), np.asarray(data.counts)),
        },
        coords={
            "time": np.asarray(data.time),
            "frequency": np.asarray(data.frequency),
        },
        attrs={"units": data.units},
    )


def _observed_scattered(data) -> xr.Dataset:
    return xr.Dataset(
        {
            "power": (("ordinate",), np.asarray(data.power)),
            "counts": (("ordinate",), np.asarray(data.counts)),
            "time": (("ordinate",), np.asarray(data.time)),
            "frequency": (("ordinate",), np.asarray(data.frequency)),
        },
        coords={"ordinate": np.arange(np.asarray(data.power).size)},
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
    vi: "StageResult | None" = None
    vi_posterior: xr.Dataset | None = None
    vi_spectrum: xr.DataArray | None = None
    log_likelihood: xr.Dataset | None = None
    observed_data: xr.Dataset | None = None

    @classmethod
    def from_stationary(
        cls,
        *,
        posterior: xr.Dataset,
        sample_stats: xr.Dataset | None,
        data: "WishartData",
        spline_model: "SpectralComponents",
        config: "PipelineConfig",
        vi: "StageResult | None" = None,
        log_likelihood: xr.Dataset | None = None,
        sampling_eta: float | None = None,
    ) -> "PSDResult":
        vi_posterior = None
        vi_spectrum = None
        if vi is not None:
            values = vi.samples if vi.samples is not None else vi.init_values
            vi_posterior = _values_to_dataset(
                values, values_are_draws=vi.samples is not None
            )
            if vi_posterior is not None:
                vi_spectrum = _stationary_spectrum(
                    vi_posterior, spline_model, data
                )

        metadata = {
            "data_type": "multivariate",
            "scaling_factor": float(data.scaling_factor or 1.0),
            "channel_stds": (
                None
                if data.channel_stds is None
                else np.asarray(data.channel_stds)
            ),
            "max_tree_depth": int(config.max_tree_depth),
            "eta": float(config.eta),
            "sampling_eta": float(
                config.eta if sampling_eta is None else sampling_eta
            ),
            "compute_lnz": bool(config.compute_lnz),
        }
        return cls(
            posterior=posterior,
            sample_stats=sample_stats,
            spectrum=_stationary_spectrum(posterior, spline_model, data),
            metadata=metadata,
            vi=vi,
            vi_posterior=vi_posterior,
            vi_spectrum=vi_spectrum,
            log_likelihood=log_likelihood,
            observed_data=_observed_wishart(data),
        )

    @classmethod
    def from_power(
        cls,
        *,
        posterior: xr.Dataset,
        sample_stats: xr.Dataset | None,
        data: "PowerSpectrum",
        spline: "LogPSpline",
        config: "PowerSplineConfig",
        log_likelihood: xr.Dataset | None = None,
    ) -> "PSDResult":
        return cls(
            posterior=posterior,
            sample_stats=sample_stats,
            spectrum=_power_spectrum(posterior, spline),
            metadata={
                **asdict(config),
                "data_type": "power",
                "likelihood": "power_whittle",
                "units": data.units,
            },
            log_likelihood=log_likelihood,
            observed_data=_observed_power(data),
        )

    @classmethod
    def from_scattered_power(
        cls,
        *,
        posterior: xr.Dataset,
        sample_stats: xr.Dataset | None,
        data,
        spline: "LogPSpline",
        config: "PowerSplineConfig",
        log_likelihood: xr.Dataset | None = None,
    ) -> "PSDResult":
        return cls(
            posterior=posterior,
            sample_stats=sample_stats,
            spectrum=_power_spectrum(posterior, spline),
            metadata={
                **asdict(config),
                "data_type": "power",
                "likelihood": "power_whittle",
                "units": data.units,
            },
            log_likelihood=log_likelihood,
            observed_data=_observed_scattered(data),
        )

    @property
    def frequency(self) -> np.ndarray:
        return np.asarray(self.spectrum.coords["frequency"].values, dtype=float)

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
            for name, var in dataset.data_vars.items():
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
    def from_netcdf(cls, path: str | Path) -> "PSDResult":
        """Load a native result written by to_netcdf."""
        stored = xr.load_dataset(Path(path), engine="h5netcdf")

        def group(prefix: str) -> xr.Dataset | None:
            marker = f"{prefix}__"
            names = [name for name in stored.data_vars if name.startswith(marker)]
            if not names:
                return None
            return xr.Dataset(
                {name[len(marker):]: stored[name] for name in names}
            )

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
