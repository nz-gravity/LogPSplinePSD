"""Summarize every power-spectrum draw in bounded frequency chunks."""

import numpy as np
import xarray as xr


def power_result_spectra(
    posterior, evaluate, time, frequency, channels, config
):
    """Return a draw preview and all-draw summaries of diagonal spectra.

    evaluate(slice) returns (chain, draw, time, frequency, channel) variances.
    All posterior draws enter quantiles and both means, regardless of preview
    size. The caller's scientific forward model is evaluated before reduction.
    """
    nc, nd = posterior.sizes["chain"], posterior.sizes["draw"]
    keep = (
        nd if config.spectrum_draws is None else min(nd, config.spectrum_draws)
    )
    indices = np.linspace(0, nd - 1, keep, dtype=int)
    nt, nf, p = len(time), len(frequency), len(channels)
    preview = np.zeros((nc, keep, nt, nf, p, p), dtype=np.complex128)
    quantiles = np.zeros((3, nt, nf, p, p), dtype=np.complex128)
    mean = np.zeros((nt, nf, p, p), dtype=np.complex128)
    geometric = np.zeros_like(mean)
    diagonal = np.arange(p)
    for start in range(0, nf, config.spectrum_chunk_size):
        stop = min(nf, start + config.spectrum_chunk_size)
        draws = np.asarray(evaluate(slice(start, stop)), dtype=float)
        expected = (nc, nd, nt, stop - start, p)
        if (
            draws.shape != expected
            or not np.isfinite(draws).all()
            or np.any(draws <= 0)
        ):
            raise ValueError(
                f"posterior spectrum must be finite and positive with shape {expected}"
            )
        preview[:, :, :, start:stop, diagonal, diagonal] = draws[:, indices]
        quantiles[:, :, start:stop, diagonal, diagonal] = np.percentile(
            draws, [5, 50, 95], axis=(0, 1)
        )
        mean[:, start:stop, diagonal, diagonal] = draws.mean(axis=(0, 1))
        geometric[:, start:stop, diagonal, diagonal] = np.exp(
            np.log(draws).mean(axis=(0, 1))
        )
    dims = ("time", "frequency", "channel", "channel_aux")
    coords = dict(
        time=time,
        frequency=frequency,
        channel=list(channels),
        channel_aux=list(channels),
    )
    spectrum = xr.DataArray(
        preview,
        dims=("chain", "draw", *dims),
        coords={**coords, "chain": np.arange(nc), "draw": indices},
        name="spectral_density",
    )
    summary = xr.Dataset(
        {
            "quantiles": xr.DataArray(
                quantiles,
                dims=("percentile", *dims),
                coords={**coords, "percentile": [5.0, 50.0, 95.0]},
            ),
            "mean": xr.DataArray(mean, dims=dims, coords=coords),
            "geometric_mean": xr.DataArray(
                geometric, dims=dims, coords=coords
            ),
        },
        attrs={"draws_per_chain": nd, "num_chains": nc},
    )
    return spectrum, summary
