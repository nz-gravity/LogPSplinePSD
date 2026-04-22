"""Minimal posterior PSD summary plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..arviz_utils.from_arviz import get_psd_dataset


def plot_psd_summary(
    idata: xr.DataTree,
    *,
    truth: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot posterior PSD summaries for each auto-spectrum."""

    psd_ds = get_psd_dataset(idata, source="best")
    freqs = np.asarray(psd_ds.coords["frequency"].values, dtype=float)
    spectral_density = np.asarray(psd_ds["spectral_density"].values)
    diag = np.real(
        spectral_density[
            :,
            :,
            np.arange(spectral_density.shape[2]),
            np.arange(spectral_density.shape[2]),
            :,
        ]
    )
    diag = diag.reshape(-1, diag.shape[2], diag.shape[3])

    q05, q50, q95 = np.percentile(diag, [5.0, 50.0, 95.0], axis=0)
    n_channels = int(q50.shape[0])

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(8, max(3, 2.5 * n_channels)),
        squeeze=False,
        sharex=True,
    )

    truth_diag = None
    if truth is not None:
        truth_arr = np.asarray(truth)
        if truth_arr.ndim == 1:
            truth_diag = truth_arr[None, :]
        elif truth_arr.ndim == 3:
            truth_diag = np.real(
                truth_arr[
                    :,
                    np.arange(truth_arr.shape[1]),
                    np.arange(truth_arr.shape[2]),
                ]
            ).T

    for channel in range(n_channels):
        ax = axes[channel, 0]
        ax.plot(freqs, q50[channel], color="C0", linewidth=1.5, label="median")
        ax.fill_between(
            freqs, q05[channel], q95[channel], color="C0", alpha=0.2
        )
        if truth_diag is not None and truth_diag.shape[-1] == freqs.size:
            ax.plot(
                freqs,
                truth_diag[channel],
                color="C3",
                linestyle="--",
                linewidth=1.2,
                label="truth",
            )
        ax.set_ylabel(f"PSD {channel}")
        ax.grid(True, alpha=0.3)
        if channel == 0:
            ax.legend(frameon=False)

    axes[-1, 0].set_xlabel("Frequency")
    fig.tight_layout()
    return fig
