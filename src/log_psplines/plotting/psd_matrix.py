from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD, _get_coherence
from ..logger import logger
from .base import extract_plotting_data, setup_plot_style

# Setup default plot styling
setup_plot_style()


def _extract_empirical_psd_from_idata(idata) -> EmpiricalPSD | None:
    """
    Extract empirical PSD from multivariate idata for plotting.

    For multivariate data, the structure is different from univariate.
    This function handles the case where data is stored as separate
    frequency, channel, and FFT components.
    """
    try:
        # Check if we have the multivariate data structure
        if "observed_data" in idata:
            obs_data = idata["observed_data"]

            # Check for the components that indicate multivariate structure
            if all(
                key in obs_data
                for key in ["freq", "channels", "fft_re", "fft_im"]
            ):
                freq = obs_data["freq"].values
                channels = obs_data["channels"].values
                fft_re = obs_data["fft_re"].values
                fft_im = obs_data["fft_im"].values

                # Reconstruct the complex FFT
                fft_complex = fft_re + 1j * fft_im

                # Compute PSD matrix
                # For multivariate data, we need to compute the PSD for each channel pair
                n_channels = len(channels)
                n_freq = len(freq)

                # Initialize PSD matrix
                psd_matrix = np.zeros(
                    (n_freq, n_channels, n_channels), dtype=np.complex128
                )

                for i in range(n_channels):
                    for j in range(n_channels):
                        psd_matrix[:, i, j] = fft_complex[:, i] * np.conj(
                            fft_complex[:, j]
                        )

                coherence = _get_coherence(psd_matrix)

                return EmpiricalPSD(
                    freq=freq,
                    psd=psd_matrix,
                    coherence=coherence,
                    channels=channels,
                )

        return None

    except Exception as e:
        # If extraction fails, return None and let the plotting function handle it
        logger.warning(f"Could not extract empirical PSD from idata: {e}")
        return None


def _pack_ci_dict(psd_samples, show_coherence: bool):
    """Compute 5/50/95% bands for diag PSDs, and coherence OR Re/Im cross-spectra."""
    ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}}
    _, _, n_channels, _ = psd_samples.shape
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                q05 = np.percentile(psd_samples[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, i].real, 95, axis=0)
                ci_dict["psd"][(i, i)] = (q05, q50, q95)
            elif show_coherence and i > j:
                coh = np.abs(psd_samples[:, :, i, j]) ** 2 / (
                    np.abs(psd_samples[:, :, i, i])
                    * np.abs(psd_samples[:, :, j, j])
                )
                q05 = np.percentile(coh, 5, axis=0)
                q50 = np.percentile(coh, 50, axis=0)
                q95 = np.percentile(coh, 95, axis=0)
                ci_dict["coh"][(i, j)] = (q05, q50, q95)
            elif not show_coherence:
                re_q05 = np.percentile(psd_samples[:, :, i, j].real, 5, axis=0)
                re_q50 = np.percentile(
                    psd_samples[:, :, i, j].real, 50, axis=0
                )
                re_q95 = np.percentile(
                    psd_samples[:, :, i, j].real, 95, axis=0
                )
                im_q05 = np.percentile(psd_samples[:, :, i, j].imag, 5, axis=0)
                im_q50 = np.percentile(
                    psd_samples[:, :, i, j].imag, 50, axis=0
                )
                im_q95 = np.percentile(
                    psd_samples[:, :, i, j].imag, 95, axis=0
                )
                ci_dict["re"][(i, j)] = (re_q05, re_q50, re_q95)
                ci_dict["im"][(i, j)] = (im_q05, im_q50, im_q95)
    return ci_dict


def _format_text(
    axes,
    channel_labels=None,
    show_coherence: bool = True,
    add_channel_labels: bool = True,
):
    p = axes.shape[0]
    if channel_labels is None:
        channel_labels = [str(i + 1) for i in range(p)]
    elif isinstance(channel_labels, str):
        channel_labels = list(channel_labels)
    assert (
        len(channel_labels) == p
    ), "channel_labels must match number of channels"

    if not add_channel_labels:
        return

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if not ax.axison:
                continue
            if show_coherence:
                if i == j:
                    lbl = f"$\\mathbf{{S}}_{{{channel_labels[i]}{channel_labels[j]}}}$"
                elif i > j:
                    lbl = f"$\\mathbf{{C}}_{{{channel_labels[i]}{channel_labels[j]}}}$"
                else:
                    continue
            else:
                base = (
                    f"\\mathbf{{S}}_{{{channel_labels[i]}{channel_labels[j]}}}"
                )
                if i < j:
                    lbl = f"$\\Re({base})$"
                elif i > j:
                    lbl = f"$\\Im({base})$"
                else:
                    lbl = f"${base}$"

            ax.text(
                0.96,
                0.93,
                lbl,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                weight="bold",
            )


def _ylabel_for(i: int, j: int, show_coherence: bool) -> str:
    if i == j:
        return "PSD [1/Hz]"
    if show_coherence and i > j:
        return "Coherence"  # unitless
    if not show_coherence and i > j:
        return "Re[PSD]"
    if not show_coherence and i < j:
        return "Im[PSD]"
    return ""  # hidden panel


def plot_psd_matrix(
    idata=None,
    ci_dict: dict | None = None,
    freq: np.ndarray | None = None,
    empirical_psd: EmpiricalPSD | None = None,
    true_psd: np.ndarray | None = None,
    outdir: str = ".",
    filename: str = "psd_matrix.png",
    dpi: int = 150,
    show_coherence: bool = True,
    channel_labels: list[str] | str | None = None,
    diag_yscale: str = "log",
    xscale: str = "linear",
    label: Optional[str] = None,
    model_color: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[np.ndarray] = None,
    save: bool = True,
    close: Optional[bool] = None,
):
    """
    Publication-ready multivariate PSD matrix plotter with adaptive per-axis y-labels.

    Returns
    -------
    (matplotlib.figure.Figure, np.ndarray)
        Figure and axes handle for further customisation or additional overlays.

    Parameters
    ----------
    label : str, optional
        Legend label used for the median PSD curve. Useful when overlaying
        multiple results on the same axes.
    fig, ax : optional
        Existing Matplotlib figure and axes to reuse for overlaid plots. When
        provided, the function skips layout adjustments and automatic saving.
    save : bool, default=True
        Whether to save the figure to ``outdir/filename`` when the figure is
        created inside this function.
    close : bool, optional
        Whether to close the figure at the end. Defaults to ``save`` when the
        figure is created inside this function; otherwise the figure is left
        open.
    """
    # ----- Extract/validate -----
    if idata is not None:
        # Check for required data
        extracted = extract_plotting_data(idata)
        quantiles = extracted.get("posterior_psd_matrix_quantiles")

        if quantiles is None:
            raise ValueError(
                "idata missing posterior_psd matrix quantiles for plotting"
            )

        freq = extracted.get("frequencies", freq)
        true_psd = extracted.get("true_psd", true_psd)

        # For multivariate data, try to extract empirical PSD if not provided
        if empirical_psd is None:
            empirical_psd = _extract_empirical_psd_from_idata(idata)

        percentiles = quantiles["percentile"]

        def _grab(arr: np.ndarray, target: float) -> np.ndarray:
            idx = int(np.argmin(np.abs(percentiles - target)))
            return arr[idx]

        real_q = quantiles["real"]
        imag_q = quantiles["imag"]
        coh_q = quantiles.get("coherence")

        ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}}
        n_channels = real_q.shape[2]
        for i in range(n_channels):
            for j in range(n_channels):
                q05_r = _grab(real_q[:, :, i, j], 5.0)
                q50_r = _grab(real_q[:, :, i, j], 50.0)
                q95_r = _grab(real_q[:, :, i, j], 95.0)
                ci_dict["psd"][(i, j)] = (q05_r, q50_r, q95_r)
                if i != j:
                    q05_im = _grab(imag_q[:, :, i, j], 5.0)
                    q50_im = _grab(imag_q[:, :, i, j], 50.0)
                    q95_im = _grab(imag_q[:, :, i, j], 95.0)
                    ci_dict["re"][(i, j)] = (q05_r, q50_r, q95_r)
                    ci_dict["im"][(i, j)] = (q05_im, q50_im, q95_im)

        if coh_q is not None and show_coherence:
            for i in range(n_channels):
                for j in range(n_channels):
                    q05_c = _grab(coh_q[:, :, i, j], 5.0)
                    q50_c = _grab(coh_q[:, :, i, j], 50.0)
                    q95_c = _grab(coh_q[:, :, i, j], 95.0)
                    ci_dict["coh"][(i, j)] = (q05_c, q50_c, q95_c)

    elif ci_dict is None:
        raise ValueError("Provide either `idata` or `ci_dict`.")

    if freq is None:
        raise ValueError("Frequency array `freq` is required.")

    if true_psd is not None:
        true_psd = np.asarray(true_psd)
        if true_psd.shape[0] != len(freq):
            logger.warning(
                "Skipping true PSD overlay: expected %d frequency bins, got %d.",
                len(freq),
                true_psd.shape[0],
            )
            true_psd = None

    if empirical_psd is not None:
        n_channels = empirical_psd.psd.shape[1]
    elif "psd" in ci_dict and len(ci_dict["psd"]) > 0:
        n_channels = max(max(i, j) for (i, j) in ci_dict["psd"].keys()) + 1
    else:
        raise ValueError("Could not infer number of channels.")

    # ----- Figure -----
    fig_provided = fig is not None and ax is not None
    if fig_provided:
        axes = np.asarray(ax)
        if axes.shape != (n_channels, n_channels):
            raise ValueError(
                f"Provided axes have shape {axes.shape}, expected "
                f"({n_channels}, {n_channels})."
            )
        created_fig = False
    else:
        fig, axes = plt.subplots(
            n_channels,
            n_channels,
            figsize=(3.9 * n_channels, 3.9 * n_channels),
        )
        if n_channels == 1:
            axes = np.array([[axes]])
        created_fig = True
    axes = np.asarray(axes)

    # ----- Plot -----
    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]
            ax.set_xscale(xscale)
            ax.tick_params(which="both", direction="in", top=True, right=True)

            if i == j:  # auto-PSDs
                q05, q50, q95 = ci_dict["psd"][(i, i)]
                if empirical_psd is not None:
                    ax.plot(
                        empirical_psd.freq,
                        empirical_psd.psd[:, i, i].real,
                        color="0.4",
                        lw=1.0,
                        alpha=0.7,
                        ls="--",
                        label="Empirical",
                        zorder=-5,
                    )
                line_kwargs = {"lw": 1.5}
                if label is not None:
                    line_kwargs["label"] = label
                else:
                    line_kwargs["label"] = "Median"

                if model_color is not None:
                    line_kwargs["color"] = model_color
                elif label is None:
                    line_kwargs.setdefault("color", "tab:blue")
                line = ax.plot(freq, q50, **line_kwargs)[0]
                ax.fill_between(
                    freq,
                    q05,
                    q95,
                    color=line.get_color(),
                    alpha=0.25,
                    zorder=line.get_zorder() - 1,
                )
                if true_psd is not None:
                    ax.plot(
                        freq,
                        true_psd[:, i, i].real,
                        color="k",
                        lw=1.2,
                        label="True",
                    )
                ax.set_yscale(diag_yscale)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=9)

            elif i > j:  # lower triangle
                if show_coherence:
                    if "coh" in ci_dict and (i, j) in ci_dict["coh"]:
                        q05, q50, q95 = ci_dict["coh"][(i, j)]
                        # draw empirical first so it's visible under the band
                        if empirical_psd is not None:
                            ax.plot(
                                empirical_psd.freq,
                                empirical_psd.coherence[:, i, j],
                                color="0.3",
                                lw=1.0,
                                ls="--",
                                label="Empirical",
                                alpha=0.6,
                                zorder=-5,
                            )
                        coh_color = model_color or "tab:blue"
                        ax.fill_between(
                            freq, q05, q95, color=coh_color, alpha=0.25
                        )
                        ax.plot(
                            freq,
                            q50,
                            color=coh_color,
                            lw=1.5,
                            label="Median" if label is None else label,
                        )
                        if true_psd is not None:
                            true_coh = np.abs(true_psd[:, i, j]) ** 2 / (
                                np.abs(true_psd[:, i, i])
                                * np.abs(true_psd[:, j, j])
                            )
                            ax.plot(
                                freq, true_coh, color="k", lw=1.2, label="True"
                            )
                        ax.set_ylim(0, 1)
                    else:
                        raise ValueError(
                            "ci_dict missing coherence (i,j)={i,j}"
                        )
                else:
                    q05, q50, q95 = ci_dict["re"][(i, j)]
                    ax.fill_between(
                        freq, q05, q95, color="tab:green", alpha=0.25
                    )
                    ax.plot(freq, q50, color="tab:green", lw=1.5)
                    if empirical_psd is not None:
                        ax.plot(
                            empirical_psd.freq,
                            empirical_psd.psd[:, i, j].real,
                            "k--",
                            lw=1.0,
                            alpha=0.6,
                            zorder=-5,
                        )
                    if true_psd is not None:
                        ax.plot(freq, true_psd[:, i, j].real, "k", lw=1.2)

            elif not show_coherence and i < j:  # upper triangle imag parts
                q05, q50, q95 = ci_dict["im"][(i, j)]
                ax.fill_between(freq, q05, q95, color="tab:orange", alpha=0.25)
                ax.plot(freq, q50, color="tab:orange", lw=1.5)
                if empirical_psd is not None:
                    ax.plot(
                        empirical_psd.freq,
                        empirical_psd.psd[:, i, j].imag,
                        "k--",
                        lw=1.0,
                        alpha=0.6,
                        zorder=-5,
                    )
                if true_psd is not None:
                    ax.plot(freq, true_psd[:, i, j].imag, "k", lw=1.2)
            else:
                ax.axis("off")

            # adaptive y-labels only for visible panels
            ylab = _ylabel_for(i, j, show_coherence)
            if ylab:
                ax.set_ylabel(ylab, fontsize=11)

            if i == n_channels - 1:
                ax.set_xlabel("Frequency [Hz]", fontsize=11)

    if created_fig:
        _format_text(
            axes, channel_labels=channel_labels, show_coherence=show_coherence
        )
        plt.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.98,
            bottom=0.10,
            wspace=0.30,
            hspace=0.30,
        )

    effective_save = save and created_fig and outdir is not None and filename
    if effective_save:
        fig.savefig(f"{outdir}/{filename}", dpi=dpi, bbox_inches="tight")

    close_fig = (
        close if close is not None else (created_fig and effective_save)
    )
    if close_fig:
        plt.close(fig)

    return fig, axes
