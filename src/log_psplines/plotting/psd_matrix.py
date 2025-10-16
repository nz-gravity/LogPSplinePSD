import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD
from ..logger import logger
from .base import (
    COLORS,
    PlotConfig,
    compute_coherence_ci,
    compute_cross_spectra_ci,
    extract_plotting_data,
    setup_plot_style,
    validate_plotting_data,
)

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
                    (n_freq, n_channels, n_channels), dtype=complex
                )

                # For now, assume diagonal structure (auto-PSDs only)
                # This is a simplified version - in practice, you might need
                # more sophisticated logic to handle cross-channel relationships
                for i in range(n_channels):
                    # Compute periodogram for each channel
                    # This is a placeholder - you may need to adjust based on
                    # how your multivariate data is actually structured
                    channel_fft = (
                        fft_complex[:, i]
                        if fft_complex.ndim > 1
                        else fft_complex
                    )
                    psd_matrix[:, i, i] = np.abs(channel_fft) ** 2

                # Create EmpiricalPSD object
                return EmpiricalPSD(
                    freq=freq, psd=psd_matrix, channels=channels
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
    *,
    fig=None,
    axes=None,
    save: bool = True,
    return_fig: bool = False,
):
    """
    Publication-ready multivariate PSD matrix plotter with adaptive per-axis y-labels.
    """
    # ----- Extract/validate -----
    if idata is not None:
        # Check for required data
        if "psd_matrix" not in idata.posterior_psd:
            raise ValueError("idata missing posterior_psd['psd_matrix']")

        # Extract data using shared utility
        extracted_data = extract_plotting_data(idata)
        psd_samples = extracted_data.get("posterior_psd_matrix")

        if psd_samples is None:
            raise ValueError("Could not extract PSD matrix from idata")

        freq = extracted_data.get("frequencies", freq)
        true_psd = extracted_data.get("true_psd", true_psd)

        # For multivariate data, try to extract empirical PSD if not provided
        if empirical_psd is None:
            empirical_psd = _extract_empirical_psd_from_idata(idata)

        # Compute confidence intervals using shared utilities
        if show_coherence:
            ci_dict = dict(coh=compute_coherence_ci(psd_samples))
            # Add PSD diagonal elements
            ci_dict["psd"] = {}
            for i in range(psd_samples.shape[2]):
                q05 = np.percentile(psd_samples[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, i].real, 95, axis=0)
                ci_dict["psd"][(i, i)] = (q05, q50, q95)

        else:
            real_dict, imag_dict = compute_cross_spectra_ci(psd_samples)
            ci_dict = {"psd": {}, "coh": {}, "re": real_dict, "im": imag_dict}
            # Add PSD diagonal elements
            for i in range(psd_samples.shape[2]):
                q05 = np.percentile(psd_samples[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, i].real, 95, axis=0)
                ci_dict["psd"][(i, i)] = (q05, q50, q95)

    elif ci_dict is None:
        raise ValueError("Provide either `idata` or `ci_dict`.")

    if freq is None:
        raise ValueError("Frequency array `freq` is required.")

    if empirical_psd is not None:
        n_channels = empirical_psd.psd.shape[1]
    elif "psd" in ci_dict and len(ci_dict["psd"]) > 0:
        n_channels = max(max(i, j) for (i, j) in ci_dict["psd"].keys()) + 1
    else:
        raise ValueError("Could not infer number of channels.")

    # ----- Figure -----
    created_axes = False
    if axes is None:
        if fig is None:
            fig, axes = plt.subplots(
                n_channels,
                n_channels,
                figsize=(3.9 * n_channels, 3.9 * n_channels),
            )
        else:
            axes = fig.subplots(n_channels, n_channels)
        created_axes = True
    else:
        if fig is None:
            fig = axes.flat[0].figure
    if n_channels == 1:
        axes = np.array([[axes]])

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
                ax.fill_between(freq, q05, q95, color="tab:blue", alpha=0.25)
                ax.plot(freq, q50, color="tab:blue", lw=1.5, label="Median")
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
                        ax.fill_between(
                            freq, q05, q95, color="tab:blue", alpha=0.25
                        )
                        ax.plot(
                            freq, q50, color="tab:blue", lw=1.5, label="Median"
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

    _format_text(
        axes, channel_labels=channel_labels, show_coherence=show_coherence
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.98, bottom=0.10, wspace=0.30, hspace=0.30
    )
    if save:
        fig.savefig(f"{outdir}/{filename}", dpi=dpi, bbox_inches="tight")
    if not return_fig and created_axes:
        plt.close(fig)
    return fig, axes
