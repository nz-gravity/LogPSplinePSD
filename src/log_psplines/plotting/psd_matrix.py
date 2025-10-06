import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


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
):
    """
    Publication-ready multivariate PSD matrix plotter with adaptive per-axis y-labels.
    """
    # ----- Extract/validate -----
    if idata is not None:
        if "psd_matrix" not in idata.posterior_psd:
            raise ValueError("idata missing posterior_psd['psd_matrix']")
        psd_samples = idata.posterior_psd["psd_matrix"].values
        freq = idata.attrs.get("frequencies", freq)
        ci_dict = _pack_ci_dict(psd_samples, show_coherence)
        if "true_psd" in idata.attrs and true_psd is None:
            true_psd = idata.attrs["true_psd"]
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
    fig, axes = plt.subplots(
        n_channels, n_channels, figsize=(3.9 * n_channels, 3.9 * n_channels)
    )
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
                    if (i, j) in ci_dict["coh"]:
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
    plt.subplots_adjust(
        left=0.12, right=0.98, top=0.98, bottom=0.10, wspace=0.30, hspace=0.30
    )
    plt.savefig(f"{outdir}/{filename}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
