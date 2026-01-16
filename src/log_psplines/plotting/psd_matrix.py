from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD, _get_coherence
from ..logger import logger
from .base import extract_plotting_data, setup_plot_style

# Setup default plot styling
setup_plot_style()


EMPIRICAL_KWGS = dict(
    color="0.4",
    lw=1.0,
    alpha=0.3,
    ls="--",
    label="Empirical",
    zorder=-5,
)
TRUE_KWGS = dict(color="k", lw=1.2, label="Analytical", zorder=-2)


def _quantiles_to_ci_dict(
    quantiles: dict,
    show_coherence: bool,
    show_csd_magnitude: bool,
) -> dict:
    """Convert stored quantiles into CI dictionaries."""
    percentiles = np.asarray(quantiles["percentile"])
    real_q = np.asarray(quantiles["real"])
    imag_q = np.asarray(quantiles["imag"])
    coh_q = quantiles.get("coherence")

    def _grab(arr: np.ndarray, target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(percentiles - target)))
        return arr[idx]

    ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}, "mag": {}}
    n_channels = real_q.shape[2]
    for i in range(n_channels):
        for j in range(n_channels):
            q05_r = _grab(real_q[:, :, i, i], 5.0) if i == j else None
            q95_r = _grab(real_q[:, :, i, i], 95.0) if i == j else None
            if i == j:
                q50_r = _grab(real_q[:, :, i, i], 50.0)
                ci_dict["psd"][(i, i)] = (q05_r, q50_r, q95_r)
                continue

            q05_re = _grab(real_q[:, :, i, j], 5.0)
            q50_re = _grab(real_q[:, :, i, j], 50.0)
            q95_re = _grab(real_q[:, :, i, j], 95.0)
            q05_im = _grab(imag_q[:, :, i, j], 5.0)
            q50_im = _grab(imag_q[:, :, i, j], 50.0)
            q95_im = _grab(imag_q[:, :, i, j], 95.0)
            ci_dict["re"][(i, j)] = (q05_re, q50_re, q95_re)
            ci_dict["im"][(i, j)] = (q05_im, q50_im, q95_im)

            if show_coherence and coh_q is not None and i > j:
                coh05 = _grab(coh_q[:, :, i, j], 5.0)
                coh50 = _grab(coh_q[:, :, i, j], 50.0)
                coh95 = _grab(coh_q[:, :, i, j], 95.0)
                ci_dict["coh"][(i, j)] = (coh05, coh50, coh95)
            if show_csd_magnitude and i > j:
                mag_q05 = np.sqrt(np.maximum(q05_re**2 + q05_im**2, 0.0))
                mag_q50 = np.sqrt(np.maximum(q50_re**2 + q50_im**2, 0.0))
                mag_q95 = np.sqrt(np.maximum(q95_re**2 + q95_im**2, 0.0))
                ci_dict["mag"][(i, j)] = (mag_q05, mag_q50, mag_q95)

    return ci_dict


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

            # Prefer directly stored periodogram (already coarse-grained)
            if "periodogram" in obs_data:
                periodogram = obs_data["periodogram"]
                freq = periodogram.coords["freq"].values
                channels = periodogram.coords["channels"].values
                psd_matrix = periodogram.values
                coherence = _get_coherence(psd_matrix)
                return EmpiricalPSD(
                    freq=freq,
                    psd=psd_matrix,
                    coherence=coherence,
                    channels=channels,
                )

            # Fallback to reconstructing from FFT components
            if all(
                key in obs_data
                for key in ["freq", "channels", "fft_re", "fft_im"]
            ):
                freq = obs_data["freq"].values
                channels = obs_data["channels"].values
                fft_re = obs_data["fft_re"].values
                fft_im = obs_data["fft_im"].values

                fft_complex = fft_re + 1j * fft_im
                n_channels = len(channels)
                n_freq = len(freq)

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


def _pack_ci_dict(
    psd_samples,
    show_coherence: bool,
    show_csd_magnitude: bool = False,
):
    """Compute 5/50/95% bands for diag PSDs and requested cross terms."""
    ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}, "mag": {}}
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
            elif show_csd_magnitude and i > j:
                mag = np.abs(psd_samples[:, :, i, j])
                q05 = np.percentile(mag, 5, axis=0)
                q50 = np.percentile(mag, 50, axis=0)
                q95 = np.percentile(mag, 95, axis=0)
                ci_dict["mag"][(i, j)] = (q05, q50, q95)
            elif not show_coherence and not show_csd_magnitude:
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
    show_csd_magnitude: bool = False,
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
            elif show_csd_magnitude:
                if i == j:
                    lbl = f"$\\mathbf{{S}}_{{{channel_labels[i]}{channel_labels[j]}}}$"
                elif i > j:
                    lbl = f"$|\\mathbf{{S}}_{{{channel_labels[i]}{channel_labels[j]}}}|$"
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


def _ylabel_for(
    i: int,
    j: int,
    show_coherence: bool,
    show_csd_magnitude: bool,
    psd_unit_label: str,
) -> str:
    if i == j:
        return f"PSD [{psd_unit_label}]"
    if show_coherence and i > j:
        return "Coherence"  # unitless
    if show_csd_magnitude and i > j:
        return f"|CSD| [{psd_unit_label}]"
    if not show_coherence and i > j:
        return f"Re[PSD] [{psd_unit_label}]"
    if not show_coherence and i < j:
        return f"Im[PSD] [{psd_unit_label}]"
    return ""  # hidden panel


def _resolve_scale(
    freq: np.ndarray,
    psd_scale: np.ndarray | float | Callable[[np.ndarray], np.ndarray] | None,
    *,
    base_freq: np.ndarray | None = None,
) -> np.ndarray | None:
    if psd_scale is None:
        return None
    if callable(psd_scale):
        scale = np.asarray(psd_scale(freq))
    else:
        scale = np.asarray(psd_scale)
        if scale.ndim == 0:
            scale = np.full_like(freq, float(scale), dtype=float)
        else:
            if base_freq is None:
                base_freq = freq
            if scale.shape != base_freq.shape:
                raise ValueError("psd_scale array must match base_freq shape.")
            if base_freq.shape != freq.shape or not np.allclose(
                base_freq, freq
            ):
                scale = np.interp(freq, base_freq, scale)
    if scale.shape != freq.shape:
        raise ValueError("psd_scale must match frequency shape.")
    if np.any(scale < 0):
        raise ValueError("psd_scale must be non-negative.")
    if np.any(scale == 0):
        zero_mask = scale == 0
        if not np.allclose(freq[zero_mask], 0.0):
            raise ValueError("psd_scale has zeros at nonzero frequencies.")
        scale = scale.copy()
        scale[zero_mask] = np.nan
    return scale


def _scale_ci_dict(ci_dict: dict, scale: np.ndarray) -> dict:
    scaled = {key: dict(val) for key, val in ci_dict.items()}
    for key in ("psd", "re", "im", "mag"):
        if key not in ci_dict:
            continue
        for idx, (q05, q50, q95) in ci_dict[key].items():
            scaled[key][idx] = (q05 * scale, q50 * scale, q95 * scale)
    return scaled


def _scale_empirical_psd(
    empirical_psd: EmpiricalPSD, scale: np.ndarray
) -> EmpiricalPSD:
    psd = empirical_psd.psd * scale[:, None, None]
    return EmpiricalPSD(
        freq=empirical_psd.freq,
        psd=psd,
        coherence=empirical_psd.coherence,
        channels=empirical_psd.channels,
    )


def plot_psd_matrix(
    idata=None,
    ci_dict: dict | None = None,
    freq: np.ndarray | None = None,
    empirical_psd: EmpiricalPSD | None = None,
    extra_empirical_psd: list[EmpiricalPSD] | None = None,
    extra_empirical_labels: list[str] | None = None,
    extra_empirical_styles: list[dict] | None = None,
    true_psd: np.ndarray | None = None,
    outdir: str = ".",
    filename: str = "psd_matrix.png",
    dpi: int = 150,
    show_coherence: bool = True,
    show_csd_magnitude: bool = False,
    channel_labels: list[str] | str | None = None,
    diag_yscale: str = "log",
    offdiag_yscale: str = "linear",
    xscale: str = "linear",
    label: Optional[str] = None,
    model_color: Optional[str] = "tab:blue",
    fig: Optional[plt.Figure] = None,
    ax: Optional[np.ndarray] = None,
    save: bool = True,
    close: Optional[bool] = None,
    overlay_vi: bool = False,
    vi_color: Optional[str] = "tab:orange",
    vi_label: str = "VI median",
    vi_alpha: float = 0.2,
    freq_range: Optional[tuple[float, float]] = None,
    psd_scale: (
        np.ndarray | float | Callable[[np.ndarray], np.ndarray] | None
    ) = None,
    psd_unit_label: str = "1/Hz",
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
    show_csd_magnitude : bool, optional
        When ``True`` the lower-triangular panels display |CSD_ij| instead of
        coherence or Re/Im parts.
    psd_scale : array-like or callable, optional
        Scale factor applied to PSD/CSD panels. If callable, it is evaluated
        as ``psd_scale(freq)`` for each frequency grid.
    psd_unit_label : str, optional
        Unit label for PSD/CSD axes, used in y-axis labels.
    overlay_vi : bool, optional
        When ``True`` and VI diagnostics are present, overlay the VI median /
        quantile bands alongside posterior results for comparison.
    vi_color : str, optional
        Line/fill color for the VI overlay.
    vi_label : str, optional
        Legend label for the VI median when overlaying.
    """
    # ----- Extract/validate -----
    if show_coherence and show_csd_magnitude:
        raise ValueError(
            "Choose either coherence display or |CSD| magnitude, not both."
        )

    vi_ci_dict = None

    if idata is not None:
        # Check for required data
        extracted = extract_plotting_data(idata)
        quantiles = extracted.get("posterior_psd_matrix_quantiles")
        vi_quantiles = extracted.get("vi_psd_matrix_quantiles")

        using_vi_only = False
        if quantiles is None:
            quantiles = vi_quantiles
            using_vi_only = quantiles is not None

        if quantiles is None:
            raise ValueError(
                "idata missing posterior_psd matrix quantiles for plotting"
            )

        freq = extracted.get("frequencies", freq)
        true_psd = extracted.get("true_psd", true_psd)

        # For multivariate data, try to extract empirical PSD if not provided
        if empirical_psd is None:
            empirical_psd = _extract_empirical_psd_from_idata(idata)

        ci_dict = _quantiles_to_ci_dict(
            quantiles,
            show_coherence=show_coherence,
            show_csd_magnitude=show_csd_magnitude,
        )

        if overlay_vi and not using_vi_only and vi_quantiles is not None:
            vi_ci_dict = _quantiles_to_ci_dict(
                vi_quantiles,
                show_coherence=show_coherence,
                show_csd_magnitude=show_csd_magnitude,
            )
        elif overlay_vi and vi_quantiles is None:
            logger.warning(
                "overlay_vi requested but VI quantiles unavailable; ignoring."
            )

    elif ci_dict is None:
        raise ValueError("Provide either `idata` or `ci_dict`.")

    if freq is None:
        raise ValueError("Frequency array `freq` is required.")

    if true_psd is not None:
        true_psd = np.asarray(true_psd)
        if true_psd.shape[0] != len(freq):
            logger.warning(
                f"Skipping true PSD overlay: expected {len(freq)} frequency bins, got {true_psd.shape[0]}.",
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
    # Normalise optional extra empirical inputs
    if extra_empirical_psd is None:
        extra_empirical_psd = []
    if extra_empirical_labels is None:
        extra_empirical_labels = []
    if extra_empirical_styles is None:
        extra_empirical_styles = []

    scale_main = _resolve_scale(freq, psd_scale)
    if scale_main is not None:
        ci_dict = _scale_ci_dict(ci_dict, scale_main)
        if vi_ci_dict is not None:
            vi_ci_dict = _scale_ci_dict(vi_ci_dict, scale_main)
        if true_psd is not None:
            true_psd = true_psd * scale_main[:, None, None]
        if empirical_psd is not None:
            scale_emp = _resolve_scale(
                empirical_psd.freq, psd_scale, base_freq=freq
            )
            empirical_psd = _scale_empirical_psd(empirical_psd, scale_emp)
        if extra_empirical_psd:
            scaled_extra = []
            for extra in extra_empirical_psd:
                scale_extra = _resolve_scale(
                    extra.freq, psd_scale, base_freq=freq
                )
                scaled_extra.append(_scale_empirical_psd(extra, scale_extra))
            extra_empirical_psd = scaled_extra

    vi_label_added = False
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
                        **EMPIRICAL_KWGS,
                    )
                # Additional empirical overlays (e.g. Welch PSD)
                for idx, extra_emp in enumerate(extra_empirical_psd):
                    kw = dict(EMPIRICAL_KWGS)
                    if idx < len(extra_empirical_styles):
                        kw.update(extra_empirical_styles[idx] or {})
                    if idx < len(extra_empirical_labels):
                        kw["label"] = extra_empirical_labels[idx]
                    else:
                        kw.setdefault("label", f"Empirical {idx + 2}")
                    ax.plot(
                        extra_emp.freq,
                        extra_emp.psd[:, i, i].real,
                        **kw,
                    )
                line_kwargs = {"lw": 1.5}
                if label is not None:
                    line_kwargs["label"] = label
                else:
                    line_kwargs["label"] = (
                        "Posterior median" if overlay_vi else "Median"
                    )

                line_kwargs["color"] = model_color
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
                        **TRUE_KWGS,
                    )
                if vi_ci_dict and (i, i) in vi_ci_dict["psd"]:
                    vi_q05, vi_q50, vi_q95 = vi_ci_dict["psd"][(i, i)]
                    ax.fill_between(
                        freq,
                        vi_q05,
                        vi_q95,
                        color=vi_color,
                        alpha=vi_alpha,
                        zorder=line.get_zorder() - 2,
                    )
                    ax.plot(
                        freq,
                        vi_q50,
                        color=vi_color,
                        lw=1.3,
                        ls="--",
                        label=vi_label if not vi_label_added else None,
                    )
                    vi_label_added = True
                ax.set_yscale(diag_yscale)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=9)

            elif i > j:  # lower triangle cross terms
                if show_coherence:
                    if "coh" in ci_dict and (i, j) in ci_dict["coh"]:
                        q05, q50, q95 = ci_dict["coh"][(i, j)]
                        # draw empirical first so it's visible under the band
                        if empirical_psd is not None:
                            ax.plot(
                                empirical_psd.freq,
                                empirical_psd.coherence[:, i, j],
                                **EMPIRICAL_KWGS,
                            )
                        for idx, extra_emp in enumerate(extra_empirical_psd):
                            kw = dict(EMPIRICAL_KWGS)
                            if idx < len(extra_empirical_styles):
                                kw.update(extra_empirical_styles[idx] or {})
                            if idx < len(extra_empirical_labels):
                                kw["label"] = extra_empirical_labels[idx]
                            else:
                                kw.setdefault("label", f"Empirical {idx + 2}")
                            ax.plot(
                                extra_emp.freq,
                                extra_emp.coherence[:, i, j],
                                **kw,
                            )
                        ax.fill_between(
                            freq, q05, q95, color=model_color, alpha=0.25
                        )
                        ax.plot(
                            freq,
                            q50,
                            color=model_color,
                            lw=1.5,
                            label="Median" if label is None else label,
                        )
                        if true_psd is not None:
                            true_coh = np.abs(true_psd[:, i, j]) ** 2 / (
                                np.abs(true_psd[:, i, i])
                                * np.abs(true_psd[:, j, j])
                            )
                            ax.plot(
                                freq,
                                true_coh,
                                **TRUE_KWGS,
                            )
                        ax.set_ylim(0, 1)
                        if (
                            vi_ci_dict
                            and "coh" in vi_ci_dict
                            and (i, j) in vi_ci_dict["coh"]
                        ):
                            vi_q05, vi_q50, vi_q95 = vi_ci_dict["coh"][(i, j)]
                            ax.fill_between(
                                freq,
                                vi_q05,
                                vi_q95,
                                color=vi_color,
                                alpha=vi_alpha,
                                zorder=1,
                            )
                            ax.plot(
                                freq,
                                vi_q50,
                                color=vi_color,
                                lw=1.2,
                                ls="--",
                                label=vi_label if not vi_label_added else None,
                            )
                            vi_label_added = True
                    else:
                        raise ValueError(
                            "ci_dict missing coherence (i,j)={i,j}"
                        )
                elif show_csd_magnitude:
                    if "mag" not in ci_dict or (i, j) not in ci_dict["mag"]:
                        raise ValueError(
                            f"ci_dict missing |CSD| quantiles for (i,j)=({i},{j})"
                        )
                    q05, q50, q95 = ci_dict["mag"][(i, j)]
                    ax.fill_between(
                        freq, q05, q95, color=model_color, alpha=0.25
                    )
                    ax.plot(
                        freq,
                        q50,
                        color=model_color,
                        lw=1.5,
                        label="Median" if label is None else label,
                    )
                    if empirical_psd is not None:
                        ax.plot(
                            empirical_psd.freq,
                            np.abs(empirical_psd.psd[:, i, j]),
                            **EMPIRICAL_KWGS,
                        )
                    for idx, extra_emp in enumerate(extra_empirical_psd):
                        kw = dict(EMPIRICAL_KWGS)
                        if idx < len(extra_empirical_styles):
                            kw.update(extra_empirical_styles[idx] or {})
                        if idx < len(extra_empirical_labels):
                            kw["label"] = extra_empirical_labels[idx]
                        else:
                            kw.setdefault("label", f"Empirical {idx + 2}")
                        ax.plot(
                            extra_emp.freq,
                            np.abs(extra_emp.psd[:, i, j]),
                            **kw,
                        )
                    if true_psd is not None:
                        ax.plot(
                            freq,
                            np.abs(true_psd[:, i, j]),
                            **TRUE_KWGS,
                        )
                    if vi_ci_dict and (i, j) in vi_ci_dict["mag"]:
                        vi_q05, vi_q50, vi_q95 = vi_ci_dict["mag"][(i, j)]
                        ax.fill_between(
                            freq,
                            vi_q05,
                            vi_q95,
                            color=vi_color,
                            alpha=vi_alpha,
                        )
                        ax.plot(
                            freq,
                            vi_q50,
                            color=vi_color,
                            lw=1.3,
                            ls="--",
                            label=vi_label if not vi_label_added else None,
                        )
                        vi_label_added = True
                else:
                    q05, q50, q95 = ci_dict["re"][(i, j)]
                    ax.fill_between(
                        freq, q05, q95, color=model_color, alpha=0.25
                    )
                    ax.plot(freq, q50, color=model_color, lw=1.5)
                    if empirical_psd is not None:
                        ax.plot(
                            empirical_psd.freq,
                            empirical_psd.psd[:, i, j].real,
                            **EMPIRICAL_KWGS,
                        )
                    for idx, extra_emp in enumerate(extra_empirical_psd):
                        kw = dict(EMPIRICAL_KWGS)
                        if idx < len(extra_empirical_styles):
                            kw.update(extra_empirical_styles[idx] or {})
                        if idx < len(extra_empirical_labels):
                            kw["label"] = extra_empirical_labels[idx]
                        else:
                            kw.setdefault("label", f"Empirical {idx + 2}")
                        ax.plot(
                            extra_emp.freq,
                            extra_emp.psd[:, i, j].real,
                            **kw,
                        )
                    if true_psd is not None:
                        ax.plot(freq, true_psd[:, i, j].real, **TRUE_KWGS)
                    if vi_ci_dict and (i, j) in vi_ci_dict["re"]:
                        vi_q05, vi_q50, vi_q95 = vi_ci_dict["re"][(i, j)]
                        ax.fill_between(
                            freq,
                            vi_q05,
                            vi_q95,
                            color=vi_color,
                            alpha=vi_alpha,
                        )
                        ax.plot(
                            freq,
                            vi_q50,
                            color=vi_color,
                            lw=1.3,
                            ls="--",
                            label=vi_label if not vi_label_added else None,
                        )
                        vi_label_added = True
                ax.set_yscale(offdiag_yscale)

            elif not show_coherence and not show_csd_magnitude and i < j:
                q05, q50, q95 = ci_dict["im"][(i, j)]
                ax.fill_between(freq, q05, q95, color=model_color, alpha=0.25)
                ax.plot(freq, q50, color=model_color, lw=1.5)
                if empirical_psd is not None:
                    ax.plot(
                        empirical_psd.freq,
                        empirical_psd.psd[:, i, j].imag,
                        **EMPIRICAL_KWGS,
                    )
                for idx, extra_emp in enumerate(extra_empirical_psd):
                    kw = dict(EMPIRICAL_KWGS)
                    if idx < len(extra_empirical_styles):
                        kw.update(extra_empirical_styles[idx] or {})
                    if idx < len(extra_empirical_labels):
                        kw["label"] = extra_empirical_labels[idx]
                    else:
                        kw.setdefault("label", f"Empirical {idx + 2}")
                    ax.plot(
                        extra_emp.freq,
                        extra_emp.psd[:, i, j].imag,
                        **kw,
                    )
                if true_psd is not None:
                    ax.plot(freq, true_psd[:, i, j].imag, **TRUE_KWGS)
                if (
                    not show_coherence
                    and not show_csd_magnitude
                    and vi_ci_dict
                    and (i, j) in vi_ci_dict["im"]
                ):
                    vi_q05, vi_q50, vi_q95 = vi_ci_dict["im"][(i, j)]
                    ax.fill_between(
                        freq,
                        vi_q05,
                        vi_q95,
                        color=vi_color,
                        alpha=vi_alpha,
                    )
                    ax.plot(
                        freq,
                        vi_q50,
                        color=vi_color,
                        lw=1.3,
                        ls="--",
                        label=vi_label if not vi_label_added else None,
                    )
                    vi_label_added = True
            else:
                ax.axis("off")

            # adaptive y-labels only for visible panels
            ylab = _ylabel_for(
                i,
                j,
                show_coherence,
                show_csd_magnitude,
                psd_unit_label,
            )
            if ylab:
                ax.set_ylabel(ylab, fontsize=11)

            if i == n_channels - 1:
                ax.set_xlabel("Frequency [Hz]", fontsize=11)

    if freq_range is not None:
        for i in range(n_channels):
            for j in range(n_channels):
                ax = axes[i, j]
                if ax.axison:
                    ax.set_xlim(freq_range)

    if created_fig:
        _format_text(
            axes,
            channel_labels=channel_labels,
            show_coherence=show_coherence,
            show_csd_magnitude=show_csd_magnitude,
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
