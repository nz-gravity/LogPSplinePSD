import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import mne
import numpy as np
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.preprocessing.coarse_grain.config import CoarseGrainConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CHANNEL_COLORS = {
    "EEG 005": "tab:orange",
    "EEG 021": "tab:blue",
    "EEG 041": "tab:green",
    "EEG 050": "tab:red",
    "EEG 058": "tab:purple",
}


def _load_sample_eeg():
    """Return the MNE sample EEG data."""
    data_path = Path(mne.datasets.sample.data_path())
    raw = mne.io.read_raw_fif(
        data_path / "MEG" / "sample" / "sample_audvis_raw.fif",
        preload=True,
    )
    raw.pick_types(meg=False, eeg=True)
    return raw


def _select_named_channels(raw, desired_names=None):
    available = raw.ch_names
    if desired_names is None:
        desired_names = [
            "EEG 005",
            "EEG 021",  # frontal/right
            "EEG 041",  # vertex
            "EEG 050",  # parietal
            "EEG 058",  # occipital
        ]
    selected_indices = []
    selected_names = []
    missing = []
    for name in desired_names:
        if name in available:
            selected_indices.append(available.index(name))
            selected_names.append(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"The following desired EEG channels were not found: {missing}. "
            "Please update `desired_names` to match your dataset."
        )
    return selected_indices, selected_names


def _add_sensor_inset(
    ax,
    info,
    channel_names,
    color_map,
    width="18%",
    height="45%",
    loc="upper left",
    **kwargs,
):
    dummy_data = np.zeros((len(channel_names), 1))
    info_copy = info.copy()
    evoked = mne.EvokedArray(dummy_data, info_copy, tmin=0.0)
    inset = inset_axes(ax, width=width, height=height, loc=loc, **kwargs)
    evoked.plot_sensors(
        axes=inset,
        kind="topomap",
        show_names=False,
        title="",
        show=False,
        to_sphere=False,
    )
    scatter = None
    for artist in inset.collections:
        if hasattr(artist, "get_offsets"):
            scatter = artist
            break
    if scatter is not None:
        colors = np.array(
            [
                color_map.get(ch, CUSTOM_CHANNEL_COLORS.get(ch, "k"))
                for ch in channel_names
            ]
        )
        scatter.set_facecolors(colors)
        scatter.set_edgecolors(colors)
    return inset


def _save_time_domain_butterfly(raw, selected_indices, channel_names):
    info = mne.pick_info(raw.info, sel=selected_indices)
    selected_data = raw.get_data(picks=selected_indices)

    fs = int(raw.info["sfreq"])
    n_times = selected_data.shape[1]
    n_times_plot = min(fs, n_times)

    if info.get_montage() is None and raw.get_montage() is not None:
        info.set_montage(raw.get_montage())

    snippet = selected_data[:, :n_times_plot]
    times = np.arange(n_times_plot) / fs
    data_uv = snippet * 1e6

    fig, ax = plt.subplots(figsize=(12, 5.5))
    color_map = {}
    for idx, ch_name in enumerate(channel_names):
        color = CUSTOM_CHANNEL_COLORS.get(ch_name, f"C{idx}")
        color_map[ch_name] = color
        ax.plot(times, data_uv[idx], color=color, label=ch_name, linewidth=1.5)

    ax.set_title("EEG (selected channels)", fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("µV", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    leg = ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    _add_sensor_inset(ax, info, channel_names, color_map)

    fig.tight_layout()
    path = RESULTS_DIR / "time_domain_evoked_butterfly.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path, selected_data, fs, color_map, info


def _run_multivar_pspline(selected_data, fs):
    multivar_ts = MultivariateTimeseries(
        y=selected_data.T,
        t=np.arange(selected_data.shape[1]) / fs,
    )

    idata_file = RESULTS_DIR / "multivar_psd_estimation.nc"
    if idata_file.exists():
        import arviz as az

        print("Found cached inference data, loading from disk.")
        idata = az.from_netcdf(str(idata_file))
    else:
        coarse_cfg = CoarseGrainConfig(
            enabled=True,
            Nc=200,
        )
        idata = run_mcmc(
            data=multivar_ts,
            n_samples=256,
            n_warmup=256,
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            Nb=200,
            only_vi=True,
            vi_steps=20_000,
            vi_lr=1e-3,
            rng_key=0,
            outdir=str(RESULTS_DIR),
            verbose=True,
            coarse_grain_config=coarse_cfg,
            fmin=1e-2,
        )
        import arviz as az

        az.to_netcdf(idata, str(idata_file))
        print(f"Saved inference data to {idata_file}")

    plot_path = RESULTS_DIR / "multivar_psd_matrix.png"
    plot_psd_matrix(
        idata=idata,
        outdir=str(RESULTS_DIR),
        filename=plot_path.name,
        show_coherence=True,
        diag_yscale="log",
        offdiag_yscale="linear",
        xscale="log",
        label="Multivar log-P-spline",
    )
    return idata, plot_path


def _extract_posterior_psd_and_coherence(idata):
    extracted = extract_plotting_data(idata)
    quantiles = extracted.get("posterior_psd_matrix_quantiles")
    if quantiles is None:
        raise RuntimeError("Inference data missing PSD matrix quantiles.")

    percentiles = np.asarray(quantiles["percentile"])
    if percentiles.shape[0] < 3:
        raise RuntimeError("Need at least three percentiles for CI bands.")

    freq = extracted.get("frequencies")
    if freq is None:
        freq = np.asarray(idata.posterior_psd.coords["freq"].values)
    freq = np.asarray(freq, dtype=float)

    real_psd = np.asarray(quantiles["real"])
    coherence = np.asarray(quantiles.get("coherence"))
    p = real_psd.shape[2]

    psd_quantiles = np.zeros((3, p, real_psd.shape[1]), dtype=float)
    for ch in range(p):
        psd_quantiles[:, ch, :] = real_psd[:, :, ch, ch]

    if coherence is None:
        raise RuntimeError("Inference data missing coherence quantiles.")

    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    coh_quantiles = np.zeros((3, len(pairs), coherence.shape[1]), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        coh_quantiles[:, idx, :] = coherence[:, :, i, j]

    psd_quantiles = np.clip(psd_quantiles, 1e-18, None)
    coh_quantiles = np.clip(coh_quantiles, 0.0, 1.0)
    return freq, psd_quantiles, pairs, coh_quantiles


def _save_vi_psd_csd(idata, channel_names):
    if not hasattr(idata, "vi_posterior_psd"):
        print("No VI posterior PSD group found; skipping PSD/CSD export.")
        return None

    ds = idata.vi_posterior_psd
    if (
        "psd_matrix_real" not in ds.data_vars
        or "psd_matrix_imag" not in ds.data_vars
    ):
        print("VI posterior PSD dataset missing PSD matrix entries.")
        return None

    freq = np.asarray(ds.coords["freq"].values, dtype=float)
    percentiles = np.asarray(ds.coords["percentile"].values, dtype=float)
    psd_real = np.asarray(ds["psd_matrix_real"].values, dtype=float)
    psd_imag = np.asarray(ds["psd_matrix_imag"].values, dtype=float)
    coherence = (
        np.asarray(ds["coherence"].values, dtype=float)
        if "coherence" in ds.data_vars
        else None
    )

    out_path = RESULTS_DIR / "vi_posterior_psd_csd.npz"
    np.savez(
        out_path,
        freq=freq,
        percentiles=percentiles,
        psd_matrix_real=psd_real,
        psd_matrix_imag=psd_imag,
        coherence=coherence,
        channel_names=np.asarray(channel_names),
    )
    print(f"Saved VI posterior PSD/CSD arrays to {out_path}")
    return out_path


def _plot_pairwise_coherence(
    freqs, coh_quantiles, pairs, channel_names, channel_colors, channel_info
):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    q05, q50, q95 = coh_quantiles
    for idx, (i, j) in enumerate(pairs):
        color_i = channel_colors.get(
            channel_names[i],
            CUSTOM_CHANNEL_COLORS.get(channel_names[i], f"C{i % 10}"),
        )
        color_j = channel_colors.get(
            channel_names[j],
            CUSTOM_CHANNEL_COLORS.get(channel_names[j], f"C{j % 10}"),
        )
        ax.fill_between(
            freqs, q05[idx], q95[idx], color=color_i, alpha=0.7, lw=0
        )
        ax.plot(
            freqs,
            q50[idx],
            label=f"{channel_names[i]} vs {channel_names[j]}",
            color=color_j,
            alpha=0.5,
            linewidth=2,
        )

    # custom legend (one entry per channel)
    handles = []
    labels = []
    for ch_name in channel_names:
        color = channel_colors.get(
            ch_name, CUSTOM_CHANNEL_COLORS.get(ch_name, "C0")
        )
        handles.append(plt.Line2D([0], [0], color=color, lw=2))
        labels.append(ch_name)
    # ax.legend(handles, labels, title="Channels", fontsize=11, ncol=2, frameon=False)
    _add_sensor_inset(
        ax,
        channel_info,
        channel_names,
        channel_colors,
        width=2.0,
        height=1.5,
        loc="lower left",
    )

    ax.set_title("Channel-to-channel Coherence", fontsize=18)
    ax.set_xlabel("Frequency [Hz]", fontsize=16)
    ax.set_ylabel("Coherence", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = np.array([1, 2, 5, 10, 20, 40, 80], dtype=float)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}" for x in xticks])
    ax.set_xlim(1, 80)
    ticks = [0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}" for t in ticks])

    ax.set_ylim(top=1.0)
    path = RESULTS_DIR / "selected_channel_coherence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    raw = _load_sample_eeg()
    indices, names = _select_named_channels(raw)
    print("Selected EEG channels:", names)
    (
        time_path,
        selected_data,
        fs,
        spatial_colors,
        channel_info,
    ) = _save_time_domain_butterfly(raw, indices, names)
    print(f"Saved time-domain butterfly to {time_path}")

    idata, plot_path = _run_multivar_pspline(selected_data, fs)
    print(f"Saved multivariate PSD matrix to {plot_path}")
    print(idata)

    freqs, psd_quantiles, pairs, coh_quantiles = (
        _extract_posterior_psd_and_coherence(idata)
    )

    coherence_path = _plot_pairwise_coherence(
        freqs, coh_quantiles, pairs, names, spatial_colors, channel_info
    )
    print(f"Saved pairwise coherence overlay to {coherence_path}")

    _save_vi_psd_csd(idata, names)


if __name__ == "__main__":
    main()
