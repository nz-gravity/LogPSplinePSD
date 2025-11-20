import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import mne
import numpy as np
from cycler import cycler

from log_psplines.coarse_grain.config import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
            "EEG 001",
            "EEG 021",  # frontal/right
            "EEG 041",  # vertex
            "EEG 050",  # parietal
            "EEG 060",  # occipital
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


def _save_time_domain_butterfly(
    raw, selected_indices, channel_names, channel_colors=None
):
    info = mne.pick_info(raw.info, sel=selected_indices)
    selected_data = raw.get_data(picks=selected_indices)
    n_times = selected_data.shape[1]
    fs = int(raw.info["sfreq"])
    n_times_plot = min(fs, n_times)

    time_evoked = mne.EvokedArray(
        selected_data[:, :n_times_plot],
        info,
        tmin=0.0,
        comment="Time-domain snippet",
    )

    if channel_colors is None:
        channel_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = None
    with plt.rc_context(
        {"axes.prop_cycle": cycler(color=channel_colors[: len(channel_names)])}
    ):
        fig = time_evoked.plot(
            picks="eeg",
            spatial_colors=False,
            gfp=True,
            window_title="Time-domain Evoked-style (selected channels)",
            time_unit="s",
            show=False,
        )
    fig.tight_layout()
    path = RESULTS_DIR / "time_domain_evoked_butterfly.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, selected_data, fs


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
            n_log_bins=200,
            f_transition=1e-3,
            f_min=1e-2,
        )
        idata = run_mcmc(
            data=multivar_ts,
            sampler="multivar_blocked_nuts",
            n_samples=256,
            n_warmup=256,
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            n_time_blocks=200,
            only_vi=True,
            vi_steps=20_000,
            vi_lr=1e-3,
            rng_key=0,
            outdir=str(RESULTS_DIR),
            verbose=True,
            coarse_grain_config=coarse_cfg,
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
    n_channels = real_psd.shape[2]

    psd_quantiles = np.zeros((3, n_channels, real_psd.shape[1]), dtype=float)
    for ch in range(n_channels):
        psd_quantiles[:, ch, :] = real_psd[:, :, ch, ch]

    if coherence is None:
        raise RuntimeError("Inference data missing coherence quantiles.")

    pairs = [
        (i, j) for i in range(n_channels) for j in range(i + 1, n_channels)
    ]
    coh_quantiles = np.zeros((3, len(pairs), coherence.shape[1]), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        coh_quantiles[:, idx, :] = coherence[:, :, i, j]

    psd_quantiles = np.clip(psd_quantiles, 1e-18, None)
    coh_quantiles = np.clip(coh_quantiles, 1e-6, 1.0)
    return freq, psd_quantiles, pairs, coh_quantiles


def _plot_channel_psds(freqs, psd_quantiles, channel_names, colors):
    colors = colors[: len(channel_names)]
    q05, q50, q95 = psd_quantiles
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(channel_names):
        color = colors[idx]
        ax.fill_between(freqs, q05[idx], q95[idx], color=color, alpha=0.2)
        ax.loglog(freqs, q50[idx], label=name, color=color)
    ax.set_title("Selected EEG Channel PSDs")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    path = RESULTS_DIR / "selected_channel_psds.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_pairwise_coherence(freqs, coh_quantiles, pairs, channel_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    q05, q50, q95 = coh_quantiles
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (i, j) in enumerate(pairs):
        color = color_cycle[idx % len(color_cycle)]
        ax.fill_between(freqs, q05[idx], q95[idx], color=color, alpha=0.2)
        ax.semilogy(
            freqs,
            q50[idx],
            label=f"{channel_names[i]} vs {channel_names[j]}",
            color=color,
        )
    ax.set_title("Channel-to-channel Coherence")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence")
    ax.grid(ls=":")
    ax.legend()
    ax.set_xscale("log")
    path = RESULTS_DIR / "selected_channel_coherence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    raw = _load_sample_eeg()
    indices, names = _select_named_channels(raw)
    print("Selected EEG channels:", names)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    channel_colors = base_colors[: len(names)]

    time_path, selected_data, fs = _save_time_domain_butterfly(
        raw, indices, names, channel_colors=channel_colors
    )
    print(f"Saved time-domain butterfly to {time_path}")

    idata, plot_path = _run_multivar_pspline(selected_data, fs)
    print(f"Saved multivariate PSD matrix to {plot_path}")

    freqs, psd_quantiles, pairs, coh_quantiles = (
        _extract_posterior_psd_and_coherence(idata)
    )
    psd_path = _plot_channel_psds(freqs, psd_quantiles, names, channel_colors)
    print(f"Saved channel PSD overlay to {psd_path}")

    coherence_path = _plot_pairwise_coherence(
        freqs, coh_quantiles, pairs, names
    )
    print(f"Saved pairwise coherence overlay to {coherence_path}")


if __name__ == "__main__":
    main()
