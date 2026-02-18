import os

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.lisa_data import LISAData
from log_psplines.example_datasets.lvk_data import LVKData
from log_psplines.example_datasets.varma_data import VARMAData

OUT = "out_example_datasets"


def test_ar(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ar_data = ARData(
            order=i + 1, duration=8.0, fs=1024.0, sigma=1e-21, seed=42
        )
        ax = ar_data.plot(ax=ax)
        ax.set_title(f"AR({i + 1}) Process")
        ax.grid(True)
        # turn off axes spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/ar_processes.png", bbox_inches="tight", dpi=300)


def test_lisa_data(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)

    lisa_data = LISAData.load()
    lisa_data.plot(f"{outdir}/lisa_spectra_trri.png")

    # assert that the empirical PSD is around the [10**-16] range at 10**-4 Hz and 10**-10 range at 5*10**-2 Hz
    freq_hz = np.asarray(lisa_data.freq)
    psd = lisa_data.true_matrix
    idx_1e_4 = np.argmin(np.abs(freq_hz - 1e-4))
    idx_5e_2 = np.argmin(np.abs(freq_hz - 5e-2))
    psd_at_1e_4 = psd[idx_1e_4, 0, 0]
    psd_at_5e_2 = psd[idx_5e_2, 0, 0]
    print(f"PSD at 1e-4 Hz: {psd_at_1e_4}")
    print(f"PSD at 5e-2 Hz: {psd_at_5e_2}")
    assert np.isclose(psd_at_1e_4, 1e-16, rtol=1e-10)
    assert np.isclose(psd_at_5e_2, 1e-10, rtol=1e-10)

    # RIAE between true and empirical PSD should be small
    # plot errors between true and empirical PSD (for all channels) and |CSD|
    empirical_psd = lisa_data.matrix
    riae = np.abs(empirical_psd - psd) / np.abs(psd)
    plt.figure(figsize=(8, 6))
    for i in range(3):
        for j in range(3):
            plt.loglog(
                freq_hz,
                riae[:, i, j],
                label=f"RIAE PSD TDI {['X', 'Y', 'Z'][i]}-{['X', 'Y', 'Z'][j]}",
            )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative Integrated Absolute Error")
    plt.title("RIAE between True and Empirical PSDs for LISA TDI Channels")
    plt.legend()
    plt.savefig(
        f"{outdir}/lisa_riae_psd_trri.png", bbox_inches="tight", dpi=300
    )


def test_lvk_data(outdir, monkeypatch):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)

    def _fake_fetch_open_data(detector, gps_start, gps_end):
        from gwpy.timeseries import TimeSeries

        sample_rate = 256
        n_samples = int((gps_end - gps_start) * sample_rate)
        rng = np.random.default_rng(12345)
        data = rng.normal(scale=1e-21, size=n_samples)
        return TimeSeries(data, sample_rate=sample_rate, t0=gps_start)

    monkeypatch.setattr(
        "log_psplines.example_datasets.lvk_data.TimeSeries.fetch_open_data",
        _fake_fetch_open_data,
    )

    lvk_data = LVKData.download_data(
        detector="L1",
        gps_start=1126259462,
        duration=1,
    )
    lvk_data.plot_psd(fname=f"{outdir}/lvk_psd_analysis.png")


def test_varma_data(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)
    varma_data = VARMAData(n_samples=512, fs=64.0, seed=0)
    varma_data.plot(fname=f"{outdir}/varma_data_analysis.png")
    freq_hz = np.asarray(varma_data.freq)

    # Frequency axis should live in Hz up to the Nyquist frequency
    assert np.all(freq_hz > 0)
    assert np.isclose(freq_hz.max(), varma_data.fs / 2.0)
    np.testing.assert_allclose(np.diff(freq_hz), freq_hz[0])

    # True PSD should integrate to the channel variances (within sampling noise)
    psd = varma_data.get_true_psd()
    psd_vars = np.array(
        [np.trapezoid(psd[:, i, i].real, freq_hz) for i in range(varma_data.p)]
    )
    empirical_vars = np.var(varma_data.data, axis=0)
    np.testing.assert_allclose(psd_vars, empirical_vars, rtol=0.1)


def test_varma_validity_flags():
    stable = VARMAData(
        n_samples=128,
        var_coeffs=np.array([[[0.35, 0.0], [0.0, 0.25]]], dtype=float),
        vma_coeffs=np.eye(2)[None, ...],
        sigma=np.eye(2),
        seed=1,
    )
    assert stable.is_var_stationary is True
    assert stable.is_valid_var_dataset is True
    assert stable.var_companion_spectral_radius is not None
    assert stable.var_companion_spectral_radius < 1.0

    non_stationary = VARMAData(
        n_samples=128,
        var_coeffs=np.array([[[1.05, 0.0], [0.0, 0.95]]], dtype=float),
        vma_coeffs=np.eye(2)[None, ...],
        sigma=np.eye(2),
        seed=2,
    )
    assert non_stationary.is_var_stationary is False
    assert non_stationary.is_valid_var_dataset is False
    assert non_stationary.var_companion_spectral_radius is not None
    assert non_stationary.var_companion_spectral_radius >= 1.0
