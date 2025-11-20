import os

import matplotlib.pyplot as plt

from log_psplines.example_datasets.ar_data import ARData
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


def test_lvk_data(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)
    lvk_data = LVKData.download_data(
        detector="L1",
        gps_start=1126259462,
        duration=4,
    )
    lvk_data.plot_psd(fname=f"{outdir}/lvk_psd_analysis.png")


def test_varma_data(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)
    varma_data = VARMAData()
    varma_data.plot(fname=f"{outdir}/varma_data_analysis.png")


def test_lisa_data(outdir):
    outdir = f"{outdir}/{OUT}"
    os.makedirs(outdir, exist_ok=True)
    from log_psplines.example_datasets.lisa_data import LISAData

    lisa_data = LISAData.load()
    lisa_data.plot(f"{outdir}/lisa_spectra_triangle.png")
