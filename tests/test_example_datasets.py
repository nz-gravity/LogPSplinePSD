import matplotlib.pyplot as plt

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.lvk_data import LVKData



def test_ar(outdir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ar_data = ARData(
            order=i + 1, duration=8.0, fs=1024.0, sigma=1.0, seed=42
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
    processor = LVKData.download_data(
        detector="L1",
        gps_start=1126259462,
        duration=4,
    )
    processor.print_knot_info()
    processor.plot_psd_analysis(include_lines=True, fname=f"{outdir}/lvk_psd_analysis.png")
