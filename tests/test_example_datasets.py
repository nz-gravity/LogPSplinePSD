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


# def test_lvk_data(outdir):
#     # Download data and compute PSDs.
#     lvk_data = LVKData.load(
#         detector="H1",
#         gps_start=1126259462,
#         duration=4,
#         segment_duration=1,
#         segment_overlap=0.5,
#         min_freq=10,
#         max_freq=512,
#     )
#     # Access number of segments:
#     print("Number of segments:", lvk_data.n_segments)
#     # Plot individual and median PSDs.
#     fig = lvk_data.plot_psd()
#     plt.savefig(f"{outdir}/lvk_psd.png", bbox_inches="tight")
