import glob
import os
from typing import List

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.mcmc import run_mcmc

# Get device information
DEVICE = jax.devices()[0].platform
print(f"Running on: {DEVICE}")

AR_DEFAULTS = dict(
    order=4,
    duration=2.0,
    fs=128.0,
    sigma=1.0,
    seed=42,
)

MCMC_DEFAULTS = dict(
    n_samples=50,
    n_warmup=50,
    verbose=False,
    n_knots=10,
    knot_kwargs=dict(frac_uniform=1.0),
)


class RuntimeBenchmark:
    """Benchmark runtime performance of MCMC sampling for different configurations."""

    def __init__(self, outdir: str = "plots"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def run_data_size_analysis(
        self,
        sampler: str = "nuts",
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        num_points: int = 10,
        reps: int = 3,
    ) -> None:
        """Analyze runtime vs data size (duration)."""

        print(f"Running data size analysis with {sampler} sampler")
        mcmc_kwgs = MCMC_DEFAULTS.copy()
        mcmc_kwgs["sampler"] = sampler

        durations = np.linspace(min_duration, max_duration, num=num_points)
        runtimes = []
        ns = []

        for duration in tqdm(durations, desc="Data sizes"):
            # Generate AR data with specified duration
            pdgrm_kwgs = AR_DEFAULTS.copy()
            pdgrm_kwgs["duration"] = duration
            ar_data = ARData(**pdgrm_kwgs)
            pdgrm = ar_data.periodogram

            duration_runtimes = []
            for rep in range(reps):
                idata = run_mcmc(
                    pdgrm=pdgrm,
                    rng_key=rep,
                    **mcmc_kwgs,
                )
                duration_runtimes.append(idata.attrs["runtime"])

            runtimes.append(duration_runtimes)
            ns.append(len(pdgrm.freqs))

        # Save results
        data_file = f"{self.outdir}/data_size_runtimes_{sampler}_{DEVICE}.npy"
        runtimes_array = np.array(runtimes)
        median_runtimes = np.median(runtimes_array, axis=1)
        std_runtimes = np.std(runtimes_array, axis=1)

        np.save(
            data_file,
            {
                "ns": ns,
                "durations": durations,
                "median_runtimes": median_runtimes,
                "std_runtimes": std_runtimes,
                "all_runtimes": runtimes_array,
                "sampler": sampler,
                "device": DEVICE,
            },
        )

        print(f"Data size analysis saved to {data_file}")

    def run_knots_analysis(
        self,
        sampler: str = "nuts",
        min_knots: int = 5,
        max_knots: int = 30,
        num_points: int = 10,
        reps: int = 3,
    ) -> None:
        """Analyze runtime vs number of knots."""

        print(f"Running knots analysis with {sampler} sampler")

        # Fixed AR data
        ar_data = ARData(**AR_DEFAULTS)
        pdgrm = ar_data.periodogram

        # MCMC parameters
        mcmc_kwgs = MCMC_DEFAULTS.copy()
        mcmc_kwgs["sampler"] = sampler

        ks = np.linspace(min_knots, max_knots, num=num_points, dtype=int)
        runtimes = []

        for k in tqdm(ks, desc="Number of knots"):
            k_runtimes = []
            for rep in range(reps):
                idata = run_mcmc(n_knots=int(k), rng_key=rep, **mcmc_kwgs)
                k_runtimes.append(idata.attrs["runtime"])

            runtimes.append(k_runtimes)

        # Save results
        data_file = f"{self.outdir}/knots_runtimes_{sampler}_{DEVICE}.npy"
        runtimes_array = np.array(runtimes)
        median_runtimes = np.median(runtimes_array, axis=1)
        std_runtimes = np.std(runtimes_array, axis=1)

        np.save(
            data_file,
            {
                "ks": ks,
                "median_runtimes": median_runtimes,
                "std_runtimes": std_runtimes,
                "all_runtimes": runtimes_array,
                "sampler": sampler,
                "device": DEVICE,
            },
        )

        print(f"Knots analysis saved to {data_file}")

    def plot(self) -> None:
        """Plot all analyses."""

        runtimes_n = glob.glob(f"{self.outdir}/data_size_runtimes_*.npy")
        runtimes_knots = glob.glob(f"{self.outdir}/knots_runtimes_*.npy")

        if runtimes_n:
            print(f"Plotting data size results {len(runtimes_n)} files found")
            plot_data_size_results(runtimes_n)

        if runtimes_knots:
            print(f"Plotting knots results {len(runtimes_knots)} files found")
            # plot_knots_results(runtimes_knots)  # Uncomment when implemented


def plot_data_size_results(filepaths: List[str]) -> None:
    """Plot data size analysis results."""

    fig, ax = plt.subplots(1, 1)

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Data file {filepath} not found")
            continue

        data = np.load(filepath, allow_pickle=True).item()

        ax.errorbar(
            data["ns"],
            data["median_runtimes"],
            yerr=data["std_runtimes"],
            fmt="o-",
            capsize=3,
            label=f"{data['sampler'].upper()} on {data['device']}",
        )

    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Runtime (seconds)")

    plt.tight_layout()

    fdir = os.path.dirname(filepaths[0])

    plt.savefig(f"{fdir}/N_vs_runtime.png", dpi=150)
    plt.close()


#
# def plot_knots_results(self, sampler: str = "nuts") -> None:
#     """Plot knots analysis results."""
#
#     data_file = f"{self.outdir}/knots_runtimes_{sampler}_{DEVICE}.npy"
#     if not os.path.exists(data_file):
#         print(f"Data file {data_file} not found")
#         return
#
#     data = np.load(data_file, allow_pickle=True).item()
#
#     plt.figure(figsize=(8, 6))
#     plt.errorbar(
#         data['ks'],
#         data['median_runtimes'],
#         yerr=data['std_runtimes'],
#         fmt='o-',
#         capsize=3,
#         label=f'{sampler.upper()} on {DEVICE}'
#     )
#     plt.xlabel('Number of knots')
#     plt.ylabel('Runtime (seconds)')
#     plt.title('Runtime vs Number of Knots')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{self.outdir}/knots_runtime_{sampler}_{DEVICE}.png", dpi=150)
#     plt.close()
