import glob
import os
from typing import List
import json

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.mcmc import run_mcmc
from .plotting import plot_data_size_results, plot_knots_results

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
    n_samples=1000,
    n_warmup=1000,
    verbose=False,
    n_knots=10,
    knot_kwargs=dict(frac_uniform=1.0),

)


class RuntimeBenchmark:
    """Benchmark runtime performance of MCMC sampling for different configurations."""

    def __init__(self, outdir: str = "plots", verbose: bool = False, n_mcmc: int = 2000, ):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.verbose = verbose
        self.n_mcmc = n_mcmc // 2
        self.n_warmup = n_mcmc // 2


    def _run_data_size_analysis(
            self,
            sampler: str = "nuts",
            min_n: float = 128.0,
            max_n: float = 1024.0,
            num_points: int = 10,
            reps: int = 3,
    ) -> None:
        """Analyze runtime vs data size (duration)."""

        mcmc_kwgs = MCMC_DEFAULTS.copy()
        mcmc_kwgs["sampler"] = sampler
        mcmc_kwgs["verbose"] = self.verbose
        mcmc_kwgs["n_samples"] = self.n_mcmc
        mcmc_kwgs["n_warmup"] = self.n_warmup

        assert min_n < max_n, f"min_n {min_n} must be less than max_n {max_n}"
        _ns = np.geomspace(min_n, max_n, num=num_points, dtype=int)
        durations = _ns / AR_DEFAULTS["fs"]  # Convert to durations in seconds
        runtimes = []
        ns = []
        ess = []

        for duration in tqdm(durations, desc=f"Varying data sizes [{sampler}, {DEVICE}]"):
            # Generate AR data with specified duration
            pdgrm_kwgs = AR_DEFAULTS.copy()
            pdgrm_kwgs["duration"] = duration
            ar_data = ARData(**pdgrm_kwgs)
            pdgrm = ar_data.periodogram

            runtimes_i = []
            ess_i = []
            for rep in range(reps):
                idata = run_mcmc(
                    pdgrm=pdgrm,
                    rng_key=rep,
                    **mcmc_kwgs,
                )
                runtimes_i.append(idata.attrs["runtime"])
                ess_i.append(idata.attrs["ess"])

            runtimes.append(runtimes_i)
            ns.append(len(pdgrm.freqs))
            ess.append(np.concatenate(ess_i))

        # Save results
        data_file = f"{self.outdir}/data_size_runtimes_{sampler}_{DEVICE}.json"
        runtimes_array = np.array(runtimes)
        ess = np.array(ess)

        with open(data_file, "w") as f:
            json.dump(
                {
                    "ns": ns,
                    "durations": durations.tolist(),
                    "runtimes": runtimes_array.tolist(),
                    "ess": ess.tolist(),
                    "sampler": sampler,
                    "device": DEVICE,
                },
                f,
                indent=2,
            )

    def _run_knots_analysis(
            self,
            sampler: str = "nuts",
            min_knots: int = 5,
            max_knots: int = 30,
            num_points: int = 10,
            reps: int = 3,
    ) -> None:
        """Analyze runtime vs number of knots."""
        # Fixed AR data
        ar_data = ARData(**AR_DEFAULTS)
        pdgrm = ar_data.periodogram

        # MCMC parameters
        mcmc_kwgs = MCMC_DEFAULTS.copy()
        mcmc_kwgs["verbose"] = self.verbose
        mcmc_kwgs["sampler"] = sampler
        mcmc_kwgs["pdgrm"] = pdgrm
        mcmc_kwgs["n_samples"] = self.n_mcmc
        mcmc_kwgs["n_warmup"] = self.n_warmup

        ks = np.linspace(min_knots, max_knots, num=num_points, dtype=int)
        runtimes = []
        ess = []

        for k in tqdm(ks, desc=f"Varying k-knots [{sampler}, {DEVICE}]"):
            runtimes_i = []
            ess_i = []
            for rep in range(reps):
                mcmc_kwgs_ = mcmc_kwgs.copy()
                mcmc_kwgs_["n_knots"] = int(k)
                idata = run_mcmc(rng_key=rep, **mcmc_kwgs)
                runtimes_i.append(idata.attrs["runtime"])
                ess_i.append(idata.attrs["ess"])
            ess.append(np.concatenate(ess_i))
            runtimes.append(runtimes_i)

        # Save results
        data_file = f"{self.outdir}/knots_runtimes_{sampler}_{DEVICE}.json"
        runtimes_array = np.array(runtimes)
        ess = np.array(ess)

        with open(data_file, "w") as f:
            json.dump(
                {
                    "ks": ks.tolist(),
                    "ess": ess.tolist(),
                    "runtimes": runtimes_array.tolist(),
                    "sampler": sampler,
                    "device": DEVICE,
                },
                f,
                indent=2,
            )

    def run_analysis(
            self,
            n_points: int = 10,
            n_reps: int = 3,
            min_n: float = 128.0,
            max_n: float = 1024.0,
            min_knots: int = 5,
            max_knots: int = 30,
    ):
        """run analyses for both MH and NUTS samplers."""
        for sampler in ["nuts", "mh"]:
            self._run_data_size_analysis(
                sampler=sampler,
                min_n=min_n,
                max_n=max_n,
                num_points=n_points,
                reps=n_reps,
            )
            self._run_knots_analysis(
                sampler=sampler,
                min_knots=min_knots,
                max_knots=max_knots,
                num_points=n_points,
                reps=n_reps,
            )
        self.plot()

    def plot(self) -> None:
        """Plot all analyses."""

        runtimes_n = glob.glob(f"{self.outdir}/data_size_runtimes_*.json")
        runtimes_knots = glob.glob(f"{self.outdir}/knots_runtimes_*.json")

        if runtimes_n:
            plot_data_size_results(runtimes_n)

        if runtimes_knots:
            plot_knots_results(runtimes_knots)
