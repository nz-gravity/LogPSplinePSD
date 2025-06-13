import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm.auto import tqdm

from log_psplines.datasets import Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm

outdir = "plots"
os.makedirs(outdir, exist_ok=True)
data_file = f"{outdir}/mcmc_runtimes.npy"


def run_analysis():

    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 4096
    fs = 100  # Sampling frequency in Hz.
    dt = 1.0 / fs
    t = jnp.array(np.linspace(0, (n_samples - 1) * dt, n_samples))
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    noise = jnp.array((noise - np.mean(noise)) / np.std(noise))
    mock_pdgrm = Timeseries(t, noise).to_periodogram().highpass(5)

    kwgs = dict(pdgrm=mock_pdgrm, num_samples=50, num_warmup=50, verbose=False)

    runtimes = []
    ks = np.linspace(8, 100, num=20, dtype=int)
    reps = 5
    for k in tqdm(ks):
        print(f"Running all reps for k: {k}\n")
        spline_model, samples = None, None
        for rep in range(reps):
            t0 = time.time()
            mcmc, spline_model = run_mcmc(n_knots=k, **kwgs)
            runtime = float(time.time()) - t0
            runtimes.append(runtime)

        samples = mcmc.get_samples()
        fig, ax = plot_pdgrm(mock_pdgrm, spline_model, samples["weights"])
        fig.savefig(os.path.join(outdir, f"test_mcmc_{k}.png"))
        plt.close(fig)

    # save  [k , runtime]
    median_runtimes = np.median(
        np.array(runtimes).reshape(len(ks), reps), axis=1
    )
    std_runtimes = np.std(np.array(runtimes).reshape(len(ks), reps), axis=1)
    np.save(
        data_file,
        np.array([ks, median_runtimes, std_runtimes]),
    )


def plot():
    data = np.load(data_file)
    ks, median_runtimes, std_runtimes = data[0], data[1], data[2]
    plt.figure()
    plt.errorbar(ks, median_runtimes, yerr=std_runtimes, fmt="o", color="k")
    plt.xlabel("Number of knots")
    plt.ylabel("Runtime (s)")
    plt.xlim(ks[0] - 2, ks[-1] + 2)
    plt.savefig(os.path.join(outdir, "mcmc_runtimes.png"))


if __name__ == "__main__":
    if not os.path.exists(data_file):
        run_analysis()
    plot()
    print(f"Plots saved to {outdir}")
