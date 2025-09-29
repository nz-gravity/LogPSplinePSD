""" Multivariate PSD simulation study with VARMA data

Inputs: N (data size), K (number of knots), SEED (random seed)
Outputs: Estimated PSDs, coverage probabilities, and performance metrics
"""

import os
import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting import plot_psd_matrix


def simulation_study(N=1024, K=7, SEED=42, outdir="results/multivar_psd/simulation_study"):