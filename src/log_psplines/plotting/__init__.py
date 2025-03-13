from typing import List, Tuple

import arviz as az
import matplotlib.pyplot as plt
from numpyro.infer import MCMC

from .pdgrm import plot_pdgrm

__all__ = ["plot_pdgrm", "plot_trace"]


def plot_trace(mcmc: MCMC, fname=None):
    inf_obj = az.from_numpyro(mcmc)
    ax = az.plot_trace(inf_obj)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close(ax.flatten()[0].figure)
