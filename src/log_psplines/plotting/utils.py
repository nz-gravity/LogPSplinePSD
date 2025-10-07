import dataclasses

import jax.numpy as jnp
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines
from .base import compute_confidence_intervals, subsample_weights

__all__ = ["unpack_data"]


@dataclasses.dataclass
class PlottingData:
    freqs: np.ndarray = None
    pdgrm: np.ndarray = None
    model: np.ndarray = None
    ci: np.ndarray = None

    @property
    def n(self):
        if self.freqs is not None:
            return len(self.freqs)
        elif self.pdgrm is not None:
            return len(self.pdgrm)
        elif self.model is not None:
            return len(self.model)
        else:
            raise ValueError("No data to get length from.")


def unpack_data(
    pdgrm: Periodogram = None,
    spline_model: LogPSplines = None,
    weights=None,
    yscalar=1.0,
    use_uniform_ci=True,
    use_parametric_model=True,
    freqs=None,
    posterior_psd=None,
    model_ci=None,
):
    plt_dat = PlottingData()
    if pdgrm is not None:
        plt_dat.pdgrm = np.array(pdgrm.power, dtype=np.float64) * yscalar
        plt_dat.freqs = np.array(pdgrm.freqs)

    if plt_dat.freqs is None and freqs is None:
        plt_dat.freqs = np.linspace(0, 1, plt_dat.n)
    elif freqs is not None:
        plt_dat.freqs = freqs

    if model_ci is not None:
        plt_dat.ci = model_ci
        plt_dat.model = model_ci[1]

    if posterior_psd is not None:
        ci = np.percentile(posterior_psd, q=jnp.array([16, 50, 84]), axis=0)
        plt_dat.ci = ci
        plt_dat.model = ci[1]

    if plt_dat.model is None and spline_model is not None:

        if weights is None:
            # just use the initial weights/0 weights
            ln_spline = spline_model(use_parametric_model=use_parametric_model)

        elif weights.ndim == 1:
            # only one set of weights -- no CI possible
            ln_spline = spline_model(weights, use_parametric_model)

        else:  # weights.ndim == 2
            # multiple sets of weights -- CI possible

            if weights.shape[0] > 500:
                # subsample to speed up
                idx = np.random.choice(
                    weights.shape[0], size=500, replace=False
                )
                weights = weights[idx]

            ln_splines = jnp.array(
                [spline_model(w, use_parametric_model) for w in weights]
            )

            if use_uniform_ci:
                ln_ci = compute_confidence_intervals(
                    ln_splines, method="uniform"
                )
            else:  # percentile
                ln_ci = compute_confidence_intervals(
                    ln_splines, method="percentile"
                )
            ln_ci = jnp.array(ln_ci)
            plt_dat.ci = np.exp(ln_ci, dtype=np.float64) * yscalar
            ln_spline = ln_ci[1]
        plt_dat.model = np.exp(ln_spline, dtype=np.float64) * yscalar

    return plt_dat
