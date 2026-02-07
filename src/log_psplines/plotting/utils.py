import dataclasses
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines
from .base import compute_confidence_intervals, subsample_weights

__all__ = ["unpack_data"]


@dataclasses.dataclass
class PlottingData:
    freqs: Optional[np.ndarray] = None
    pdgrm: Optional[np.ndarray] = None
    model: Optional[np.ndarray] = None
    ci: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        if self.freqs is not None:
            return len(self.freqs)
        elif self.pdgrm is not None:
            return len(self.pdgrm)
        elif self.model is not None:
            return len(self.model)
        else:
            raise ValueError("No data to get length from.")


def unpack_data(
    pdgrm: Periodogram | None = None,
    spline_model: LogPSplines | None = None,
    weights: Optional[np.ndarray] = None,
    yscalar: float = 1.0,
    use_uniform_ci: bool = True,
    use_parametric_model: bool = True,
    freqs: Optional[np.ndarray] = None,
    posterior_psd_quantiles: Optional[dict[str, Any]] = None,
    model_ci: Optional[np.ndarray] = None,
) -> PlottingData:
    plt_dat = PlottingData()
    if pdgrm is not None:
        plt_dat.pdgrm = np.array(pdgrm.power, dtype=np.float64) * yscalar
        plt_dat.freqs = np.array(pdgrm.freqs)

    if plt_dat.freqs is None and freqs is None:
        plt_dat.freqs = np.linspace(0, 1, plt_dat.n)
    elif freqs is not None:
        plt_dat.freqs = np.asarray(freqs)

    if model_ci is not None:
        plt_dat.ci = np.asarray(model_ci)
        plt_dat.model = np.asarray(model_ci[1])

    if posterior_psd_quantiles is not None:
        percentiles = np.asarray(posterior_psd_quantiles["percentile"])
        values = np.asarray(posterior_psd_quantiles["values"])

        def _grab(target: float) -> np.ndarray:
            idx = int(np.argmin(np.abs(percentiles - target)))
            return values[idx]

        q05 = _grab(5.0)
        q50 = _grab(50.0)
        q95 = _grab(95.0)
        ci = np.stack([q05, q50, q95], axis=0)
        plt_dat.ci = ci
        plt_dat.model = q50

    if plt_dat.model is None and spline_model is not None:

        if weights is None:
            # just use the initial weights/0 weights
            ln_spline = np.asarray(
                spline_model(use_parametric_model=use_parametric_model)
            )

        elif np.asarray(weights).ndim == 1:
            # only one set of weights -- no CI possible
            ln_spline = np.asarray(
                spline_model(jnp.asarray(weights), use_parametric_model)
            )

        else:  # weights.ndim == 2
            # multiple sets of weights -- CI possible

            weights_arr = np.asarray(weights)
            if weights_arr.shape[0] > 500:
                # subsample to speed up
                idx = np.random.choice(
                    weights_arr.shape[0], size=500, replace=False
                )
                weights_arr = weights_arr[idx]

            ln_splines = jnp.array(
                [
                    spline_model(jnp.asarray(w), use_parametric_model)
                    for w in weights_arr
                ]
            )
            ln_splines_np = np.asarray(ln_splines)

            if use_uniform_ci:
                ln_ci = compute_confidence_intervals(
                    ln_splines_np, method="uniform"
                )
            else:  # percentile
                ln_ci = compute_confidence_intervals(
                    ln_splines_np, method="percentile"
                )
            ln_ci_arr = jnp.asarray(ln_ci)
            plt_dat.ci = np.exp(ln_ci_arr, dtype=np.float64) * yscalar
            ln_spline = np.asarray(ln_ci[1])
        plt_dat.model = np.exp(ln_spline, dtype=np.float64) * yscalar

    return plt_dat
