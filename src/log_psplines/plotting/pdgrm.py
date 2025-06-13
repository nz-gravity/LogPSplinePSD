import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines
from .utils import unpack_data

DATA_COL = "lightgray"
MODEL_COL = "tab:orange"
KNOTS_COL = "tab:red"


def plot_pdgrm(
    pdgrm: Periodogram = None,
    spline_model: LogPSplines = None,
    weights=None,
    show_knots=True,
    use_uniform_ci=True,
    use_parametric_model=True,
    freqs=None,
    yscalar=1.0,
    ax=None,
):
    # first, we unpack data
    plt_data = unpack_data(
        pdgrm=pdgrm,
        spline_model=spline_model,
        weights=weights,
        yscalar=yscalar,
        use_uniform_ci=use_uniform_ci,
        use_parametric_model=use_parametric_model,
        freqs=freqs,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = ax.get_figure()

    if plt_data.pdgrm is not None:
        ax.loglog(
            plt_data.freqs,
            plt_data.pdgrm,
            color=DATA_COL,
            label="Data",
            zorder=-10,
        )

    if plt_data.model is not None:
        ax.loglog(
            plt_data.freqs, plt_data.model, label="Model", color=MODEL_COL
        )
        if plt_data.ci is not None:
            ax.fill_between(
                plt_data.freqs,
                plt_data.ci[0],
                plt_data.ci[-1],
                color=MODEL_COL,
                alpha=0.25,
                lw=0,
            )

        if show_knots:
            # get freq of knots (knots are at % of the freqs)
            idx = (spline_model.knots * len(plt_data.freqs)).astype(int)
            # make sure no idx is out of bounds
            idx = jnp.clip(idx, 0, len(plt_data.freqs) - 1)
            ax.loglog(
                plt_data.freqs[idx],
                plt_data.model[idx],
                "o",
                label="Knots",
                color=KNOTS_COL,
                ms=4.5,
            )
    ax.set_xlim(plt_data.freqs.min(), plt_data.freqs.max())
    fig.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    plt.tight_layout()
    return fig, ax
