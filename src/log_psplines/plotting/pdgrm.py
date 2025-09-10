import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..arviz_utils import get_periodogram, get_spline_model, get_weights
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
    idata=None,
    model_color=MODEL_COL,
    model_label="Model",
    data_color=DATA_COL,
    data_label="Data",
    knot_color=KNOTS_COL,
    show_data=True,
    figsize=(4, 3),
):
    if idata:
        pdgrm = get_periodogram(idata)
        spline_model = get_spline_model(idata)
        weights = get_weights(idata, weights)

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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig = ax.get_figure()

    if plt_data.pdgrm is not None and show_data:
        ax.loglog(
            plt_data.freqs,
            plt_data.pdgrm,
            color=data_color,
            label=data_label,
            zorder=-10,
        )

    if plt_data.model is not None:
        ax.loglog(
            plt_data.freqs,
            plt_data.model,
            label=model_label,
            color=model_color,
        )
        if plt_data.ci is not None:
            ax.fill_between(
                plt_data.freqs,
                plt_data.ci[0],
                plt_data.ci[-1],
                color=model_color,
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
                color=knot_color,
                ms=4.5,
            )

    # ensure that left most and right most freqs are shown
    ax

    ax.set_xlim(plt_data.freqs.min(), plt_data.freqs.max())
    fig.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    plt.tight_layout()
    return fig, ax
