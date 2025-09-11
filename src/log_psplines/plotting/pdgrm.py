import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from ..datatypes import Periodogram
from ..psplines import LogPSplines
from .utils import unpack_data

DATA_COL = mcolors.to_hex("lightgray")
MODEL_COL = mcolors.to_hex("tab:orange")
KNOTS_COL = mcolors.to_hex("tab:red")


def plot_pdgrm(
    pdgrm: Periodogram = None,
    spline_model: LogPSplines = None,
    weights=None,
    show_knots=True,
    use_uniform_ci=True,
    use_parametric_model=True,
    show_parametric=False,
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
    interactive=False,
):

    if idata:
        from ..arviz_utils import (
            get_periodogram,
            get_spline_model,
            get_weights,
        )

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

    if interactive:
        return _plotly_backend(
            plt_data=plt_data,
            spline_model=spline_model,
            show_knots=show_knots,
            show_parametric=show_parametric,
            model_color=model_color,
            model_label=model_label,
            data_color=data_color,
            data_label=data_label,
            knot_color=knot_color,
            show_data=show_data,
            figsize=figsize,
        )
    else:
        return _plt_backend(
            plt_data=plt_data,
            spline_model=spline_model,
            show_knots=show_knots,
            show_parametric=show_parametric,
            model_color=model_color,
            model_label=model_label,
            data_color=data_color,
            data_label=data_label,
            knot_color=knot_color,
            show_data=show_data,
            figsize=figsize,
            ax=ax,
        )


def _plt_backend(
    plt_data,
    spline_model,
    show_knots,
    show_parametric,
    model_color,
    model_label,
    data_color,
    data_label,
    knot_color,
    show_data,
    figsize,
    ax=None,
):
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

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
            idx = (spline_model.knots * len(plt_data.freqs)).astype(int)
            idx = jnp.clip(idx, 0, len(plt_data.freqs) - 1)
            ax.loglog(
                plt_data.freqs[idx],
                plt_data.model[idx],
                "o",
                label="Knots",
                color=knot_color,
                ms=4.5,
            )

    if show_parametric:
        ax.loglog(
            plt_data.freqs,
            spline_model.parametric_model,
            label="Parametric",
            color=model_color,
            ls="--",
        )

    ax.set_xlim(plt_data.freqs.min(), plt_data.freqs.max())
    fig.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    plt.tight_layout()
    return fig, ax


def _plotly_backend(
    plt_data,
    spline_model,
    show_knots,
    show_parametric,
    model_color,
    model_label,
    data_color,
    data_label,
    knot_color,
    show_data,
    figsize,
):
    import jax.numpy as jnp
    import plotly.graph_objects as go

    fig = go.Figure()
    figsize = (10, 7.5) if figsize == (4, 3) else figsize

    # format colors to ploty-styled hex
    data_color = mcolors.to_hex(data_color)[0:7]
    model_color = mcolors.to_hex(model_color)[0:7]
    knot_color = mcolors.to_hex(knot_color)[0:7]

    if plt_data.pdgrm is not None and show_data:
        fig.add_trace(
            go.Scatter(
                x=plt_data.freqs,
                y=plt_data.pdgrm,
                mode="lines",
                name=data_label,
                line=dict(color=data_color),
            )
        )

    if plt_data.model is not None:
        fig.add_trace(
            go.Scatter(
                x=plt_data.freqs,
                y=plt_data.model,
                mode="lines",
                name=model_label,
                line=dict(color=model_color),
            )
        )
        if plt_data.ci is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(plt_data.freqs) + list(plt_data.freqs[::-1]),
                    y=list(plt_data.ci[0]) + list(plt_data.ci[-1][::-1]),
                    fill="toself",
                    fillcolor=model_color,  # semi-transparent
                    opacity=0.25,
                    line=dict(width=0),
                    name=f"{model_label} CI",
                    showlegend=False,
                )
            )

        if show_knots:
            idx = (spline_model.knots * len(plt_data.freqs)).astype(int)
            idx = jnp.clip(idx, 0, len(plt_data.freqs) - 1)
            fig.add_trace(
                go.Scatter(
                    x=plt_data.freqs[idx],
                    y=plt_data.model[idx],
                    mode="markers",
                    marker=dict(color=knot_color, size=6, symbol="circle"),
                    name="Knots",
                )
            )

    if show_parametric:
        fig.add_trace(
            go.Scatter(
                x=plt_data.freqs,
                y=spline_model.parametric_model,
                mode="lines",
                line=dict(color=model_color, dash="dash"),
                name="Parametric",
            )
        )

    fig.update_xaxes(type="log", title="Frequency [Hz]")
    fig.update_yaxes(type="log", title="PSD [1/Hz]")
    fig.update_layout(
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top"),
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
    )

    return fig
