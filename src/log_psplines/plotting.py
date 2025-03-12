import matplotlib.pyplot as plt
import jax.numpy as jnp
from log_psplines.datasets import Periodogram
from log_psplines.psplines import LogPSplines


def plot_pdgrm(pdgrm: Periodogram, spline_model: LogPSplines = None, weights=None, show_knots=True):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(
        pdgrm.freqs,
        pdgrm.power,
        color="lightgray",
        label="Data",
    )
    if spline_model is not None:
        ln_spline, qtls = None, None
        if weights is None:
            ln_spline = spline_model()
        elif weights.ndim == 1:
            ln_spline = spline_model(weights)
        elif weights.ndim == 2:
            ln_splines = jnp.array([spline_model(w) for w in weights])
            qtls = jnp.percentile(ln_splines, q=jnp.array([16, 50, 84]), axis=0)
            ln_spline = qtls[1]

        spline = jnp.exp(ln_spline)
        ax.loglog(pdgrm.freqs, spline, label="Spline", color="tab:orange")
        if qtls is not None:
            ax.fill_between(pdgrm.freqs, jnp.exp(qtls[0]), jnp.exp(qtls[2]), color="tab:orange", alpha=0.25)

        if show_knots:
            # get freq of knots (knots are at % of the freqs)
            idx = (spline_model.knots * len(pdgrm.freqs)).astype(int)
            ax.loglog(
                pdgrm.freqs[idx],
                spline[idx],
                "o",
                label="Knots",
                color="tab:orange",
                ms=4.5,
            )
    ax.legend(frameon=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    plt.tight_layout()
    return fig, ax
