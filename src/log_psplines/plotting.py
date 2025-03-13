import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.datasets import Periodogram
from log_psplines.psplines import LogPSplines

DATA_COL = "lightgray"
MODEL_COL = "tab:orange"
KNOTS_COL = "tab:red"


def plot_pdgrm(
    pdgrm: Periodogram,
    spline_model: LogPSplines = None,
    weights=None,
    show_knots=True,
    ax=None,
    use_uniform_ci=True,
    yscalar=1.0,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = ax.get_figure()

    scaled_pdgrm = np.array(pdgrm.power, dtype=np.float64) * yscalar
    ax.loglog(
        pdgrm.freqs, scaled_pdgrm, color=DATA_COL, label="Data", zorder=-10
    )
    if spline_model is not None:
        ln_parm = spline_model.log_parametric_model
        ln_spline, ci = None, None
        if weights is None:
            ln_spline = spline_model()
        elif weights.ndim == 1:
            ln_spline = spline_model(weights)
        elif weights.ndim == 2:
            ln_splines = jnp.array([spline_model(w) for w in weights])
            if use_uniform_ci:
                ci = compute_uniform_ci(
                    ln_splines,
                )
            else:  # percentile
                ci = jnp.percentile(
                    ln_splines, q=jnp.array([16, 50, 84]), axis=0
                )
            ln_spline = ci[1]

        spline = np.exp(ln_spline + ln_parm, dtype=np.float64) * yscalar
        ax.loglog(pdgrm.freqs, spline, label="Spline", color=MODEL_COL)
        if ci is not None:
            ci_bot = np.exp(ci[0] + ln_parm, dtype=np.float64) * yscalar
            ci_top = np.exp(ci[2] + ln_parm, dtype=np.float64) * yscalar
            ax.fill_between(
                pdgrm.freqs, ci_bot, ci_top, color=MODEL_COL, alpha=0.25, lw=0
            )

        if show_knots:
            # get freq of knots (knots are at % of the freqs)
            idx = (spline_model.knots * len(pdgrm.freqs)).astype(int)
            # make sure no idx is out of bounds
            idx = jnp.clip(idx, 0, len(pdgrm.freqs) - 1)
            ax.loglog(
                pdgrm.freqs[idx],
                spline[idx],
                "o",
                label="Knots",
                color=KNOTS_COL,
                ms=4.5,
            )
    ax.set_xlim(pdgrm.freqs.min(), pdgrm.freqs.max())
    ax.set_ylim(scaled_pdgrm.min(), scaled_pdgrm.max())
    fig.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    plt.tight_layout()
    return fig, ax


def compute_uniform_ci(samples, alpha=0.1):
    """
    Compute a uniform (simultaneous) confidence band for a set of function samples.

    Args:
        samples (jnp.ndarray): Shape (num_samples, num_points) array of function samples.
        alpha (float): Significance level (default 0.1 for 90% CI).

    Returns:
        tuple: (lower_bound, median, upper_bound) arrays.
    """
    num_samples, num_points = samples.shape

    # Compute pointwise median and standard deviation
    median = jnp.median(samples, axis=0)
    std = jnp.std(samples, axis=0)

    # Compute the max deviation over all samples
    deviations = (samples - median[None, :]) / std[
        None, :
    ]  # Normalize deviations
    max_deviation = jnp.max(
        jnp.abs(deviations), axis=1
    )  # Max deviation per sample

    # Compute the scaling factor using the distribution of max deviations
    k_alpha = jnp.percentile(
        max_deviation, 100 * (1 - alpha)
    )  # Critical value

    # Compute uniform confidence bands
    lower_bound = median - k_alpha * std
    upper_bound = median + k_alpha * std

    return lower_bound, median, upper_bound
