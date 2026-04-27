"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

import io

import arviz_plots as azp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import xarray as xr

from ._factors import factor_idatas


def _has_per_channel_stats(idata: xr.DataTree) -> bool:
    """Return True only if sample_stats has explicit per-channel fields."""
    try:
        ss = idata["sample_stats"].dataset
        if ss is None:
            return False
        return any("_channel_" in str(name) for name in ss.data_vars)
    except (KeyError, AttributeError):
        return False


def plot_energy(posteriors: xr.DataTree | dict[str, xr.DataTree]):
    """Energy diagnostic plot.

    For a joint NUTS run (univariate or multivariate joint model): delegates
    directly to azp.plot_energy on the full idata.

    For blocked multivariate NUTS (per-channel sample_stats present): plots
    per-factor energy diagnostics stacked vertically, one panel per channel.
    """
    # If caller already provides a pre-split dict, use it directly.
    if isinstance(posteriors, dict):
        factors = posteriors
        use_per_channel = True
    else:
        use_per_channel = _has_per_channel_stats(posteriors)
        if use_per_channel:
            try:
                factors = factor_idatas(posteriors)
            except Exception:
                use_per_channel = False

    if not use_per_channel:
        # Single joint NUTS trajectory — standard ArviZ plot.
        dt = posteriors
        try:
            return azp.plot_energy(dt, backend="matplotlib")
        except Exception:
            if isinstance(dt, xr.DataTree) and "sample_stats" in dt.children:
                return azp.plot_energy(
                    dt["sample_stats"], backend="matplotlib"
                )
            raise

    # Blocked multivariate: render each factor's plot as an image and stack.
    images = []
    for factor_name in sorted(factors):
        factor_dt = factors[factor_name]
        try:
            pc = azp.plot_energy(factor_dt, backend="matplotlib")
        except Exception:
            if (
                isinstance(factor_dt, xr.DataTree)
                and "sample_stats" in factor_dt.children
            ):
                pc = azp.plot_energy(
                    factor_dt["sample_stats"], backend="matplotlib"
                )
            else:
                continue
        fig = pc.viz["figure"].item()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        images.append((factor_name, mpimg.imread(buf)))
        plt.close(fig)

    if not images:
        # Fallback: joint plot on the original idata
        return azp.plot_energy(posteriors, backend="matplotlib")

    p = len(images)
    combined_fig, axes = plt.subplots(p, 1, figsize=(12, 5 * p))
    if p == 1:
        axes = [axes]
    for ax, (factor_name, img) in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Channel {factor_name}", pad=6)
    combined_fig.tight_layout()
    return combined_fig
