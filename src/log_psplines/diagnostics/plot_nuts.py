"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

import io

import arviz_plots as azp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import xarray as xr

from ._factors import factor_idatas


def plot_energy(posteriors: xr.DataTree | dict[str, xr.DataTree]):
    """Energy diagnostic plot.

    For univariate (single NUTS run): delegates directly to azp.plot_energy.
    For multivariate blocked NUTS: plots per-factor energy diagnostics stacked
    vertically, one panel per channel block.
    """
    if isinstance(posteriors, dict):
        factors = posteriors
    else:
        try:
            factors = factor_idatas(posteriors)
        except Exception:
            factors = {"0": posteriors}

    if len(factors) <= 1:
        dt = next(iter(factors.values()))
        try:
            return azp.plot_energy(dt, backend="matplotlib")
        except Exception:
            if isinstance(dt, xr.DataTree) and "sample_stats" in dt.children:
                return azp.plot_energy(
                    dt["sample_stats"], backend="matplotlib"
                )
            raise

    # Multivariate: render each factor's plot as an image and stack vertically
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
        return azp.plot_energy(
            next(iter(factors.values())), backend="matplotlib"
        )

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
