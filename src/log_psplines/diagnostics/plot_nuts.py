"""Energy diagnostics for blocked stationary NUTS channels."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import arviz_plots as azp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from log_psplines.diagnostics.summary_tables import channel_idata

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def plot_energy(result: PSDResult) -> plt.Figure:
    """Plot one energy panel for each blocked Cholesky channel."""
    if result.sample_stats is None:
        raise ValueError("Energy diagnostics require sample_stats")
    images = []
    for channel in range(int(result.spectrum.sizes["channel"])):
        plot = azp.plot_energy(
            channel_idata(result, channel), backend="matplotlib"
        )
        figure = plot.viz["figure"].item()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        images.append(mpimg.imread(buffer))
        plt.close(figure)

    combined, axes = plt.subplots(
        len(images), 1, figsize=(12, 5 * len(images))
    )
    if len(images) == 1:
        axes = [axes]
    for channel, (axis, image) in enumerate(zip(axes, images, strict=True)):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(f"Channel {channel}", pad=6)
    combined.tight_layout()
    return combined
