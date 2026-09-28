"""Post-fit diagnostics for NUTS and variational inference."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz_stats as azs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from log_psplines.diagnostics.spectrum import spectrum_diagnostics

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def _channel_idata(result: PSDResult, channel: int) -> xr.DataTree:
    """Build a diagnostics view for one blocked Cholesky channel."""
    scalar_names = {
        f"delta_{channel}",
        f"phi_delta_{channel}",
        f"weights_delta_{channel}",
    }
    prefixes = tuple(
        f"{name}_{channel}_"
        for name in (
            "delta_theta_re",
            "phi_theta_re",
            "weights_theta_re",
            "delta_theta_im",
            "phi_theta_im",
            "weights_theta_im",
        )
    )
    names = [
        name
        for name in result.posterior.data_vars
        if name in scalar_names or name.startswith(prefixes)
    ]
    if not names:
        raise ValueError(f"No posterior variables for channel {channel}")
    posterior = result.posterior[names]
    groups: dict[str, xr.Dataset] = {"posterior": posterior}
    if result.sample_stats is not None:
        suffix = f"_channel_{channel}"
        stats = {
            name.removesuffix(suffix): var
            for name, var in result.sample_stats.data_vars.items()
            if name.endswith(suffix)
        }
        if stats:
            groups["sample_stats"] = xr.Dataset(stats)
    if result.log_likelihood is not None:
        name = f"log_likelihood_block_{channel}"
        if name in result.log_likelihood:
            groups["log_likelihood"] = xr.Dataset(
                {"log_likelihood": result.log_likelihood[name]}
            )
    from arviz_base import from_dict

    return from_dict(groups)


def _stat(idata: xr.DataTree, name: str) -> np.ndarray:
    if "sample_stats" not in idata.children:
        return np.array([], dtype=float)
    dataset = idata["sample_stats"].dataset
    if dataset is None or name not in dataset:
        return np.array([], dtype=float)
    return np.asarray(dataset[name].values, dtype=float).reshape(-1)


def _tree_depth_hits(idata: xr.DataTree, max_depth: int | None) -> int:
    if max_depth is None:
        return 0
    depth = _stat(idata, "tree_depth")
    depth = depth[np.isfinite(depth)]
    if depth.size:
        return int(np.sum(depth >= max_depth))
    steps = _stat(idata, "n_steps")
    return int(np.sum(steps[np.isfinite(steps)] >= 2**max_depth))


def _reduction(summary: pd.DataFrame, name: str, reducer: str) -> float:
    if name not in summary:
        return float("nan")
    values = pd.to_numeric(summary[name], errors="coerce").to_numpy(
        dtype=float
    )
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.max(values) if reducer == "max" else np.min(values))


def sampling_diagnostics(
    result: PSDResult, *, elbo_window: int = 50
) -> dict[str, Any]:
    """Return NUTS or VI diagnostics per blocked factor.

    NumPyro reports the loss (negative ELBO), so ``final_elbo`` negates it.
    """
    output: dict[str, Any] = {}
    n_draws = int(result.posterior.sizes.get("chain", 0)) * int(
        result.posterior.sizes.get("draw", 0)
    )
    if result.sample_stats is not None:
        factors = (
            1
            if result.time is not None
            else int(result.spectrum.sizes["channel"])
        )
        rows = []
        for channel in range(factors):
            idata = (
                result.to_arviz()
                if result.time is not None
                else _channel_idata(result, channel)
            )
            summary = azs.summary(idata)
            step = _stat(idata, "step_size")
            step = step[np.isfinite(step)]
            depths = result.metadata.get("max_tree_depth_by_channel")
            max_depth = (
                int(depths[channel])
                if depths is not None and result.time is None
                else result.metadata.get("max_tree_depth")
            )
            rows.append(
                {
                    "factor": str(channel),
                    "divergences": int(np.sum(_stat(idata, "diverging") > 0)),
                    "rhat_max": _reduction(summary, "r_hat", "max"),
                    "ess_bulk_min": _reduction(summary, "ess_bulk", "min"),
                    "ess_tail_min": _reduction(summary, "ess_tail", "min"),
                    "max_treedepth_hits": _tree_depth_hits(
                        idata, None if max_depth is None else int(max_depth)
                    ),
                    "step_size": float(np.median(step))
                    if step.size
                    else float("nan"),
                    "n_draws": n_draws,
                }
            )
        output["nuts"] = rows
    if result.vi is not None:
        traces = result.vi.losses_per_block or [result.vi.losses]
        rows = []
        for index, trace in enumerate(traces):
            losses = np.asarray(trace, dtype=float).reshape(-1)
            window = min(max(int(elbo_window), 1), losses.size)
            rows.append(
                {
                    "factor": str(index),
                    "final_elbo": float(-losses[-1])
                    if losses.size
                    else float("nan"),
                    "elbo_improvement": float(losses[-window] - losses[-1])
                    if window > 1
                    else float("nan"),
                    "n_draws": n_draws,
                }
            )
        output["vi"] = rows
    return output


def plot_energy(result: PSDResult) -> plt.Figure:
    """Plot energy diagnostics for each NUTS factor."""
    if result.sample_stats is None:
        raise ValueError("Energy diagnostics require sample_stats")
    import arviz_plots as azp

    count = (
        1 if result.time is not None else int(result.spectrum.sizes["channel"])
    )
    images = []
    for channel in range(count):
        idata = (
            result.to_arviz()
            if result.time is not None
            else _channel_idata(result, channel)
        )
        plot = azp.plot_energy(idata, backend="matplotlib")
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


def save_diagnostics(
    result: PSDResult, outdir: str | Path, *, truth: np.ndarray | None = None
) -> None:
    """Write available sampling and truth-comparison diagnostics to CSV."""
    directory = Path(outdir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    sampling = sampling_diagnostics(result)
    for kind, rows in sampling.items():
        pd.DataFrame(rows).to_csv(
            directory / f"{kind}_summary.csv", index=False
        )
    spectrum = spectrum_diagnostics(result, truth=truth)
    if spectrum:
        pd.DataFrame([spectrum]).to_csv(
            directory / "spectrum_summary.csv", index=False
        )


__all__ = ["plot_energy", "sampling_diagnostics", "save_diagnostics"]
