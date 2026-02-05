"""Generate 1D marginal trace plots for weight parameters."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    idata_path = Path("docs/studies/lisa/results/lisa/inference_data.nc")
    output_dir = Path("docs/studies/lisa/results/lisa/diagnostics/weights_trace")
    output_dir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(idata_path)
    weight_vars = [
        name for name in idata.posterior.data_vars if name.startswith("weights_")
    ]

    if not weight_vars:
        raise ValueError("No posterior variables starting with 'weights_' found.")

    for var_name in weight_vars:
        var = idata.posterior[var_name]
        subplot_dims = [dim for dim in var.dims if dim not in ("chain", "draw")]
        num_subplots = int(np.prod([var.sizes[dim] for dim in subplot_dims]))
        az.rcParams["plot.max_subplots"] = max(num_subplots, 1)

        axes = az.plot_trace(
            idata,
            var_names=[var_name],
            combined=False,
            compact=False,
            kind="trace",
        )
        fig = axes.ravel()[0].figure
        fig.suptitle(f"Trace plots: {var_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"trace_{var_name}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
