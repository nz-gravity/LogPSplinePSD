import numpy as np
import xarray as xr
from arviz import convert_to_dataset, ess
from arviz.utils import _var_names, get_coords


def plot_ess_evolution(
    idata, n_points=50, ess_threshold=400, ax=None, color="tab:blue"
):
    coords: dict[str, object] = {}
    data = get_coords(convert_to_dataset(idata, group="posterior"), coords)
    var_names = _var_names(None, data, None)
    n_draws = data.sizes["draw"]
    n_samples = n_draws * data.sizes["chain"]

    # Setup draw slicing
    first_draw = data.draw.values[0]
    xdata = np.linspace(n_samples / n_points, n_samples, n_points)
    draw_divisions = np.linspace(
        n_draws // n_points, n_draws, n_points, dtype=int
    )

    # Compute ESS for each draw slice
    ess_dataset = xr.concat(
        [
            ess(
                data.sel(draw=slice(first_draw, first_draw + draw_div)),
                var_names=var_names,
                relative=False,
                method="bulk",
            )
            for draw_div in draw_divisions
        ],
        dim="ess_dim",
    )

    ess_tail_dataset = xr.concat(
        [
            ess(
                data.sel(draw=slice(first_draw, first_draw + draw_div)),
                var_names=var_names,
                relative=False,
                method="tail",
            )
            for draw_div in draw_divisions
        ],
        dim="ess_dim",
    )

    # Convert datasets to (n_vars, n_points) arrays
    def _dataset_to_ndarray(dataset):
        return np.concatenate(
            [
                v.values.reshape(v.shape[0], -1).T
                for v in dataset.data_vars.values()
            ],
            axis=0,
        )

    x = _dataset_to_ndarray(ess_dataset)
    xtail = _dataset_to_ndarray(ess_tail_dataset)

    for xi, xtaili in zip(x, xtail):
        ax.plot(xdata, xi, alpha=0.5, color=color)
        ax.plot(xdata, xtaili, alpha=0.5, color=color, linestyle="dotted")
    ax.axhline(
        ess_threshold,
        linestyle="--",
        color="gray",
        label=f"ESS = {ess_threshold}",
    )
    ax.set_xlabel("Total number of draws")
    ax.set_ylabel("ESS")
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(xdata), max(xdata))
    ax.set_title("ESS Evolution (Bulk & Tail)")
