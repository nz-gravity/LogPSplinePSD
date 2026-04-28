import arviz as az
import matplotlib.pyplot as plt

from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix

f1 = "out_lvk_mcmc/inference_data.nc"
f2 = "out_lvk_mcmc_more_knots/inference_data.nc"

i1 = az.from_netcdf(f1)
i2 = az.from_netcdf(f2)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plot_psd_matrix(
    PSDMatrixPlotSpec(
        idata=i1,
        fig=fig,
        ax=ax,
        label="3 knots",
        model_color="tab:blue",
        show_knots=True,
    )
)
plot_psd_matrix(
    PSDMatrixPlotSpec(
        idata=i2,
        fig=fig,
        ax=ax,
        label="5 knot",
        model_color="darkorange",
        show_knots=True,
        show_empirical=False,
    )
)
ax.set_xscale("linear")
ax.legend()
plt.tight_layout()
fig.savefig(
    "compare_knots.png", transparent=False, bbox_inches="tight", dpi=300
)
# plt.show()


f3 = "out_lvk_mcmc_nuts/inference_data.nc"
i3 = az.from_netcdf(f3)
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plot_psd_matrix(
    PSDMatrixPlotSpec(
        idata=i3,
        fig=fig,
        ax=ax,
        label="NUTS",
        model_color="tab:blue",
        show_knots=False,
    )
)
plot_psd_matrix(
    PSDMatrixPlotSpec(
        idata=i2,
        fig=fig,
        ax=ax,
        label="MCMC",
        model_color="darkorange",
        show_knots=False,
        show_empirical=False,
    )
)
ax.set_xscale("linear")
# ax.legend()
plt.tight_layout()
plt.savefig(
    "compare_nuts_mcmc.png", transparent=False, bbox_inches="tight", dpi=300
)
