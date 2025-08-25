import arviz as az
import numpy as np
from log_psplines.arviz_utils import compare_results
from log_psplines.plotting import  plot_pdgrm
import matplotlib.pyplot as plt


f1 = "out_lvk_mcmc/inference_data.nc"
f2 = "out_lvk_mcmc_more_knots/inference_data.nc"

i1 = az.from_netcdf(f1)
i2 = az.from_netcdf(f2)

# compare_results(
#     az.from_netcdf(f1),
#     az.from_netcdf(f2),
#     labels=["MH 3 knots", "MH 5 knots"],
#     outdir=".",
#     colors=["blue", "orange"],
# )

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plot_pdgrm(
    idata=i1,
    ax=ax,
    model_label="3 knots",
    model_color="tab:blue",
    knot_color="lightblue",
    show_knots=True,
)
plot_pdgrm(
    idata=i2,
    ax=ax,
    model_color="darkorange",
    knot_color="orange",
    model_label="5 knot",
    data_label="_",
    show_knots=True,
)
ax.set_xscale("linear")
ax.legend()
plt.tight_layout()
fig.savefig("compare_knots.png", transparent=False, bbox_inches="tight", dpi=300)
# plt.show()



f3 = "out_lvk_mcmc_nuts/inference_data.nc"
i3 = az.from_netcdf(f3)
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plot_pdgrm(
    idata=i3,
    ax=ax,
    model_label="NUTS",
    model_color="tab:blue",
    knot_color="lightblue",
    show_knots=False,
)
plot_pdgrm(
    idata=i2,
    ax=ax,
    model_color="darkorange",
    knot_color="orange",
    model_label="MCMC",
    data_label="_",
    show_knots=False,
)
ax.set_xscale("linear")
# ax.legend()
plt.tight_layout()
plt.savefig("compare_nuts_mcmc.png", transparent=False, bbox_inches="tight", dpi=300)
