import matplotlib.pyplot as plt

from log_psplines.arviz_utils.from_arviz import get_spline_model
from log_psplines.example_datasets import ARData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.plotting import plot_pdgrm

ar4 = ARData(order=4, duration=2.0, fs=512.0, sigma=1.0, seed=42)
model_cfg = ModelConfig(n_knots=15, knot_kwargs={"method": "uniform"})
diagnostics_cfg = DiagnosticsConfig(outdir="out/nuts_out")
vi_cfg = VIConfig(init_from_vi=True)
run_cfg = RunMCMCConfig(
    n_samples=2500,
    n_warmup=1000,
    rng_key=0,
    model=model_cfg,
    diagnostics=diagnostics_cfg,
    vi=vi_cfg,
)
inference_nuts = run_mcmc(ar4.ts, config=run_cfg)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(
    ar4.freqs,
    ar4.psd_theoretical,
    color="k",
    linestyle="--",
    label="True PSD",
    zorder=10,
)
plot_pdgrm(
    idata=inference_nuts,
    ax=ax,
    model_color="tab:orange",
    model_label="NUTS",
    data_label="_",
    show_knots=True,
)

ax.set_xscale("linear")
fig.savefig("demo.png", transparent=True, bbox_inches="tight", dpi=300)

get_spline_model(inference_nuts).plot_basis()
