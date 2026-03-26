"""MCMC inference wrapper for LISA simulation study."""

from __future__ import annotations

from typing import Optional, Sequence

import arviz as az
import numpy as np

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.logger import logger
from log_psplines.mcmc import run_mcmc
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

FMIN = 1e-4
FMAX = 1e-1


def run_lisa_mcmc(
    ts: MultivariateTimeseries,
    *,
    Nb: int,
    coarse_cfg: CoarseGrainConfig,
    freq_true: np.ndarray,
    S_true: np.ndarray,
    K: int = 20,
    knot_method: str = "density",
    diff_order: int = 2,
    n_samples: int = 1000,
    n_warmup: int = 1500,
    num_chains: int = 4,
    target_accept: float = 0.7,
    max_tree_depth: int = 10,
    dense_mass: bool = True,
    alpha_delta: float = 3.0,
    beta_delta: float = 3.0,
    vi: bool = False,
    vi_steps: int = 500_000,
    vi_lr: float = 5e-3,
    vi_guide: str = "diag",
    vi_posterior_draws: int = 1024,
    coarse_grain_config_vi: CoarseGrainConfig | None = None,
    auto_coarse_vi: bool = False,
    auto_coarse_vi_target_nfreq: int = 192,
    fmin: float = FMIN,
    fmax: float = FMAX,
    wishart_window: str | tuple[str, float] | None = None,
    wishart_detrend: str | bool = "constant",
    wishart_floor_fraction: float | None = None,
    exclude_freq_bands: Sequence[tuple[float, float]] = (),
    tau: Optional[float] = None,
    only_vi: bool = False,
    outdir: str = ".",
) -> az.InferenceData:
    """Run multivariate MCMC on LISA data.

    Parameters
    ----------
    knot_method : str
        Knot placement method: "density" (quantile-based), "log", or "uniform".
        NOTE: the knot locator kwarg key is ``method``, not ``strategy`` — the
        old ``knot_kwargs=dict(strategy="log")`` was silently ignored.
    diff_order : int
        P-spline difference penalty order (1 or 2).
    wishart_window : str or tuple or None
        Taper applied to each block before the Wishart likelihood FFT.
        None = rectangular, "hann", or ("tukey", alpha).
    """
    true_psd_source = (freq_true, S_true)

    logger.info(
        f"Running LISA MCMC: K={K}, knot_method={knot_method}, "
        f"diff_order={diff_order}, Nb={Nb}, chains={num_chains}, "
        f"warmup={n_warmup}, samples={n_samples}, "
        f"target_accept={target_accept}, max_tree_depth={max_tree_depth}, "
        f"wishart_window={wishart_window}, "
        f"wishart_detrend={wishart_detrend}, "
        f"wishart_floor_fraction={wishart_floor_fraction}, "
        f"vi={'on' if vi else 'off'}, only_vi={only_vi}, "
        f"n_excluded_bands={len(exclude_freq_bands)}, tau={tau}."
    )

    idata = run_mcmc(
        data=ts,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        n_knots=K,
        degree=2,
        diffMatrixOrder=diff_order,
        knot_kwargs=dict(method=knot_method),
        outdir=outdir,
        verbose=True,
        coarse_grain_config=coarse_cfg,
        wishart_window=wishart_window,
        wishart_detrend=wishart_detrend,
        wishart_floor_fraction=wishart_floor_fraction,
        Nb=Nb,
        fmin=fmin,
        fmax=fmax,
        exclude_freq_bands=tuple(exclude_freq_bands),
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        only_vi=only_vi,
        init_from_vi=vi,
        vi_steps=vi_steps if vi else 0,
        vi_lr=vi_lr,
        vi_guide=vi_guide,
        vi_posterior_draws=vi_posterior_draws,
        coarse_grain_config_vi=coarse_grain_config_vi,
        auto_coarse_vi=auto_coarse_vi,
        auto_coarse_vi_target_nfreq=auto_coarse_vi_target_nfreq,
        vi_progress_bar=True,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        true_psd=true_psd_source,
        compute_lnz=False,
        design_psd=true_psd_source if tau is not None else None,
        tau=tau,
    )

    return idata
