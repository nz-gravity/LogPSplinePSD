from typing import Literal, Optional

import arviz as az
import jax.numpy as jnp

from .datatypes import Periodogram, Timeseries
from .datatypes.multivar import MultivarFFT
from .psplines import LogPSplines
from .psplines.multivar_psplines import MultivariateLogPSplines
from .samplers import (
    MetropolisHastingsConfig,
    MetropolisHastingsSampler,
    NUTSConfig,
    NUTSSampler,
)


def run_mcmc(
    pdgrm: Periodogram,
    parametric_model: jnp.ndarray = None,
    sampler: str = "nuts",
    n_samples: int = 1000,
    n_warmup: int = 500,
    # Direct parameters instead of TypedDict unpacking
    n_knots: int = 10,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    knot_kwargs: dict = {},
    # Sampler parameters
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
    rng_key: int = 42,
    verbose: bool = True,
    outdir: Optional[str] = None,
    compute_lnz: bool = False,
    # NUTS specific
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    # MH specific
    target_accept_rate: float = 0.44,
    adaptation_window: int = 50,
    **kwargs
) -> az.InferenceData:
    """
    Bayesian spectral estimation using MCMC sampling with log P-splines.

    Performs non-parametric Bayesian inference of power spectral densities using
    penalized B-spline models. Particularly useful for gravitational wave data
    analysis where smooth spectral estimates are needed while preserving sharp
    spectral features.

    Parameters
    ----------
    pdgrm : Periodogram
        Input periodogram data, typically from gravitational wave strain
        measurements or detector noise characterization
    parametric_model : jnp.ndarray, optional, default=None
        Known parametric component to subtract (e.g., instrumental lines,
        astrophysical templates). Must match periodogram frequency grid
    sampler : {'nuts', 'mh'}, default='nuts'
        MCMC sampler algorithm:

        - 'nuts': No-U-Turn Sampler (efficient for smooth posteriors)
        - 'mh': Metropolis-Hastings (robust for complex/multimodal posteriors)

    n_samples : int, default=1000
        Number of posterior samples to collect after warmup phase
    n_warmup : int, default=500
        Number of warmup iterations for sampler adaptation and burn-in
    n_knots : int, default=10
        Number of knots for the P-spline basis
    degree : int, default=3
        Degree of the B-spline basis functions
    diffMatrixOrder : int, default=2
        Order of the difference matrix for the penalty term
    knot_kwargs : dict, default={}
        Additional keyword arguments for knot allocation
    alpha_phi : float, default=1.0
        Alpha parameter for the precision prior
    beta_phi : float, default=1.0
        Beta parameter for the precision prior
    alpha_delta : float, default=1e-4
        Alpha parameter for the smoothing prior
    beta_delta : float, default=1e-4
        Beta parameter for the smoothing prior
    rng_key : int, default=42
        Random number generator key
    verbose : bool, default=True
        Whether to print progress information
    outdir : str, optional, default=None
        Directory to save output files
    compute_lnz : bool, default=False
        Whether to compute the log evidence
    target_accept_prob : float, default=0.8
        Target acceptance probability for NUTS
    max_tree_depth : int, default=10
        Maximum tree depth for NUTS
    target_accept_rate : float, default=0.44
        Target acceptance rate for MH
    adaptation_window : int, default=50
        Adaptation window for MH

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object containing:

        - Posterior samples for spline coefficients and smoothing parameters
        - MCMC diagnostics (R-hat, ESS, divergences, energy statistics)
        - Log-likelihood traces and model comparison metrics
        - Reconstructed power spectral density samples
    """

    # Create spline model
    spline_model = LogPSplines.from_periodogram(
        pdgrm,
        n_knots=n_knots,
        degree=degree,
        diffMatrixOrder=diffMatrixOrder,
        parametric_model=parametric_model,
        knot_kwargs=knot_kwargs,
    )

    if sampler == "nuts":
        config = NUTSConfig(
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
            rng_key=rng_key,
            verbose=verbose,
            outdir=outdir,
            compute_lnz=compute_lnz,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        print("Spline model:", spline_model)
        print("Sampler config:", config)
        sampler_obj = NUTSSampler(
            periodogram=pdgrm, spline_model=spline_model, config=config
        )

    elif sampler == "mh":
        config = MetropolisHastingsConfig(
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
            rng_key=rng_key,
            verbose=verbose,
            outdir=outdir,
            compute_lnz=compute_lnz,
            target_accept_rate=target_accept_rate,
            adaptation_window=adaptation_window,
            adaptation_start=100,  # Hardcoded default
            step_size_factor=1.1,  # Hardcoded default
            min_step_size=1e-6,  # Hardcoded default
            max_step_size=10.0,  # Hardcoded default
        )
        print("Spline model:", spline_model)
        print("Sampler config:", config)
        sampler_obj = MetropolisHastingsSampler(
            periodogram=pdgrm, spline_model=spline_model, config=config
        )
    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. Choose 'nuts' or 'mh'."
        )

    # ensure no extra kwargs remain
    if kwargs:
        raise ValueError(f"Unknown arguments: {', '.join(kwargs.keys())}")
    return sampler_obj.sample(n_samples=n_samples, n_warmup=n_warmup)


# TODO: Add unified run_mcmc function and factory functions
# def run_mcmc_unified(data, sampler="nuts", **kwargs) -> az.InferenceData:
#     """Unified interface for both univariate and multivariate MCMC sampling."""
#     sampler_obj = create_sampler_and_model(data, sampler, **kwargs)
#     return sampler_obj.sample(**kwargs)


# def create_sampler_and_model(data, sampler_type="nuts", **kwargs):
#     """Factory function that creates appropriate model and sampler based on data type."""
#     if isinstance(data, Periodogram):
#         # Univariate case
#         from .psplines import LogPSplines
#
#         parametric_model = kwargs.pop("parametric_model", None)
#         spline_kwargs = {k: kwargs.pop(k) for k in ["n_knots", "degree", "diffMatrixOrder", "knot_kwargs"] if k in kwargs}
#
#         model = LogPSplines.from_periodogram(
#             data,
#             n_knots=spline_kwargs.pop("n_knots", 10),
#             degree=spline_kwargs.pop("degree", 3),
#             diffMatrixOrder=spline_kwargs.pop("diffMatrixOrder", 2),
#             parametric_model=parametric_model,
#             knot_kwargs=spline_kwargs.pop("knot_kwargs", {}),
#         )
#
#         # Create appropriate sampler
#         config = create_sampler_config(sampler_type, **kwargs)
#         if sampler_type.lower() == "nuts":
#             sampler = NUTSSampler(data, model, config)
#         elif sampler_type.lower() == "mh":
#             sampler = MetropolisHastingsSampler(data, model, config)
#         else:
#             raise ValueError(f"Unknown sampler: {sampler_type}")
#
#     elif isinstance(data, MultivarFFT):
#         # Multivariate case
#         spline_kwargs = {k: kwargs.pop(k) for k in ["n_knots", "degree", "diffMatrixOrder", "knot_kwargs"] if k in kwargs}
#
#         model = MultivariateLogPSplines.from_multivar_fft(
#             data,
#             n_knots=spline_kwargs.pop("n_knots", 10),
#             degree=spline_kwargs.pop("degree", 3),
#             diffMatrixOrder=spline_kwargs.pop("diffMatrixOrder", 2),
#             knot_kwargs=spline_kwargs.pop("knot_kwargs", {}),
#         )
#
#         # For multivariate, we currently only support NUTS
#         if sampler_type.lower() != "nuts":
#             print(f"Warning: Only NUTS sampling supported for multivariate case. Using NUTS instead of {sampler_type}.")
#
#         # Import multivariate NUTS sampler when ready
#         # config = create_sampler_config("nuts", **kwargs)
#         # sampler = MultivarNUTSSampler(data, model, config)
#         raise NotImplementedError("Multivariate sampling not yet integrated with unified interface")
#
#     else:
#         raise ValueError(f"Unsupported data type: {type(data).__name__}. Expected Periodogram or MultivarFFT.")
#
#     return sampler


# def create_sampler_config(sampler_type, **kwargs):
#     """Create appropriate config object based on sampler type."""
#     common_kwargs = {
#         "alpha_phi": kwargs.pop("alpha_phi", 1.0),
#         "beta_phi": kwargs.pop("beta_phi", 1.0),
#         "alpha_delta": kwargs.pop("alpha_delta", 1e-4),
#         "beta_delta": kwargs.pop("beta_delta", 1e-4),
#         "rng_key": kwargs.pop("rng_key", 42),
#         "verbose": kwargs.pop("verbose", True),
#         "outdir": kwargs.pop("outdir", None),
#         "compute_lnz": kwargs.pop("compute_lnz", False),
#     }
#
#     if sampler_type.lower() == "nuts":
#         return NUTSConfig(
#             **common_kwargs,
#             target_accept_prob=kwargs.pop("target_accept_prob", 0.8),
#             max_tree_depth=kwargs.pop("max_tree_depth", 10),
#         )
#     elif sampler_type.lower() == "mh":
#         return MetropolisHastingsConfig(
#             **common_kwargs,
#             target_accept_rate=kwargs.pop("target_accept_rate", 0.44),
#             adaptation_window=kwargs.pop("adaptation_window", 50),
#             adaptation_start=kwargs.pop("adaptation_start", 100),
#             step_size_factor=kwargs.pop("step_size_factor", 1.1),
#             min_step_size=kwargs.pop("min_step_size", 1e-6),
#             max_step_size=kwargs.pop("max_step_size", 10.0),
#         )
#     else:
#         raise ValueError(f"Unknown sampler type: {sampler_type}")
