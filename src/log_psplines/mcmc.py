from typing import Literal
from typing import get_type_hints
from typing_extensions import get_args

import arviz as az
import jax.numpy as jnp
from typing_extensions import Unpack

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
from .types import MHKwargs, NUTSKwargs, SamplerKwargs, SplineKwargs


def run_mcmc(
    pdgrm: Periodogram,
    parametric_model: jnp.ndarray = None,
    sampler: str = "nuts",
    n_samples: int = 1000,
    n_warmup: int = 500,
    **kwgs: (
        Unpack[NUTSKwargs]
        | Unpack[MHKwargs]
        | Unpack[SplineKwargs]
        | Unpack[SamplerKwargs]
    ),
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
    **kwgs : TypedDict parameters
        Configuration parameters from :class:`NUTSKwargs`, :class:`MHKwargs`, 
        :class:`SplineKwargs`, and :class:`SamplerKwargs`. Key parameters include:
        
        **Spline model**: ``n_knots`` (default=10), ``degree`` (default=3)
        **NUTS**: ``target_accept_prob`` (default=0.8), ``max_tree_depth`` (default=10)  
        **MH**: ``target_accept_rate`` (default=0.44), ``adaptation_window`` (default=50)
        **Priors**: ``alpha_phi``, ``beta_phi`` (default=1.0), ``alpha_delta``, ``beta_delta`` (default=1e-4)

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object containing:
        
        - Posterior samples for spline coefficients and smoothing parameters
        - MCMC diagnostics (R-hat, ESS, divergences, energy statistics)
        - Log-likelihood traces and model comparison metrics
        - Reconstructed power spectral density samples

    Notes
    -----
    The log P-spline approach models the log power spectral density as a smooth
    function using B-spline basis functions with a roughness penalty. This provides
    flexible spectral estimation while avoiding overfitting through automatic
    smoothness selection via the hierarchical Bayesian framework.

    For gravitational wave applications, this method excels at:
    - Noise power spectral density estimation for detector characterization
    - Non-parametric background estimation for transient searches  
    - Spectral line identification and characterization
    - Model-independent spectral reconstruction

    Examples
    --------
    Basic spectral estimation with default NUTS sampler:

    >>> idata = run_mcmc(pdgrm, n_samples=2000, n_warmup=1000)

    High-resolution analysis for detecting narrow spectral features:

    >>> idata = run_mcmc(
    ...     pdgrm, 
    ...     n_knots=20, 
    ...     degree=3,
    ...     target_accept_prob=0.9,
    ...     n_samples=3000
    ... )

    Robust sampling with Metropolis-Hastings for difficult posteriors:

    >>> idata = run_mcmc(
    ...     pdgrm, 
    ...     sampler='mh',
    ...     target_accept_rate=0.44,
    ...     adaptation_window=100,
    ...     n_samples=5000
    ... )

    Analysis with known parametric component removal:

    >>> # Subtract known 60 Hz line and harmonics
    >>> template = create_line_template(frequencies, [60, 120, 180])
    >>> idata = run_mcmc(pdgrm, parametric_model=template)

    Custom prior specification for strong smoothing:

    >>> idata = run_mcmc(
    ...     pdgrm,
    ...     alpha_delta=1e-2,  # Stronger smoothing prior
    ...     beta_delta=1e-2,
    ...     n_knots=15
    ... )

    See Also
    --------
    NUTSKwargs : NUTS sampler configuration parameters
    MHKwargs : Metropolis-Hastings sampler configuration  
    SplineKwargs : P-spline model configuration
    SamplerKwargs : Common sampler parameters
    """

    # Extract spline model kwargs
    spline_kwargs = {
        key: kwgs.pop(key)
        for key in ["n_knots", "degree", "diffMatrixOrder", "knot_kwargs"]
        if key in kwgs
    }

    # Common sampler kwgs
    common_kwgs = dict(
        alpha_phi=kwgs.pop("alpha_phi", 1.0),
        beta_phi=kwgs.pop("beta_phi", 1.0),
        alpha_delta=kwgs.pop("alpha_delta", 1e-4),
        beta_delta=kwgs.pop("beta_delta", 1e-4),
        rng_key=kwgs.pop("rng_key", 42),
        verbose=kwgs.pop("verbose", True),
        outdir=kwgs.pop("outdir", None),
        compute_lnz=kwgs.pop("compute_lnz", False),
    )

    if sampler == "nuts":
        # Extract NUTS-specific kwargs
        config = NUTSConfig(
            **common_kwgs,
            target_accept_prob=kwgs.pop("target_accept_prob", 0.8),
            max_tree_depth=kwgs.pop("max_tree_depth", 10),
        )
        sampler_class = NUTSSampler

    elif sampler == "mh":
        # Extract Metropolis-Hastings specific kwargs
        config = MetropolisHastingsConfig(
            **common_kwgs,
            target_accept_rate=kwgs.pop("target_accept_rate", 0.44),
            adaptation_window=kwgs.pop("adaptation_window", 50),
            adaptation_start=kwgs.pop("adaptation_start", 100),
            step_size_factor=kwgs.pop("step_size_factor", 1.1),
            min_step_size=kwgs.pop("min_step_size", 1e-6),
            max_step_size=kwgs.pop("max_step_size", 10.0),
        )
        sampler_class = MetropolisHastingsSampler
    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. Choose 'nuts' or 'mh'."
        )

    # ensure no extra kwargs remain
    if kwgs:
        raise ValueError(f"Unknown arguments: {', '.join(kwgs.keys())}")

    # Create spline model
    spline_model = LogPSplines.from_periodogram(
        pdgrm,
        n_knots=spline_kwargs.pop("n_knots", 10),
        degree=spline_kwargs.pop("degree", 3),
        diffMatrixOrder=spline_kwargs.pop("diffMatrixOrder", 2),
        parametric_model=parametric_model,
        knot_kwargs=spline_kwargs.pop("knot_kwargs", {}),
    )

    print("Spline model:", spline_model)
    print("Sampler config:", config)

    # Initialize sampler + run
    sampler_obj = sampler_class(
        periodogram=pdgrm, spline_model=spline_model, config=config
    )
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
