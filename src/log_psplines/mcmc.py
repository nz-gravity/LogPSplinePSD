import arviz as az
import jax.numpy as jnp

from .datatypes import Periodogram, Timeseries
from .psplines import LogPSplines
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
    **kwgs,
) -> az.InferenceData:
    """
    MCMC sampling with log P-splines.

    Parameters
    ----------
    pdgrm : Periodogram
        Input periodogram data
    parametric_model : jnp.ndarray, optional
        Parametric model component
    n_samples : int
        Number of samples to collect
    n_warmup : int
        Number of warmup iterations
    sampler : str
        Sampler type: 'nuts' or 'metropolis' (or 'mh')
    **kwgs
        Additional arguments passed to sampler:

        For NUTS:
        - chains: int = 1
        - target_accept_prob: float = 0.8
        - max_tree_depth: int = 10
        - alpha_phi, beta_phi, alpha_delta, beta_delta: float
        - rng_key: int = 42
        - verbose: bool = True

        For Metropolis-Hastings:
        - target_accept_rate: float = 0.44
        - adaptation_window: int = 50
        - adaptation_start: int = 100
        - step_size_factor: float = 1.1
        - min_step_size, max_step_size: float
        - alpha_phi, beta_phi, alpha_delta, beta_delta: float
        - rng_key: int = 42
        - verbose: bool = True

        For spline model:
        - n_knots: int = 10
        - degree: int = 3
        - diffMatrixOrder: int = 2

    Returns
    -------
    az.InferenceData
        ArviZ inference data object containing samples and diagnostics

    Examples
    --------
    # NUTS sampling (default)
    idata = run_mcmc(pdgrm, n_samples=1000, n_warmup=500)

    # Metropolis-Hastings sampling
    idata = run_mcmc(pdgrm, sampler='metropolis', target_accept_rate=0.5)

    # With custom spline configuration
    idata = run_mcmc(pdgrm, sampler='nuts', n_knots=15, degree=3, chains=2)
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

    # Initialize sampler + run
    sampler_obj = sampler_class(
        periodogram=pdgrm, spline_model=spline_model, config=config
    )
    return sampler_obj.sample(n_samples=n_samples, n_warmup=n_warmup)
