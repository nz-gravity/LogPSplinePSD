from typing import Union, Literal, Optional

import arviz as az
import jax.numpy as jnp

from .datatypes import Periodogram
from .datatypes.multivar import MultivarFFT
from .datatypes.univar import Timeseries
from .datatypes.multivar import MultivariateTimeseries
from .datatypes.multivar import MultivarFFT
from .psplines import LogPSplines, MultivariateLogPSplines
from .samplers import (
    MetropolisHastingsConfig,
    MetropolisHastingsSampler,
    MultivarNUTSConfig,
    MultivarNUTSSampler,
    NUTSConfig,
    NUTSSampler,
)


def run_mcmc(
    data: Union[Timeseries, MultivariateTimeseries, Periodogram, MultivarFFT],
    sampler: Literal["nuts", "mh"] = "nuts",
    n_samples: int = 1000,
    n_warmup: int = 500,
    num_chains: int = 1,
    # Model parameters
    n_knots: int = 10,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    knot_kwargs: dict = {},
    parametric_model: Optional[jnp.ndarray] = None,
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
    Unified MCMC interface for both univariate and multivariate PSD estimation.

    Parameters
    ----------
    data : Timeseries, MultivariateTimeseries, Periodogram or MultivarFFT
        Input data for analysis. When timeseries data is provided, it will be
        automatically standardized for numerical stability, and posterior samples
        will be rescaled to original units.
    sampler : {"nuts", "mh"}
        MCMC sampler type (note: multivariate only supports "nuts")
    n_samples : int, default=1000
        Number of posterior samples to collect
    n_warmup : int, default=500
        Number of warmup/burn-in samples
    num_chains : int, default=1
        Number of MCMC chains to run for convergence diagnostics
    n_knots : int, default=10
        Number of knots for B-spline basis
    degree : int, default=3
        Degree of B-spline basis functions
    diffMatrixOrder : int, default=2
        Order of difference penalty matrix
    knot_kwargs : dict, default={}
        Additional keyword arguments for knot allocation
    parametric_model : Optional[jnp.ndarray], default=None
        Known parametric component (univariate only)
    alpha_phi : float, default=1.0
        Alpha parameter for precision prior
    beta_phi : float, default=1.0
        Beta parameter for precision prior
    alpha_delta : float, default=1e-4
        Alpha parameter for smoothing prior
    beta_delta : float, default=1e-4
        Beta parameter for smoothing prior
    rng_key : int, default=42
        Random number generator key
    verbose : bool, default=True
        Whether to print progress information
    outdir : Optional[str], default=None
        Directory to save output files
    compute_lnz : bool, default=False
        Whether to compute log evidence
    target_accept_prob : float, default=0.8
        Target acceptance probability for NUTS
    max_tree_depth : int, default=10
        Maximum tree depth for NUTS
    target_accept_rate : float, default=0.44
        Target acceptance rate for MH
    adaptation_window : int, default=50
        Adaptation window for MH
    **kwargs
        Additional keyword arguments

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object with MCMC results
    """

    # Handle raw timeseries input - standardize automatically
    processed_data = None

    if isinstance(data, (Timeseries, MultivariateTimeseries)):
        # Standardize the raw timeseries for numerical stability
        standardized_ts = data.standardise_for_psd()

        # Convert to processed format for existing pipeline
        if isinstance(data, Timeseries):
            processed_data = standardized_ts.to_periodogram()
        else:  # MultivariateTimeseries
            processed_data = standardized_ts.to_cross_spectral_density()

        if verbose:
            print(f"Standardized data: original scale ~{processed_data.scaling_factor:.2e}")

        # Validate sampler for standardized data
        if isinstance(processed_data, MultivarFFT) and sampler != "nuts":
            if verbose:
                print(f"Warning: Multivariate analysis only supports NUTS. Using NUTS instead of {sampler}")
            sampler = "nuts"

    if isinstance(data, (Periodogram, MultivarFFT)):
        processed_data = data  # Use as is


    # Create model based on processed data type
    if isinstance(processed_data, Periodogram):
        # Univariate case
        model = LogPSplines.from_periodogram(
            processed_data,
            n_knots=n_knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            parametric_model=parametric_model if isinstance(data, (Periodogram, Timeseries)) else None,
            knot_kwargs=knot_kwargs,
        )
    elif isinstance(processed_data, MultivarFFT):
        # Multivariate case
        if parametric_model is not None and isinstance(data, Periodogram):
            raise ValueError("parametric_model is not supported for multivariate data. "
                           "Parametric models are only available for univariate Periodogram data.")
        model = MultivariateLogPSplines.from_multivar_fft(
            processed_data,
            n_knots=n_knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            knot_kwargs=knot_kwargs,
        )
    else:
        raise ValueError(f"Unsupported processed data type: {type(processed_data)}.")

    # Create sampler
    # Always use processed_data (the standardized Periodogram or MultivarFFT) for the sampler and model
    sampler_obj = create_sampler(
        data=processed_data,  # Always use processed_data, which has correct scaling_factor
        model=model,
        sampler_type=sampler,
        num_chains=num_chains,
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
        target_accept_rate=target_accept_rate,
        adaptation_window=adaptation_window,
        scaling_factor=processed_data.scaling_factor,  # Pass scaling info to sampler
        **kwargs
    )

    return sampler_obj.sample(n_samples=n_samples, n_warmup=n_warmup)

def create_sampler(
    data: Union[Periodogram, MultivarFFT],
    model,
    sampler_type: Literal["nuts", "mh"] = "nuts",
    num_chains: int = 1,
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
    rng_key: int = 42,
    verbose: bool = True,
    outdir: Optional[str] = None,
    compute_lnz: bool = False,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    target_accept_rate: float = 0.44,
    adaptation_window: int = 50,
    scaling_factor: float = 1.0,
    **kwargs
):

    """Factory function to create appropriate sampler."""

    common_config_kwargs = {
        "alpha_phi": alpha_phi,
        "beta_phi": beta_phi,
        "alpha_delta": alpha_delta,
        "beta_delta": beta_delta,
        "num_chains": num_chains,
        "rng_key": rng_key,
        "verbose": verbose,
        "outdir": outdir,
        "compute_lnz": compute_lnz,
        "scaling_factor": scaling_factor
    }

    if isinstance(data, Periodogram):
        # Univariate case
        if sampler_type == "nuts":
            config = NUTSConfig(
                **common_config_kwargs,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
            )
            return NUTSSampler(data, model, config)
        elif sampler_type == "mh":
            config = MetropolisHastingsConfig(
                **common_config_kwargs,
                target_accept_rate=target_accept_rate,
                adaptation_window=adaptation_window,
                adaptation_start=100,
                step_size_factor=1.1,
                min_step_size=1e-6,
                max_step_size=10.0,
            )
            return MetropolisHastingsSampler(data, model, config)
        else:
            raise ValueError(f"Unknown sampler_type '{sampler_type}' for univariate data. Choose 'nuts' or 'mh'.")

    elif isinstance(data, MultivarFFT):
        # Multivariate case (NUTS only for now)
        if sampler_type != "nuts":
            if verbose:
                print(f"Warning: Multivariate analysis only supports NUTS. Using NUTS instead of {sampler_type}")
        config = MultivarNUTSConfig(
            **common_config_kwargs,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        return MultivarNUTSSampler(data, model, config)

    else:
        raise ValueError(f"Unsupported data type: {type(data).__name__}. Expected Periodogram or MultivarFFT.")

