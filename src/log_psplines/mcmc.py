import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from .bayesian_model import bayesian_model
from .psplines import LogPSplines
from .datasets import Periodogram


def run_mcmc(
        pdgrm: Periodogram,
        alpha_phi=1.0,
        beta_phi=1.0,
        alpha_delta=1e-4,
        beta_delta=1e-4,
        num_warmup=500,
        num_samples=1000,
        rng_key=0,
):
    # Initialize the model + starting values
    rng_key = jax.random.PRNGKey(rng_key)
    log_pdgrm = jnp.log(pdgrm.power)
    spline_model = LogPSplines.from_periodogram(
        pdgrm,
        n_knots=20,
        degree=3,
        diffMatrixOrder=2,
    )
    delta_0 = alpha_delta / beta_delta
    phi_0 = alpha_phi / (beta_phi * delta_0)
    init_strategy = init_to_value(values=dict(
        delta=delta_0,
        phi=phi_0,
        weights=spline_model.weights
    ))

    # Setup and run MCMC using NUTS
    kernel = NUTS(bayesian_model, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True, jit_model_args=True)
    mcmc.run(
        rng_key,
        log_pdgrm,
        spline_model.basis,
        spline_model.penalty_matrix,
        alpha_phi,
        beta_phi,
        alpha_delta,
        beta_delta
    )

    samples = mcmc.get_samples()
    return samples, spline_model
