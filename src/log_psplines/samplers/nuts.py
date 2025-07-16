import time
from dataclasses import dataclass

import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..psplines import LogPSplines, Periodogram
from .base_sampler import BaseSampler, SamplerConfig, log_likelihood


@dataclass
class NUTSConfig(SamplerConfig):
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10


def bayesian_model(
    log_pdgrm: jnp.ndarray,
    lnspline_basis: jnp.ndarray,
    penalty_matrix: jnp.ndarray,
    ln_parametric: jnp.ndarray,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
):
    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample("delta", delta_dist)

    phi_dist = dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi)
    phi = numpyro.sample("phi", phi_dist)

    # Sample v from an unregularized Normal(0,1). We do dimension k from penalty_matrix
    k = penalty_matrix.shape[0]
    w = numpyro.sample("weights", dist.Normal(0, 1).expand([k]).to_event(1))

    # Add a custom factor for the prior p(v | phi, delta) ~ MVN(0, (phi*P)^-1)
    wPw = jnp.dot(w, jnp.dot(penalty_matrix, w))
    log_prior_v = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    numpyro.factor("ln_prior", log_prior_v)
    numpyro.factor(
        "ln_likelihood",
        log_likelihood(w, log_pdgrm, lnspline_basis, ln_parametric),
    )


class NUTSSampler(BaseSampler):

    def __init__(
        self,
        periodogram: Periodogram,
        spline_model: LogPSplines,
        config: NUTSConfig = None,
    ):
        if config is None:
            config = NUTSConfig()
        super().__init__(periodogram, spline_model, config)
        self.config = config  # type: NUTSConfig

    def sample(
        self,
        n_samples: int,
        n_warmup: int = 500,
        thin: int = 1,
        chains: int = 1,
        **kwargs,
    ) -> az.InferenceData:
        # Initialize starting values
        delta_0 = self.config.alpha_delta / self.config.beta_delta
        phi_0 = self.config.alpha_phi / (self.config.beta_phi * delta_0)
        init_strategy = init_to_value(
            values=dict(
                delta=delta_0, phi=phi_0, weights=self.spline_model.weights
            )
        )

        # Setup NUTS
        kernel = NUTS(
            bayesian_model,
            init_strategy=init_strategy,
            target_accept_prob=self.config.target_accept_prob,
            max_tree_depth=self.config.max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=chains,
            progress_bar=self.config.verbose,
            jit_model_args=True,
        )

        if self.config.verbose:
            print(f"NUTS sampler with {chains} chain(s) [{self.device}]")

        start_time = time.time()
        mcmc.run(
            self.rng_key,
            self.log_pdgrm,
            self.basis_matrix,
            self.penalty_matrix,
            self.log_parametric,
            self.config.alpha_phi,
            self.config.beta_phi,
            self.config.alpha_delta,
            self.config.beta_delta,
        )
        self.runtime = time.time() - start_time

        if self.config.verbose:
            print(f"Sampling completed in {self.runtime:.2f} seconds")

        return self.to_arviz(mcmc)

    def to_arviz(self, results: MCMC) -> az.InferenceData:
        idata = az.from_numpyro(results)
        idata = self._add_common_attrs_and_save(idata)
        return idata
