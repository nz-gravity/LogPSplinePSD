import time
from dataclasses import dataclass
from typing import Any, Dict

import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..psplines import LogPSplines
from .base_sampler import BaseSampler, SamplerConfig, log_likelihood


@dataclass
class NUTSConfig(SamplerConfig):
    """Configuration for NUTS sampler."""

    target_accept_prob: float = 0.8
    max_tree_depth: int = 10


# ==================== NUTS SAMPLER ====================


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
    """
    NumPyro NUTS sampler for log P-splines.
    """

    def __init__(
        self,
        log_pdgrm: jnp.ndarray,
        spline_model: LogPSplines,
        config: NUTSConfig = None,
    ):
        if config is None:
            config = NUTSConfig()
        super().__init__(log_pdgrm, spline_model, config)
        self.config = config  # type: NUTSConfig

    def sample(
        self,
        n_samples: int,
        n_warmup: int = 500,
        thin: int = 1,
        chains: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run NUTS sampling."""
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
            print(f"NUTS sampler with {chains} chain(s)")

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
            print(f"NUTS sampling completed in {self.runtime:.2f} seconds")

        # # Get samples and sample stats
        # samples = mcmc.get_samples()
        # sample_stats = mcmc.get_extra_fields()
        #
        # # Reshape for consistency
        # def reshape_samples(x, n_chains, n_samples):
        #     if x.ndim == 1:
        #         return x.reshape(n_chains, n_samples)
        #     else:
        #         return x.reshape(n_chains, n_samples, -1)
        #
        # n_chains = mcmc.num_chains
        # n_samples_per_chain = len(samples['phi']) // n_chains
        #
        # reshaped_samples = {}
        # for key, values in samples.items():
        #     reshaped_samples[key] = reshape_samples(values, n_chains, n_samples_per_chain)
        #
        # reshaped_stats = {}
        # for key, values in sample_stats.items():
        #     reshaped_stats[key] = reshape_samples(values, n_chains, n_samples_per_chain)

        return self.to_arviz(mcmc)

    def to_arviz(self, results) -> az.InferenceData:
        idata = az.from_numpyro(results)
        idata.attrs["sampler"] = "nuts"
        idata.attrs["target_accept_rate"] = self.config.target_accept_prob
        idata.attrs["max_tree_depth"] = self.config.max_tree_depth
        idata.attrs["runtime"] = self.runtime
        idata.attrs["n_samples"] = results.num_samples
        idata.attrs["n_warmup"] = results.num_warmup
        idata.attrs["n_chains"] = results.num_chains
        idata.attrs["n_weights"] = self.n_weights
        idata.attrs["spline_degree"] = self.spline_model.degree
        idata.attrs["n_knots"] = self.spline_model.n_knots

        if self.config.outdir:
            self.plot_diagnostics(idata)
        return idata
