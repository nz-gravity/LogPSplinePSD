"""
NUTS sampler for univariate PSD estimation.
"""

import time
from dataclasses import dataclass

import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..base_sampler import SamplerConfig
from .metropolis_hastings import log_posterior as mh_log_posterior
from .univar_base import UnivarBaseSampler, log_likelihood  # Updated import


@dataclass
class NUTSConfig(SamplerConfig):
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    save_nuts_diagnostics: bool = True


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
    """NumPyro model for univariate PSD estimation."""
    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample("delta", delta_dist)

    phi_dist = dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi)
    phi = numpyro.sample("phi", phi_dist)

    # Sample weights from unregularized Normal(0,1)
    k = penalty_matrix.shape[0]
    w = numpyro.sample("weights", dist.Normal(0, 1).expand([k]).to_event(1))

    # Add custom factor for the prior p(w | phi, delta) ~ MVN(0, (phi*P)^-1)
    wPw = jnp.dot(w, jnp.dot(penalty_matrix, w))
    log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    lnl = log_likelihood(w, log_pdgrm, lnspline_basis, ln_parametric)

    numpyro.factor("ln_prior", log_prior_w)
    numpyro.factor("ln_likelihood", lnl)
    numpyro.deterministic("lp", log_prior_w + lnl)


class NUTSSampler(UnivarBaseSampler):
    """NUTS sampler for univariate PSD estimation."""

    def __init__(self, periodogram, spline_model, config: NUTSConfig = None):
        if config is None:
            config = NUTSConfig()
        super().__init__(periodogram, spline_model, config)
        self.config: NUTSConfig = config  # Type hint for IDE

        # Pre-compile NumPyro model for faster warmup
        self._compile_model()

    @property
    def sampler_type(self) -> str:
        """Required by base class."""
        return "nuts"

    def sample(
        self, n_samples: int, n_warmup: int = 500, **kwargs
    ) -> az.InferenceData:
        """Run NUTS sampling."""
        # Initialize starting values
        delta_0 = self.config.alpha_delta / self.config.beta_delta
        phi_0 = self.config.alpha_phi / (self.config.beta_phi * delta_0)
        init_strategy = init_to_value(
            values=dict(
                delta=delta_0, phi=phi_0, weights=self.spline_model.weights
            )
        )

        # Setup NUTS kernel
        kernel = NUTS(
            bayesian_model,
            init_strategy=init_strategy,
            target_accept_prob=self.config.target_accept_prob,
            max_tree_depth=self.config.max_tree_depth,
            dense_mass=self.config.dense_mass,
        )

        # Setup MCMC
        mcmc = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=self.config.num_chains,
            progress_bar=self.config.verbose,
            jit_model_args=True,
        )

        if self.config.verbose:
            print(f"NUTS sampler [{self.device}] {self.rng_key}")

        # Run sampling
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
            extra_fields=(
                (
                    "potential_energy",
                    "energy",
                    "num_steps",
                    "accept_prob",
                )
                if self.config.save_nuts_diagnostics
                else ()
            ),
        )
        self.runtime = time.time() - start_time

        if self.config.verbose:
            print(f"Sampling completed in {self.runtime:.2f} seconds")

        # Extract samples and convert to ArviZ
        samples = mcmc.get_samples()
        stats = mcmc.get_extra_fields()
        stats["lp"] = samples.pop("lp")

        return self.to_arviz(samples, stats)

    @property
    def _logp_kwargs(self):
        """Arguments passed to the NumPyro model / log-density helpers."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            lnspline_basis=self.basis_matrix,
            penalty_matrix=self.penalty_matrix,
            ln_parametric=self.log_parametric,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _compile_model(self) -> None:
        """Pre-compile the NumPyro model to speed up warmup."""
        try:
            from numpyro.infer.util import initialize_model

            if self.config.verbose:
                print("Pre-compiling NumPyro model...")

            # Initialize model with dummy data to trigger compilation
            init_params = initialize_model(
                self.rng_key,
                bayesian_model,
                model_kwargs=self._logp_kwargs,
                init_strategy=init_to_value(
                    values=dict(
                        delta=self.config.alpha_delta / self.config.beta_delta,
                        phi=self.config.alpha_phi / self.config.beta_phi,
                        weights=self.spline_model.weights,
                    )
                ),
            )

            if self.config.verbose:
                print("Model pre-compilation completed")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Model pre-compilation failed: {e}")

    def _compute_log_posterior(
        self, weights: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """
        Compute log posterior for given parameters.

        This implements the log posterior computation that was in your MH sampler.
        Could be used instead of numpyro's log_density if needed.
        """
        # You could move the log_posterior function from your MH sampler here
        return float(
            mh_log_posterior(
                weights,
                phi,
                delta,
                log_pdgrm=self.log_pdgrm,
                basis_matrix=self.basis_matrix,
                log_parametric=self.log_parametric,
                penalty_matrix=self.penalty_matrix,
                alpha_phi=self.config.alpha_phi,
                beta_phi=self.config.beta_phi,
                alpha_delta=self.config.alpha_delta,
                beta_delta=self.config.beta_delta,
            )
        )
