"""
NUTS sampler for univariate PSD estimation.
"""

import time
from dataclasses import dataclass
from typing import Dict

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..base_sampler import SamplerConfig
from ..utils import (
    build_log_density_fn,
    evaluate_log_density_batch,
    pspline_hyperparameter_initials,
    sample_pspline_block,
)
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
    penalty_whiten: jnp.ndarray | None,
    penalty_unwhiten_T: jnp.ndarray | None,
    ln_parametric: jnp.ndarray,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
):
    """NumPyro model for univariate PSD estimation."""
    if penalty_whiten is None:
        penalty_whiten = jnp.linalg.cholesky(
            penalty_matrix + 1e-6 * jnp.eye(penalty_matrix.shape[0])
        )
    block = sample_pspline_block(
        delta_name="delta",
        phi_name="phi",
        weights_name="weights",
        penalty_whiten=penalty_whiten,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
    )

    weights = block["weights"]
    lnl = log_likelihood(weights, log_pdgrm, lnspline_basis, ln_parametric)
    numpyro.factor("ln_likelihood", lnl)


class NUTSSampler(UnivarBaseSampler):
    """NUTS sampler for univariate PSD estimation."""

    def __init__(self, periodogram, spline_model, config: NUTSConfig = None):
        if config is None:
            config = NUTSConfig()
        super().__init__(periodogram, spline_model, config)
        self.config: NUTSConfig = config  # Type hint for IDE

        # Pre-build JIT-compiled log-posterior for morphZ evidence
        self._logpost_fn = build_log_density_fn(
            bayesian_model, self._logp_kwargs
        )

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
        delta_0, phi_0 = pspline_hyperparameter_initials(
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
            divide_phi_by_delta=True,
        )
        weights_latent_0 = self.penalty_unwhiten_T @ (
            self.spline_model.weights * jnp.sqrt(phi_0)
        )
        init_strategy = init_to_value(
            values=dict(
                delta=delta_0,
                phi=jnp.log(phi_0),
                weights=self.spline_model.weights,
                weights_latent=weights_latent_0,
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
            self.penalty_whiten,
            self.penalty_unwhiten_T,
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
                    "diverging",
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

        params_batch = self._prepare_logpost_params(samples)
        stats["lp"] = evaluate_log_density_batch(
            self._logpost_fn, params_batch
        )

        if "phi" in samples:
            samples["phi"] = jnp.exp(samples["phi"])

        # Drop auxiliary whitening latents before exporting results.
        samples.pop("weights_latent", None)

        return self.to_arviz(samples, stats)

    def _compile_model(self) -> None:
        """Pre-compile the NumPyro model to speed up warmup."""
        try:
            from numpyro.infer.util import initialize_model

            if self.config.verbose:
                print("Pre-compiling NumPyro model...")

            # Initialize model with dummy data to trigger compilation
            delta_phi_default = pspline_hyperparameter_initials(
                alpha_phi=self.config.alpha_phi,
                beta_phi=self.config.beta_phi,
                alpha_delta=self.config.alpha_delta,
                beta_delta=self.config.beta_delta,
            )

            init_params = initialize_model(
                self.rng_key,
                bayesian_model,
                model_kwargs=self._logp_kwargs,
                init_strategy=init_to_value(
                    values=dict(
                        delta=delta_phi_default[0],
                        phi=jnp.log(delta_phi_default[1]),
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
        """Compute log posterior for given parameters via the NumPyro model."""
        weights_arr = jnp.asarray(weights)
        phi_arr = jnp.asarray(phi)
        delta_arr = jnp.asarray(delta)
        latent = self.penalty_unwhiten_T @ (weights_arr * jnp.sqrt(phi_arr))
        params = {
            "weights": weights_arr,
            "weights_latent": latent,
            "phi": jnp.log(phi_arr),
            "delta": delta_arr,
        }
        return float(self._logpost_fn(params))

    def _prepare_logpost_params(self, samples: Dict[str, jnp.ndarray]):
        """Stack posterior samples into a pytree for log-density evaluation."""
        weights = jnp.asarray(samples["weights"])
        phi_log = jnp.asarray(samples["phi"])
        delta = jnp.asarray(samples["delta"])
        params: Dict[str, jnp.ndarray] = {
            "weights": weights,
            "phi": phi_log,
            "delta": delta,
        }

        latent_key = "weights_latent"
        if latent_key in samples:
            params[latent_key] = jnp.asarray(samples[latent_key])
        else:
            sqrt_phi = jnp.exp(0.5 * phi_log)
            if weights.ndim == 1:
                latent = self.penalty_unwhiten_T @ (weights * sqrt_phi)
            else:
                latent = (
                    weights * sqrt_phi[:, None]
                ) @ self.penalty_unwhiten_T.T
            params[latent_key] = latent

        return params
