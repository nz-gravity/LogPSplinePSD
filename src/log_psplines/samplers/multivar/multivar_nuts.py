"""
NUTS sampler for multivariate PSD estimation.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..base_sampler import SamplerConfig
from ..utils import build_log_density_fn, evaluate_log_density_batch
from .multivar_base import MultivarBaseSampler


@dataclass
class MultivarNUTSConfig(SamplerConfig):
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    save_nuts_diagnostics: bool = True


@jax.jit
def whittle_likelihood_arrays(
    y_re: jnp.ndarray,
    y_im: jnp.ndarray,
    Z_re: jnp.ndarray,
    Z_im: jnp.ndarray,
    log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
    theta_re: jnp.ndarray,  # (n_freq, n_theta)
    theta_im: jnp.ndarray,  # (n_freq, n_theta)
) -> jnp.ndarray:
    """Multivariate Whittle likelihood for Cholesky PSD parameterization - JIT version."""
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)

    if Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum("fij,fj->fi", Z_re, theta_re) - jnp.einsum(
            "fij,fj->fi", Z_im, theta_im
        )
        Z_theta_im = jnp.einsum("fij,fj->fi", Z_re, theta_im) + jnp.einsum(
            "fij,fj->fi", Z_im, theta_re
        )
        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    numerator = u_re**2 + u_im**2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2


def multivariate_psplines_model(
    y_re: jnp.ndarray,  # FFT real parts
    y_im: jnp.ndarray,  # FFT imaginary parts
    Z_re: jnp.ndarray,  # Design matrix real parts
    Z_im: jnp.ndarray,  # Design matrix imaginary parts
    all_bases,  # List of basis matrices
    all_penalties,  # List of penalty matrices
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
):
    """
    NumPyro model for multivariate PSD estimation using P-splines and Cholesky parameterization.
    """
    # Extract dimensions from input arrays (these are static during compilation)
    n_freq, n_dim = y_re.shape
    n_theta = Z_re.shape[2]

    component_idx = 0
    log_delta_components = []

    # Initialize total log prior accumulator
    log_prior_total = jnp.array(0.0)

    # Diagonal components (one per channel)
    for j in range(n_dim):  # Now n_dim is a concrete Python int
        delta_dist = dist.Gamma(alpha_delta, beta_delta)
        delta = numpyro.sample(f"delta_{j}", delta_dist)
        log_prior_total += delta_dist.log_prob(delta)

        phi_dist = dist.Gamma(alpha_phi, delta * beta_phi)
        phi = numpyro.sample(f"phi_delta_{j}", phi_dist)
        log_prior_total += phi_dist.log_prob(phi)

        k = all_penalties[component_idx].shape[0]
        base_normal = dist.Normal(0, 1).expand((k,)).to_event(1)
        weights = numpyro.sample(f"weights_delta_{j}", base_normal)

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        base_log_prob = base_normal.log_prob(weights)
        log_prior_adjustment = log_prior_w - base_log_prob
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_adjustment)
        log_prior_total += log_prior_adjustment

        # Compute log diagonal element
        log_delta_sq_j = all_bases[component_idx] @ weights
        log_delta_components.append(log_delta_sq_j)
        component_idx += 1

    log_delta_sq = jnp.stack(log_delta_components, axis=1)

    # Off-diagonal components (if multivariate)
    if n_theta > 0:
        # Real part of off-diagonal terms
        delta_re_dist = dist.Gamma(alpha_delta, beta_delta)
        delta_re = numpyro.sample("delta_theta_re", delta_re_dist)
        log_prior_total += delta_re_dist.log_prob(delta_re)

        phi_re_dist = dist.Gamma(alpha_phi, delta_re * beta_phi)
        phi_re = numpyro.sample("phi_theta_re", phi_re_dist)
        log_prior_total += phi_re_dist.log_prob(phi_re)

        k = all_penalties[component_idx].shape[0]
        base_normal_re = dist.Normal(0, 1).expand((k,)).to_event(1)
        weights_re = numpyro.sample("weights_theta_re", base_normal_re)

        wPw_re = jnp.dot(
            weights_re, jnp.dot(all_penalties[component_idx], weights_re)
        )
        log_prior_w_re = 0.5 * k * jnp.log(phi_re) - 0.5 * phi_re * wPw_re
        log_prior_adjustment_re = log_prior_w_re - base_normal_re.log_prob(
            weights_re
        )
        numpyro.factor("weights_prior_theta_re", log_prior_adjustment_re)
        log_prior_total += log_prior_adjustment_re

        theta_re_base = all_bases[component_idx] @ weights_re
        theta_re = jnp.tile(theta_re_base[:, None], (1, max(1, n_theta)))
        component_idx += 1

        # Imaginary part of off-diagonal terms
        delta_im_dist = dist.Gamma(alpha_delta, beta_delta)
        delta_im = numpyro.sample("delta_theta_im", delta_im_dist)
        log_prior_total += delta_im_dist.log_prob(delta_im)

        phi_im_dist = dist.Gamma(alpha_phi, delta_im * beta_phi)
        phi_im = numpyro.sample("phi_theta_im", phi_im_dist)
        log_prior_total += phi_im_dist.log_prob(phi_im)

        k = all_penalties[component_idx].shape[0]
        base_normal_im = dist.Normal(0, 1).expand((k,)).to_event(1)
        weights_im = numpyro.sample("weights_theta_im", base_normal_im)

        wPw_im = jnp.dot(
            weights_im, jnp.dot(all_penalties[component_idx], weights_im)
        )
        log_prior_w_im = 0.5 * k * jnp.log(phi_im) - 0.5 * phi_im * wPw_im
        log_prior_adjustment_im = log_prior_w_im - base_normal_im.log_prob(
            weights_im
        )
        numpyro.factor("weights_prior_theta_im", log_prior_adjustment_im)
        log_prior_total += log_prior_adjustment_im

        theta_im_base = all_bases[component_idx] @ weights_im
        theta_im = jnp.tile(theta_im_base[:, None], (1, max(1, n_theta)))
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

    # Likelihood using individual arrays
    log_likelihood = whittle_likelihood_arrays(
        y_re, y_im, Z_re, Z_im, log_delta_sq, theta_re, theta_im
    )
    numpyro.factor("likelihood", log_likelihood)

    # Store deterministic quantities for diagnostics
    numpyro.deterministic("lp", log_prior_total + log_likelihood)
    numpyro.deterministic("log_delta_sq", log_delta_sq)
    numpyro.deterministic("theta_re", theta_re)
    numpyro.deterministic("theta_im", theta_im)
    numpyro.deterministic("log_likelihood", log_likelihood)


class MultivarNUTSSampler(MultivarBaseSampler):
    """NUTS sampler for multivariate PSD estimation using Cholesky parameterization."""

    def __init__(
        self,
        fft_data,
        spline_model,
        config: MultivarNUTSConfig = None,
    ):
        if config is None:
            config = MultivarNUTSConfig()
        super().__init__(fft_data, spline_model, config)
        self.config: MultivarNUTSConfig = config

        # Build JIT log posterior for evidence calculations
        self._logpost_fn = build_log_density_fn(
            multivariate_psplines_model, self._logp_kwargs
        )

        # Pre-compile NumPyro model for faster warmup
        self._compile_model()

    @property
    def sampler_type(self) -> str:
        return "multivariate_nuts"

    def sample(
        self, n_samples: int, n_warmup: int = 500, **kwargs
    ) -> az.InferenceData:
        """Run multivariate NUTS sampling."""
        # Initialize starting values
        init_values = self._get_initial_values()
        init_strategy = init_to_value(values=init_values)

        # Setup NUTS kernel
        kernel = NUTS(
            multivariate_psplines_model,
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
            print(
                f"Multivariate NUTS sampler [{self.device}] - {self.n_channels} channels"
            )

        # FIXED: Run sampling with individual arrays instead of MultivarFFT object
        start_time = time.time()
        mcmc.run(
            self.rng_key,
            self.y_re,  # Individual arrays that JAX can handle
            self.y_im,
            self.Z_re,
            self.Z_im,
            # Remove these lines:
            # self.n_freq,
            # self.n_channels,
            self.all_bases,
            self.all_penalties,
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
        samples.pop("lp", None)
        stats = mcmc.get_extra_fields()

        # Move deterministic quantities from samples to stats
        for key in [
            "lp",
            "log_delta_sq",
            "theta_re",
            "theta_im",
            "log_likelihood",
        ]:
            if key in samples:
                stats[key] = samples.pop(key)

        params_batch = self._prepare_logpost_params(samples)
        stats["lp"] = evaluate_log_density_batch(
            self._logpost_fn, params_batch
        )

        return self.to_arviz(samples, stats)

    def _get_initial_values(self) -> Dict[str, jnp.ndarray]:
        """Get initial values for all parameters."""
        init_values = {}

        # Initialize diagonal components
        for j in range(self.n_channels):
            init_values[f"delta_{j}"] = (
                self.config.alpha_delta / self.config.beta_delta
            )
            init_values[f"phi_delta_{j}"] = (
                self.config.alpha_phi / self.config.beta_phi
            )
            init_values[f"weights_delta_{j}"] = (
                self.spline_model.diagonal_models[j].weights
            )

        # Initialize off-diagonal components if needed
        if self.n_theta > 0:
            init_values["delta_theta_re"] = (
                self.config.alpha_delta / self.config.beta_delta
            )
            init_values["phi_theta_re"] = (
                self.config.alpha_phi / self.config.beta_phi
            )
            init_values["weights_theta_re"] = (
                self.spline_model.offdiag_re_model.weights
            )

            init_values["delta_theta_im"] = (
                self.config.alpha_delta / self.config.beta_delta
            )
            init_values["phi_theta_im"] = (
                self.config.alpha_phi / self.config.beta_phi
            )
            init_values["weights_theta_im"] = (
                self.spline_model.offdiag_im_model.weights
            )

        return init_values

    def _compile_model(self) -> None:
        """Pre-compile the NumPyro model to speed up warmup."""
        try:
            from numpyro.infer.util import initialize_model

            if self.config.verbose:
                print("Pre-compiling NumPyro model...")

            # Initialize model with dummy data to trigger compilation
            init_params = initialize_model(
                self.rng_key,
                multivariate_psplines_model,
                model_kwargs=dict(
                    y_re=self.y_re,
                    y_im=self.y_im,
                    Z_re=self.Z_re,
                    Z_im=self.Z_im,
                    all_bases=self.all_bases,
                    all_penalties=self.all_penalties,
                    alpha_phi=self.config.alpha_phi,
                    beta_phi=self.config.beta_phi,
                    alpha_delta=self.config.alpha_delta,
                    beta_delta=self.config.beta_delta,
                ),
                init_strategy=init_to_value(values=self._get_initial_values()),
            )

            if self.config.verbose:
                print("Model pre-compilation completed")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Model pre-compilation failed: {e}")

    def _get_lnz(
        self, samples: Dict[str, Any], sample_stats: Dict[str, Any]
    ) -> tuple[float, float]:
        """Compute log normalizing constant for multivariate case."""
        # Use the base class implementation which handles the multivariate case
        return super()._get_lnz(samples, sample_stats)

    @property
    def _logp_kwargs(self) -> Dict[str, Any]:
        return dict(
            y_re=self.y_re,
            y_im=self.y_im,
            Z_re=self.Z_re,
            Z_im=self.Z_im,
            all_bases=self.all_bases,
            all_penalties=self.all_penalties,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _prepare_logpost_params(
        self, samples: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        return {
            name: jnp.asarray(array)
            for name, array in samples.items()
            if name.startswith(("weights_", "phi_", "delta_"))
        }

    def _compute_log_posterior(self, params: Dict[str, jnp.ndarray]) -> float:
        params = {k: jnp.asarray(v) for k, v in params.items()}
        return float(self._logpost_fn(params))
