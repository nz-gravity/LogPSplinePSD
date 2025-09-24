"""
Metropolis-Hastings sampler for univariate PSD estimation.
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple
import tempfile

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
import morphZ

from .univar_base import UnivarBaseSampler, log_likelihood  # Updated import
from ..base_sampler import SamplerConfig


@dataclass
class MetropolisHastingsConfig(SamplerConfig):
    target_accept_rate: float = 0.44  # Optimal for component-wise (d=1) updates
    adaptation_window: int = 50  # Adapt every N iterations
    adaptation_start: int = 100  # Start adapting after N iterations

    # Step size adaptation
    step_size_factor: float = 1.1  # Factor for step size adjustment
    min_step_size: float = 1e-6  # Minimum step size
    max_step_size: float = 10.0  # Maximum step size


class MetropolisHastingsSampler(UnivarBaseSampler):
    """
    Metropolis-Hastings sampler with adaptive step sizes for univariate PSD estimation.

    Uses component-wise updates for spline weights with Gibbs updates for smoothing parameters.
    """

    def __init__(
            self,
            periodogram,
            spline_model,
            config: MetropolisHastingsConfig = None,
    ):
        if config is None:
            config = MetropolisHastingsConfig()

        super().__init__(periodogram, spline_model, config)
        self.config: MetropolisHastingsConfig = config  # Type hint

        # MH-specific state initialization
        self._initialize_sampler_state()

        # Pre-compile JIT functions
        self._warmup_jit_functions()

    def _initialize_sampler_state(self):
        """Initialize sampler-specific state variables."""
        # Current parameter values
        self.current_weights = jnp.array(self.spline_model.weights)
        self.current_phi = self.config.alpha_phi / (
                self.config.beta_phi * self.config.alpha_delta / self.config.beta_delta
        )
        self.current_delta = self.config.alpha_delta / self.config.beta_delta

        # Step size adaptation
        self.step_sizes = jnp.ones(self.n_weights) * 0.1
        self.accept_counts = np.zeros(self.n_weights)
        self.proposal_counts = np.zeros(self.n_weights)
        self.iteration = 0

        # Initialize log posterior
        self.current_log_posterior = self._log_posterior(
            self.current_weights, self.current_phi, self.current_delta
        )

    def _warmup_jit_functions(self):
        """Warm up JIT functions to avoid compilation during sampling."""
        dummy_key = jax.random.PRNGKey(0)

        # Compile log posterior
        _ = log_posterior(
            self.current_weights,
            self.current_phi,
            self.current_delta,
            self.log_pdgrm,
            self.basis_matrix,
            self.log_parametric,
            self.penalty_matrix,
            self.config.alpha_phi,
            self.config.beta_phi,
            self.config.alpha_delta,
            self.config.beta_delta,
        )

        # Compile Gibbs updates
        _ = gibbs_update_phi(
            self.current_weights,
            self.penalty_matrix,
            self.current_delta,
            self.config.alpha_phi,
            self.config.beta_phi,
            dummy_key,
        )

    @property
    def sampler_type(self) -> str:
        """Required by base class."""
        return "metropolis_hastings"

    def _log_posterior(self, weights: jnp.ndarray, phi: float, delta: float) -> float:
        """Wrapper for JIT-compiled log posterior."""
        return log_posterior(
            weights,
            phi,
            delta,
            self.log_pdgrm,
            self.basis_matrix,
            self.log_parametric,
            self.penalty_matrix,
            self.config.alpha_phi,
            self.config.beta_phi,
            self.config.alpha_delta,
            self.config.beta_delta,
        )

    def _update_weights_componentwise(self) -> Tuple[int, jnp.ndarray]:
        """Update weights using JIT-compiled component-wise Metropolis-Hastings."""
        self.rng_key, subkey = jax.random.split(self.rng_key)

        new_weights, acceptance_mask, new_log_posterior = update_weights_componentwise(
            self.current_weights,
            self.step_sizes,
            self.current_phi,
            self.current_delta,
            self.log_pdgrm,
            self.basis_matrix,
            self.log_parametric,
            self.penalty_matrix,
            subkey,
            self.n_weights,
            self.config.alpha_phi,
            self.config.beta_phi,
            self.config.alpha_delta,
            self.config.beta_delta,
        )

        self.current_weights = new_weights
        self.current_log_posterior = new_log_posterior

        accepts_np = np.array(acceptance_mask)
        self.accept_counts += accepts_np
        self.proposal_counts += 1

        return int(np.sum(accepts_np)), new_weights

    def _update_phi(self) -> float:
        """JIT-compiled Gibbs update for phi."""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.current_phi = gibbs_update_phi(
            self.current_weights,
            self.penalty_matrix,
            self.current_delta,
            self.config.alpha_phi,
            self.config.beta_phi,
            subkey,
        )
        return self.current_phi

    def _update_delta(self) -> float:
        """JIT-compiled Gibbs update for delta."""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.current_delta = gibbs_update_delta(
            self.current_phi,
            self.config.alpha_phi,
            self.config.beta_phi,
            self.config.alpha_delta,
            self.config.beta_delta,
            subkey,
        )
        return self.current_delta

    def _adapt_step_sizes(self):
        """Adapt individual step sizes based on acceptance rates."""
        if self.iteration < self.config.adaptation_start:
            return

        if self.iteration % self.config.adaptation_window == 0:
            for i in range(self.n_weights):
                if self.proposal_counts[i] > 0:
                    accept_rate = self.accept_counts[i] / self.proposal_counts[i]

                    if accept_rate < self.config.target_accept_rate:
                        self.step_sizes = self.step_sizes.at[i].multiply(
                            1 / self.config.step_size_factor
                        )
                    else:
                        self.step_sizes = self.step_sizes.at[i].multiply(
                            self.config.step_size_factor
                        )

                    self.step_sizes = self.step_sizes.at[i].set(
                        jnp.clip(
                            self.step_sizes[i],
                            self.config.min_step_size,
                            self.config.max_step_size,
                        )
                    )

            self.accept_counts.fill(0)
            self.proposal_counts.fill(0)

    def step(self) -> Dict[str, Any]:
        """Perform one MCMC step."""
        self.iteration += 1

        n_accepted_weights, new_weights = self._update_weights_componentwise()
        self._update_phi()
        self._update_delta()

        self.current_log_posterior = self._log_posterior(
            self.current_weights, self.current_phi, self.current_delta
        )

        self._adapt_step_sizes()

        return {
            "weights": np.array(self.current_weights),
            "phi": float(self.current_phi),
            "delta": float(self.current_delta),
            "log_posterior": float(self.current_log_posterior),
            "n_accepted_weights": n_accepted_weights,
            "accept_prob": n_accepted_weights / self.n_weights,
            "step_sizes": np.array(self.step_sizes),
        }

    def sample(self, n_samples: int, n_warmup: int = 500, **kwargs) -> az.InferenceData:
        """Run Metropolis-Hastings sampling."""
        total_iterations = n_warmup + n_samples

        # Initialize storage for samples and statistics
        samples = {
            "weights": np.empty((n_samples, self.n_weights), dtype=np.float32),
            "phi": np.empty(n_samples, dtype=np.float32),
            "delta": np.empty(n_samples, dtype=np.float32),
        }

        sample_stats = {
            "accept_prob": np.empty(n_samples, dtype=np.float32),
            "lp": np.empty(n_samples, dtype=np.float32),
            "step_size_mean": np.empty(n_samples, dtype=np.float32),
            "step_size_std": np.empty(n_samples, dtype=np.float32),
        }

        # Warmup JIT compilation
        self.step()

        start_time = time.time()

        if self.config.verbose:
            print(
                f"Metropolis-Hastings with adaptive step sizes [{self.device}] {self.rng_key}"
            )

        with tqdm(
                total=total_iterations,
                disable=not self.config.verbose,
                desc="MH",
                leave=True,
        ) as pbar:

            for i in range(total_iterations):
                step_info = self.step()

                # Store samples after warmup
                if i >= n_warmup:
                    j = i - n_warmup
                    samples["weights"][j] = step_info["weights"]
                    samples["phi"][j] = step_info["phi"]
                    samples["delta"][j] = step_info["delta"]

                    # Compute log posterior for diagnostics
                    lp = float(
                        log_posterior(
                            jnp.array(step_info["weights"]),
                            step_info["phi"],
                            step_info["delta"],
                            self.log_pdgrm,
                            self.basis_matrix,
                            self.log_parametric,
                            self.penalty_matrix,
                            self.config.alpha_phi,
                            self.config.beta_phi,
                            self.config.alpha_delta,
                            self.config.beta_delta,
                        )
                    )

                    sample_stats["accept_prob"][j] = step_info["accept_prob"]
                    sample_stats["lp"][j] = lp
                    sample_stats["step_size_mean"][j] = np.mean(step_info["step_sizes"])
                    sample_stats["step_size_std"][j] = np.std(step_info["step_sizes"])

                # Update progress bar
                if i % 100 == 0:
                    phase = "Warmup" if i < n_warmup else "Sampling"
                    desc = (
                        f"{phase} | Accept: {step_info['accept_prob']:.3f} | "
                        f"LogPost: {step_info['log_posterior']:.1f} | "
                        f"StepSize: {np.mean(step_info['step_sizes']):.4f}"
                    )
                    pbar.set_description(desc)

                pbar.update(1)

        self.runtime = time.time() - start_time

        if self.config.verbose:
            final_accept = 0
            if len(sample_stats["accept_prob"]) > 50:
                final_accept = np.mean(sample_stats["accept_prob"][-50:])

            print(f"\nSampling completed in {self.runtime:.2f} seconds")
            print(
                f"Final acceptance rate: {final_accept:.3f} "
                f"(target: {self.config.target_accept_rate})"
            )

        return self.to_arviz(samples, sample_stats)

    def _get_lnz(
            self,
            samples: Dict[str, np.ndarray],
            sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Compute log normalizing constant using morphZ."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        post_smp = np.concatenate([
            samples['weights'],
            samples['phi'][:, None],
            samples['delta'][:, None]
        ], axis=1)
        lp = sample_stats['lp']

        def lp_fn(sample):
            weights = sample[:self.n_weights]
            phi = sample[self.n_weights]
            delta = sample[self.n_weights + 1]
            return log_posterior(weights, phi, delta, **self._logp_kwargs)

        lnz_res = morphZ.evidence(
            post_smp, lp, lp_fn, output_path=tempfile.gettempdir()
        )[0]
        return float(lnz_res[0]), float(lnz_res[1])

    @property
    def _logp_kwargs(self) -> Dict[str, Any]:
        """Arguments for log posterior computation."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            basis_matrix=self.basis_matrix,
            log_parametric=self.log_parametric,
            penalty_matrix=self.penalty_matrix,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _compute_log_posterior(self, weights: jnp.ndarray, phi: float, delta: float) -> float:
        """Compute log posterior for given parameters (used by base class)."""
        return log_posterior(
            weights, phi, delta, **self._logp_kwargs
        )


# ==================== JAX OPTIMIZED FUNCTIONS ====================

@jax.jit
def log_prior_weights(
        weights: jnp.ndarray, phi: float, penalty_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Log prior for weights: MVN(0, (phi * P)^-1)."""
    precision = phi * penalty_matrix
    quad_form = weights.T @ precision @ weights
    k = len(weights)
    log_det_term = 0.5 * k * jnp.log(phi)
    return log_det_term - 0.5 * quad_form


@jax.jit
def log_prior_phi(
        phi: float, delta: float, alpha_phi: float, beta_phi: float
) -> jnp.ndarray:
    """Log prior for phi: Gamma(alpha_phi, delta * beta_phi)."""
    return jnp.where(
        phi > 0,
        (alpha_phi - 1) * jnp.log(phi)
        - delta * beta_phi * phi
        - jax.scipy.special.gammaln(alpha_phi)
        + alpha_phi * jnp.log(delta * beta_phi),
        -jnp.inf,
    )


@jax.jit
def log_prior_delta(
        delta: float, alpha_delta: float, beta_delta: float
) -> jnp.ndarray:
    """Log prior for delta: Gamma(alpha_delta, beta_delta)."""
    return jnp.where(
        delta > 0,
        (alpha_delta - 1) * jnp.log(delta)
        - beta_delta * delta
        - jax.scipy.special.gammaln(alpha_delta)
        + alpha_delta * jnp.log(beta_delta),
        -jnp.inf,
    )


@jax.jit
def log_posterior(
        weights: jnp.ndarray,
        phi: float,
        delta: float,
        log_pdgrm: jnp.ndarray,
        basis_matrix: jnp.ndarray,
        log_parametric: jnp.ndarray,
        penalty_matrix: jnp.ndarray,
        alpha_phi: float,
        beta_phi: float,
        alpha_delta: float,
        beta_delta: float,
) -> float:
    """Complete log posterior computation."""
    log_like = log_likelihood(weights, log_pdgrm, basis_matrix, log_parametric)
    log_prior_w = log_prior_weights(weights, phi, penalty_matrix)
    log_prior_phi_val = log_prior_phi(phi, delta, alpha_phi, beta_phi)
    log_prior_delta_val = log_prior_delta(delta, alpha_delta, beta_delta)

    total = log_like + log_prior_w + log_prior_phi_val + log_prior_delta_val
    return jnp.where(jnp.isfinite(total), total, -jnp.inf)


@jax.jit
def gibbs_update_phi(
        weights: jnp.ndarray,
        penalty_matrix: jnp.ndarray,
        current_delta: float,
        alpha_phi: float,
        beta_phi: float,
        rng_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Gibbs update for phi parameter."""
    k = len(weights)
    quad_form = weights.T @ penalty_matrix @ weights

    shape = alpha_phi + k / 2
    rate = beta_phi * current_delta + 0.5 * quad_form

    return jax.random.gamma(rng_key, shape) / rate


@jax.jit
def gibbs_update_delta(
        current_phi: float,
        alpha_phi: float,
        beta_phi: float,
        alpha_delta: float,
        beta_delta: float,
        rng_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Gibbs update for delta parameter."""
    shape = alpha_phi + alpha_delta
    rate = beta_phi * current_phi + beta_delta

    return jax.random.gamma(rng_key, shape) / rate


@partial(jax.jit, static_argnums=(9,))
def update_weights_componentwise(
        weights: jnp.ndarray,
        step_sizes: jnp.ndarray,
        phi: float,
        delta: float,
        log_pdgrm: jnp.ndarray,
        basis_matrix: jnp.ndarray,
        log_parametric: jnp.ndarray,
        penalty_matrix: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        n_weights: int,
        alpha_phi: float,
        beta_phi: float,
        alpha_delta: float,
        beta_delta: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Component-wise Metropolis-Hastings update for weights."""
    current_log_post = log_posterior(
        weights,
        phi,
        delta,
        log_pdgrm,
        basis_matrix,
        log_parametric,
        penalty_matrix,
        alpha_phi,
        beta_phi,
        alpha_delta,
        beta_delta,
    )

    perm_key, noise_key, accept_key = jax.random.split(rng_key, 3)
    indices = jax.random.permutation(perm_key, n_weights)

    noise_keys = jax.random.split(noise_key, n_weights)
    noises = jax.vmap(jax.random.normal)(noise_keys) * step_sizes

    def update_single_component(i, carry):
        weights_current, accepts = carry
        idx = indices[i]

        proposal_weights = weights_current.at[idx].add(noises[idx])

        proposal_log_post = log_posterior(
            proposal_weights,
            phi,
            delta,
            log_pdgrm,
            basis_matrix,
            log_parametric,
            penalty_matrix,
            alpha_phi,
            beta_phi,
            alpha_delta,
            beta_delta,
        )

        log_alpha = proposal_log_post - current_log_post
        alpha = jnp.minimum(1.0, jnp.exp(log_alpha))

        u_key = jax.random.fold_in(accept_key, i)
        u = jax.random.uniform(u_key)
        accept = u < alpha

        new_weights = jnp.where(accept, proposal_weights, weights_current)
        new_accepts = accepts.at[idx].set(accept)

        return new_weights, new_accepts

    accepts = jnp.zeros(n_weights, dtype=bool)
    final_weights, final_accepts = jax.lax.fori_loop(
        0, n_weights, update_single_component, (weights, accepts)
    )

    final_log_post = log_posterior(
        final_weights,
        phi,
        delta,
        log_pdgrm,
        basis_matrix,
        log_parametric,
        penalty_matrix,
        alpha_phi,
        beta_phi,
        alpha_delta,
        beta_delta,
    )

    return final_weights, final_accepts, final_log_post
