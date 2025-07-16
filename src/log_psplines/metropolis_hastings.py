"""
Adaptive MCMC sampler for log P-splines spectral density estimation.
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from .bayesian_model import build_spline, whittle_lnlike
from .psplines import LogPSplines


@dataclass
class MetropolisHastingsConfig:
    target_accept_rate: float = (
        0.44  # Optimal for component-wise (d=1) updates
    )
    adaptation_window: int = 50  # Adapt every N iterations
    adaptation_start: int = 100  # Start adapting after N iterations

    # Step size adaptation
    step_size_factor: float = 1.1  # Factor for step size adjustment
    min_step_size: float = 1e-6  # Minimum step size
    max_step_size: float = 10.0  # Maximum step size


# ==================== JAX OPTIMIZED FUNCTIONS ====================


@jax.jit
def log_prior_weights(
    weights: jnp.ndarray, phi: float, penalty_matrix: jnp.ndarray
) -> float:
    """log prior for weights: MVN(0, (phi * P)^-1)."""
    precision = phi * penalty_matrix
    quad_form = weights.T @ precision @ weights
    k = len(weights)
    log_det_term = 0.5 * k * jnp.log(phi)
    return log_det_term - 0.5 * quad_form


@jax.jit
def log_prior_phi(
    phi: float, delta: float, alpha_phi: float, beta_phi: float
) -> float:
    """log prior for phi: Gamma(alpha_phi, delta * beta_phi)."""
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
) -> float:
    """log prior for delta: Gamma(alpha_delta, beta_delta)."""
    return jnp.where(
        delta > 0,
        (alpha_delta - 1) * jnp.log(delta)
        - beta_delta * delta
        - jax.scipy.special.gammaln(alpha_delta)
        + alpha_delta * jnp.log(beta_delta),
        -jnp.inf,
    )


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> float:
    ln_spline = build_spline(basis_matrix, weights)
    ln_model = ln_spline + log_parametric
    return whittle_lnlike(log_pdgrm, ln_model)


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
    log_like = log_likelihood(weights, log_pdgrm, basis_matrix, log_parametric)
    log_prior_w = log_prior_weights(weights, phi, penalty_matrix)
    log_prior_p = log_prior_phi(phi, delta, alpha_phi, beta_phi)
    log_prior_d = log_prior_delta(delta, alpha_delta, beta_delta)

    total = log_like + log_prior_w + log_prior_p + log_prior_d
    return jnp.where(jnp.isfinite(total), total, -jnp.inf)


@jax.jit
def compute_acceptance_prob(
    current_log_posterior: float, proposal_log_posterior: float
) -> float:
    log_alpha = proposal_log_posterior - current_log_posterior
    return jnp.minimum(1.0, jnp.exp(log_alpha))


@jax.jit
def gibbs_update_phi(
    weights: jnp.ndarray,
    penalty_matrix: jnp.ndarray,
    current_delta: float,
    alpha_phi: float,
    beta_phi: float,
    rng_key: jax.random.PRNGKey,
) -> float:
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
) -> float:
    shape = alpha_phi + alpha_delta
    rate = beta_phi * current_phi + beta_delta

    return jax.random.gamma(rng_key, shape) / rate


@partial(jax.jit, static_argnums=(9,))  # n_weights is static
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
    """
    JIT-compiled component-wise weight updates.
    Returns: (new_weights, acceptance_mask, current_log_posterior)
    """
    # Current log posterior
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

    # Generate random permutation for update order
    perm_key, noise_key, accept_key = jax.random.split(rng_key, 3)
    indices = jax.random.permutation(perm_key, n_weights)

    # Generate noise for all components
    noise_keys = jax.random.split(noise_key, n_weights)
    noises = jax.vmap(jax.random.normal)(noise_keys) * step_sizes

    def update_single_component(i, carry):
        weights_current, accepts = carry
        idx = indices[i]

        # Create proposal
        proposal_weights = weights_current.at[idx].add(noises[idx])

        # Compute acceptance probability
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

        alpha = compute_acceptance_prob(current_log_post, proposal_log_post)

        # Accept/reject using uniform random
        u_key = jax.random.fold_in(accept_key, i)
        u = jax.random.uniform(u_key)
        accept = u < alpha

        # Update weights and acceptance record
        new_weights = jnp.where(accept, proposal_weights, weights_current)
        new_accepts = accepts.at[idx].set(accept)

        return new_weights, new_accepts

    # Initialize
    accepts = jnp.zeros(n_weights, dtype=bool)

    # Sequential updates
    final_weights, final_accepts = jax.lax.fori_loop(
        0, n_weights, update_single_component, (weights, accepts)
    )

    # Compute final log posterior
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


class MetropolisHastingsSampler:

    def __init__(
        self,
        log_pdgrm: jnp.ndarray,
        spline_model: LogPSplines,
        config: Optional[MetropolisHastingsConfig] = None,
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
        rng_key: int = 42,
    ):
        """
        Initialize the sampler.

        Parameters
        ----------
        log_pdgrm : jnp.ndarray
            Log periodogram data
        spline_model : LogPSplines
            Spline model object
        config : AdaptiveMCMCConfig, optional
            Configuration for adaptation
        alpha_phi, beta_phi : float
            Hyperparameters for phi prior
        alpha_delta, beta_delta : float
            Hyperparameters for delta prior
        rng_key : int
            Random seed for JAX
        """
        self.log_pdgrm = log_pdgrm
        self.spline_model = spline_model
        self.config = config or MetropolisHastingsConfig()

        # Prior hyperparameters
        self.alpha_phi = alpha_phi
        self.beta_phi = beta_phi
        self.alpha_delta = alpha_delta
        self.beta_delta = beta_delta

        # JAX random state
        self.rng_key = jax.random.PRNGKey(rng_key)

        # Current state (convert to JAX arrays)
        self.current_weights = jnp.array(spline_model.weights)
        self.current_phi = alpha_phi / (beta_phi * alpha_delta / beta_delta)
        self.current_delta = alpha_delta / beta_delta

        # Step size adaptation tracking
        self.n_weights = len(self.current_weights)
        self.step_sizes = jnp.ones(self.n_weights) * 0.1  # JAX array
        self.accept_counts = np.zeros(
            self.n_weights
        )  # NumPy for adaptation logic
        self.proposal_counts = np.zeros(self.n_weights)
        self.iteration = 0

        # Precompute static arrays for JIT (convert to JAX)
        self.penalty_matrix = jnp.array(spline_model.penalty_matrix)
        self.basis_matrix = jnp.array(spline_model.basis)
        self.log_parametric = jnp.array(spline_model.log_parametric_model)

        # Initialize log posterior
        self.current_log_posterior = self._log_posterior(
            self.current_weights, self.current_phi, self.current_delta
        )

        # Pre-compile JIT functions
        self._warmup_jit_functions()

    def _warmup_jit_functions(self):
        """Warm up JIT functions to avoid compilation time during sampling."""
        dummy_key = jax.random.PRNGKey(0)
        _ = log_posterior(
            self.current_weights,
            self.current_phi,
            self.current_delta,
            self.log_pdgrm,
            self.basis_matrix,
            self.log_parametric,
            self.penalty_matrix,
            self.alpha_phi,
            self.beta_phi,
            self.alpha_delta,
            self.beta_delta,
        )
        _ = gibbs_update_phi(
            self.current_weights,
            self.penalty_matrix,
            self.current_delta,
            self.alpha_phi,
            self.beta_phi,
            dummy_key,
        )
        _ = gibbs_update_delta(
            self.current_phi,
            self.alpha_phi,
            self.beta_phi,
            self.alpha_delta,
            self.beta_delta,
            dummy_key,
        )

    def _log_posterior(
        self, weights: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """Wrapper for log posterior."""
        return log_posterior(
            weights,
            phi,
            delta,
            self.log_pdgrm,
            self.basis_matrix,
            self.log_parametric,
            self.penalty_matrix,
            self.alpha_phi,
            self.beta_phi,
            self.alpha_delta,
            self.beta_delta,
        )

    def _update_weights_componentwise(self) -> Tuple[int, jnp.ndarray]:
        """Update weights using JIT-compiled component-wise Metropolis-Hastings."""
        self.rng_key, subkey = jax.random.split(self.rng_key)

        new_weights, acceptance_mask, new_log_posterior = (
            update_weights_componentwise(
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
                self.alpha_phi,
                self.beta_phi,
                self.alpha_delta,
                self.beta_delta,
            )
        )

        # Update state
        self.current_weights = new_weights
        self.current_log_posterior = new_log_posterior

        # Update acceptance tracking (convert back to numpy for adaptation logic)
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
            self.alpha_phi,
            self.beta_phi,
            subkey,
        )
        return self.current_phi

    def _update_delta(self) -> float:
        """JIT-compiled Gibbs update for delta."""
        self.rng_key, subkey = jax.random.split(self.rng_key)

        self.current_delta = gibbs_update_delta(
            self.current_phi,
            self.alpha_phi,
            self.beta_phi,
            self.alpha_delta,
            self.beta_delta,
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
                    accept_rate = (
                        self.accept_counts[i] / self.proposal_counts[i]
                    )

                    if accept_rate < self.config.target_accept_rate:
                        self.step_sizes = self.step_sizes.at[i].multiply(
                            1 / self.config.step_size_factor
                        )
                    else:
                        self.step_sizes = self.step_sizes.at[i].multiply(
                            self.config.step_size_factor
                        )

                    # Clip to bounds
                    self.step_sizes = self.step_sizes.at[i].set(
                        jnp.clip(
                            self.step_sizes[i],
                            self.config.min_step_size,
                            self.config.max_step_size,
                        )
                    )

            # Reset counters
            self.accept_counts.fill(0)
            self.proposal_counts.fill(0)

    def step(self) -> Dict[str, Any]:
        """
        Perform one MCMC step with adaptive step sizes.

        Updates:
        1. Component-wise Metropolis-Hastings for weights
        2. Gibbs sampling for phi and delta
        3. Adaptive step size adjustment
        """
        self.iteration += 1

        # 1. Update weights (component-wise)
        n_accepted_weights, new_weights = self._update_weights_componentwise()

        # 2. Update phi and delta (Gibbs)
        self._update_phi()
        self._update_delta()

        # 3. Update log posterior with new phi, delta
        self.current_log_posterior = self._log_posterior(
            self.current_weights, self.current_phi, self.current_delta
        )

        # 4. Adaptive step size adjustment
        self._adapt_step_sizes()

        # Return step info (convert JAX arrays to numpy for external use)
        return {
            "weights": np.array(self.current_weights),
            "phi": float(self.current_phi),
            "delta": float(self.current_delta),
            "log_posterior": float(self.current_log_posterior),
            "n_accepted_weights": n_accepted_weights,
            "acceptance_rate": n_accepted_weights / self.n_weights,
            "step_sizes": np.array(self.step_sizes),
        }

    def sample(
        self,
        n_samples: int,
        n_warmup: int = 500,
        thin: int = 1,
        verbose: bool = True,
    ) -> az.InferenceData:
        """
        Run sampler

        Parameters
        ----------
        n_samples : int
            Number of samples to collect
        n_warmup : int
            Number of warmup iterations
        thin : int
            Thinning interval
        verbose : bool
            Whether to print progress

        Returns
        -------
        dict
            Dictionary containing samples and diagnostics
        """
        total_iterations = n_warmup + n_samples * thin

        all_samples = {"weights": [], "phi": [], "delta": []}

        # Sample statistics storage
        sample_stats = {
            "acceptance_rate": [],
            "log_likelihood": [],
            "log_prior": [],
            "step_size_mean": [],
            "step_size_std": [],
        }

        diagnostics = {"step_sizes_history": [], "iteration_info": []}

        # step once to initialize JIT compilation
        self.step()  # Warm up JIT compilation

        start_time = time.time()

        # TQDM progress bar
        with tqdm(
            total=total_iterations, disable=not verbose, desc="MH ", leave=True
        ) as pbar:

            for i in range(total_iterations):
                step_info = self.step()

                # Store samples after warmup
                if i >= n_warmup and (i - n_warmup) % thin == 0:
                    all_samples["weights"].append(step_info["weights"])
                    all_samples["phi"].append(step_info["phi"])
                    all_samples["delta"].append(step_info["delta"])

                    # Compute sample statistics using JIT functions
                    log_like = float(
                        log_likelihood(
                            jnp.array(step_info["weights"]),
                            self.log_pdgrm,
                            self.basis_matrix,
                            self.log_parametric,
                        )
                    )
                    log_prior = float(
                        log_prior_weights(
                            jnp.array(step_info["weights"]),
                            step_info["phi"],
                            self.penalty_matrix,
                        )
                        + log_prior_phi(
                            step_info["phi"],
                            step_info["delta"],
                            self.alpha_phi,
                            self.beta_phi,
                        )
                        + log_prior_delta(
                            step_info["delta"],
                            self.alpha_delta,
                            self.beta_delta,
                        )
                    )

                    sample_stats["acceptance_rate"].append(
                        step_info["acceptance_rate"]
                    )
                    sample_stats["log_likelihood"].append(log_like)
                    sample_stats["log_prior"].append(log_prior)
                    sample_stats["step_size_mean"].append(
                        np.mean(step_info["step_sizes"])
                    )
                    sample_stats["step_size_std"].append(
                        np.std(step_info["step_sizes"])
                    )

                if verbose:
                    # Update progress bar with diagnostics
                    if i % 100 == 0:
                        phase = "Warmup" if i < n_warmup else "Sampling"
                        desc = (
                            f"{phase} | Accept: {step_info['acceptance_rate']:.3f} | "
                            f"LogPost: {step_info['log_posterior']:.1f} | "
                            f"StepSize: {np.mean(step_info['step_sizes']):.4f}"
                        )
                        pbar.set_description(desc)

                    # Store detailed diagnostics less frequently
                    if i % 500 == 0:
                        diagnostics["step_sizes_history"].append(
                            step_info["step_sizes"]
                        )
                        diagnostics["iteration_info"].append(
                            {
                                "iteration": i,
                                "acceptance_rate": step_info[
                                    "acceptance_rate"
                                ],
                                "log_posterior": step_info["log_posterior"],
                                "phase": (
                                    "warmup" if i < n_warmup else "sampling"
                                ),
                            }
                        )

                    pbar.update(1)

        runtime = time.time() - start_time

        if verbose:
            final_accept = (
                np.mean(sample_stats["acceptance_rate"][-50:])
                if sample_stats["acceptance_rate"]
                else 0
            )
            print(f"\nSampling completed in {runtime:.2f} seconds")
            print(
                f"Final acceptance rate: {final_accept:.3f} (target: {self.config.target_accept_rate:.3f})"
            )

        # Convert to arrays and reshape for arviz (chains, draws, ...)
        for key in all_samples:
            all_samples[key] = np.array(all_samples[key])
            # Reshape to (chains, draws, ...) - single chain for now
            if all_samples[key].ndim == 2:  # weights are 2D
                all_samples[key] = all_samples[key][
                    None, ...
                ]  # (1, draws, n_weights)
            else:  # phi, delta are 1D
                all_samples[key] = all_samples[key][None, :]  # (1, draws)

        for key in sample_stats:
            sample_stats[key] = np.array(sample_stats[key])[
                None, :
            ]  # (1, draws)

        return self.to_arviz(
            {
                "samples": all_samples,
                "sample_stats": sample_stats,
                "diagnostics": diagnostics,
                "runtime": runtime,
                "config": self.config,
                "spline_model": self.spline_model,
                "n_warmup": n_warmup,
                "n_samples": n_samples,
            }
        )

    def to_arviz(self, results: Dict[str, Any]) -> az.InferenceData:
        """
        Convert MCMC results to arviz InferenceData format.

        Parameters
        ----------
        results : dict
            Results from self.sample()

        Returns
        -------
        az.InferenceData
            ArviZ inference data object
        """
        samples = results["samples"]
        sample_stats = results["sample_stats"]

        # Create coordinate system
        coords = {
            "chain": range(1),
            "draw": range(results["n_samples"]),
            "weight_dim": range(self.n_weights),
            "freq_dim": range(len(self.log_pdgrm)),
        }

        posterior_data = {
            "phi": samples["phi"],
            "delta": samples["delta"],
            "weights": samples["weights"],
        }

        posterior_dims = {
            "phi": ["chain", "draw"],
            "delta": ["chain", "draw"],
            "weights": ["chain", "draw", "weight_dim"],
        }

        # Sample statistics
        sample_stats_data = {
            "acceptance_rate": sample_stats["acceptance_rate"],
            "lp": sample_stats["log_likelihood"]
            + sample_stats["log_prior"],  # log posterior
            "log_likelihood": sample_stats["log_likelihood"],
            "log_prior": sample_stats["log_prior"],
            "step_size_mean": sample_stats["step_size_mean"],
            "step_size_std": sample_stats["step_size_std"],
        }

        sample_stats_dims = {k: ["chain", "draw"] for k in sample_stats_data}

        observed_data = {"log_periodogram": self.log_pdgrm}

        observed_dims = {"log_periodogram": ["freq_dim"]}

        # Create InferenceData object
        idata = az.from_dict(
            posterior=posterior_data,
            sample_stats=sample_stats_data,
            observed_data=observed_data,
            dims={**posterior_dims, **sample_stats_dims, **observed_dims},
            coords=coords,
        )

        # Add metadata
        idata.attrs["sampling_method"] = "metropolis_hastings"
        idata.attrs["runtime"] = results["runtime"]
        idata.attrs["n_warmup"] = results["n_warmup"]
        idata.attrs["target_accept_rate"] = self.config.target_accept_rate
        idata.attrs["adaptation_window"] = self.config.adaptation_window
        idata.attrs["n_weights"] = self.n_weights
        idata.attrs["spline_degree"] = self.spline_model.degree
        idata.attrs["n_knots"] = self.spline_model.n_knots

        return idata
