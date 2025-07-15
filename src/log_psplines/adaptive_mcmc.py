"""
Adaptive MCMC sampler for log P-splines spectral density estimation.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import gamma
from tqdm.auto import tqdm

from .bayesian_model import build_spline, whittle_lnlike
from .datatypes import Periodogram
from .psplines import LogPSplines


@dataclass
class AdaptiveMCMCConfig:
    """Configuration for adaptive MCMC sampler."""

    # Adaptation parameters
    target_accept_rate: float = (
        0.44  # Optimal for multivariate-gibbs (see https://probability.ca/jeff/ftpdir/handbookart.pdf)
    )
    adaptation_window: int = (
        50  # Adapt every N iterations (suggested by Roberts and Rosenthal, 2009) to adapt in batches
    )
    adaptation_start: int = 100  # Start adapting after N iterations

    # Step size adaptation
    step_size_factor: float = 1.1  # Factor for step size adjustment
    min_step_size: float = 1e-6  # Minimum step size
    max_step_size: float = 10.0  # Maximum step size

    # Covariance adaptation
    adapt_covariance: bool = False  # Whether to adapt proposal covariance
    covariance_regularization: float = 1e-6  # Regularization for covariance
    memory_length: int = 1000  # Number of samples to keep for covariance


class AdaptiveMCMCSampler:
    """
    Adaptive MCMC sampler for log P-splines using component-wise Metropolis-Hastings.

    This sampler updates:
    - weights (lambda): Component-wise adaptive Metropolis
    - phi: Gibbs sampling from full conditional
    - delta: Gibbs sampling from full conditional
    """

    def __init__(
        self,
        log_pdgrm: jnp.ndarray,
        spline_model: LogPSplines,
        config: Optional[AdaptiveMCMCConfig] = None,
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
    ):
        """
        Initialize the adaptive MCMC sampler.

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
        """
        self.log_pdgrm = log_pdgrm
        self.spline_model = spline_model
        self.config = config or AdaptiveMCMCConfig()

        # Prior hyperparameters
        self.alpha_phi = alpha_phi
        self.beta_phi = beta_phi
        self.alpha_delta = alpha_delta
        self.beta_delta = beta_delta

        # Current state
        self.current_weights = spline_model.weights.copy()
        self.current_phi = alpha_phi / (beta_phi * alpha_delta / beta_delta)
        self.current_delta = alpha_delta / beta_delta
        self.current_log_posterior = -np.inf

        # Adaptation tracking
        self.n_weights = len(self.current_weights)
        self.step_sizes = (
            np.ones(self.n_weights) * 0.1
        )  # Individual step sizes
        self.accept_counts = np.zeros(self.n_weights)
        self.proposal_counts = np.zeros(self.n_weights)
        self.iteration = 0

        # Covariance adaptation
        if self.config.adapt_covariance:
            self.sample_history = []
            self.empirical_mean = np.zeros(self.n_weights)
            self.empirical_cov = np.eye(self.n_weights) * 0.1

        # Precompute for efficiency
        self.penalty_matrix = spline_model.penalty_matrix
        self.basis_matrix = spline_model.basis
        self.log_parametric = spline_model.log_parametric_model

        # Initialize log posterior
        self._update_log_posterior()

    def _update_log_posterior(self):
        """Update current log posterior."""
        self.current_log_posterior = self._log_posterior(
            self.current_weights, self.current_phi, self.current_delta
        )

    def _log_prior_weights(self, weights: jnp.ndarray, phi: float) -> float:
        """Log prior for weights: MVN(0, (phi * P)^-1)."""
        try:
            precision = phi * self.penalty_matrix
            # Use the quadratic form directly (more stable) from bilby
            quad_form = weights.T @ precision @ weights
            log_det_term = 0.5 * self.n_weights * jnp.log(phi)
            return log_det_term - 0.5 * quad_form
        except:
            return -jnp.inf

    def _log_prior_phi(self, phi: float, delta: float) -> float:
        """Log prior for phi: Gamma(alpha_phi, delta * beta_phi)."""
        if phi <= 0:
            return -jnp.inf
        return gamma.logpdf(
            phi, a=self.alpha_phi, scale=1 / (delta * self.beta_phi)
        )

    def _log_prior_delta(self, delta: float) -> float:
        """Log prior for delta: Gamma(alpha_delta, beta_delta)."""
        if delta <= 0:
            return -jnp.inf
        return gamma.logpdf(
            delta, a=self.alpha_delta, scale=1 / self.beta_delta
        )

    def _log_likelihood(self, weights: jnp.ndarray) -> float:
        """Whittle log-likelihood."""
        ln_spline = build_spline(self.basis_matrix, weights)
        ln_model = ln_spline + self.log_parametric
        return whittle_lnlike(self.log_pdgrm, ln_model)

    def _log_posterior(
        self, weights: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """Log posterior density."""
        log_like = self._log_likelihood(weights)
        log_prior_w = self._log_prior_weights(weights, phi)
        log_prior_phi = self._log_prior_phi(phi, delta)
        log_prior_delta = self._log_prior_delta(delta)

        total = log_like + log_prior_w + log_prior_phi + log_prior_delta

        if not jnp.isfinite(total):
            return -jnp.inf
        return total

    def _propose_weight_componentwise(self, weight_idx: int) -> float:
        """Propose new value for a single weight component."""
        current_val = self.current_weights[weight_idx]
        step_size = self.step_sizes[weight_idx]
        noise = np.random.normal(0, step_size)
        return current_val + noise

    def _update_weights_componentwise(self) -> int:
        """Update weights using component-wise Metropolis-Hastings."""
        n_accepted = 0

        # Randomize update order
        indices = np.random.permutation(self.n_weights)

        for idx in indices:
            # Propose new value for component idx
            proposal_weights = self.current_weights.copy()
            proposal_weights = proposal_weights.at[idx].set(
                self._propose_weight_componentwise(idx)
            )

            # Compute acceptance probability
            proposal_log_posterior = self._log_posterior(
                proposal_weights, self.current_phi, self.current_delta
            )

            log_alpha = proposal_log_posterior - self.current_log_posterior
            alpha = min(1.0, np.exp(log_alpha))

            # Accept/reject
            if np.random.random() < alpha:
                self.current_weights = proposal_weights
                self.current_log_posterior = proposal_log_posterior
                n_accepted += 1
                self.accept_counts[idx] += 1

            self.proposal_counts[idx] += 1

        return n_accepted

    def _update_phi(self) -> float:
        """Gibbs update for phi."""
        # Full conditional: Gamma(alpha_phi + k/2, beta_phi * delta + 0.5 * w^T P w)
        k = self.n_weights
        quad_form = (
            self.current_weights.T @ self.penalty_matrix @ self.current_weights
        )

        shape = self.alpha_phi + k / 2
        rate = self.beta_phi * self.current_delta + 0.5 * quad_form

        self.current_phi = gamma.rvs(a=shape, scale=1 / rate)
        return self.current_phi

    def _update_delta(self) -> float:
        """Gibbs update for delta."""
        # Full conditional: Gamma(alpha_phi + alpha_delta, beta_phi * phi + beta_delta)
        shape = self.alpha_phi + self.alpha_delta
        rate = self.beta_phi * self.current_phi + self.beta_delta

        self.current_delta = gamma.rvs(a=shape, scale=1 / rate)
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
                        self.step_sizes[i] /= self.config.step_size_factor
                    else:
                        self.step_sizes[i] *= self.config.step_size_factor

                    # Clip to bounds
                    self.step_sizes[i] = np.clip(
                        self.step_sizes[i],
                        self.config.min_step_size,
                        self.config.max_step_size,
                    )

            # Reset counters
            self.accept_counts.fill(0)
            self.proposal_counts.fill(0)

    def _adapt_covariance(self):
        """Adapt proposal covariance using sample history."""
        if not self.config.adapt_covariance:
            return

        self.sample_history.append(self.current_weights.copy())

        # Keep only recent samples
        if len(self.sample_history) > self.config.memory_length:
            self.sample_history.pop(0)

        # Update empirical covariance
        if len(self.sample_history) > 50:  # Need enough samples
            samples = np.array(self.sample_history)
            self.empirical_mean = np.mean(samples, axis=0)
            self.empirical_cov = np.cov(samples.T)

            # Add regularization
            reg = self.config.covariance_regularization
            self.empirical_cov += reg * np.eye(self.n_weights)

    def step(self) -> Dict[str, Any]:
        """Perform one MCMC step."""
        self.iteration += 1

        # 1. Update weights (component-wise)
        n_accepted_weights = self._update_weights_componentwise()

        # 2. Update phi and delta (Gibbs)
        self._update_phi()
        self._update_delta()

        # 3. Update log posterior with new phi, delta
        self._update_log_posterior()

        # 4. Adaptation
        self._adapt_step_sizes()
        self._adapt_covariance()

        # Return step info
        return {
            "weights": self.current_weights.copy(),
            "phi": self.current_phi,
            "delta": self.current_delta,
            "log_posterior": self.current_log_posterior,
            "n_accepted_weights": n_accepted_weights,
            "acceptance_rate": n_accepted_weights / self.n_weights,
            "step_sizes": self.step_sizes.copy(),
        }

    def sample(
        self,
        n_samples: int,
        n_warmup: int = 500,
        thin: int = 1,
        verbose: bool = True,
        chains: int = 1,
    ) -> az.InferenceData:
        """
        Run MCMC sampling.

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
        chains : int
            Number of chains (for arviz compatibility)

        Returns
        -------
        dict
            Dictionary containing samples and diagnostics
        """
        total_iterations = n_warmup + n_samples * thin

        # Storage for all chains
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

        start_time = time.time()

        # TQDM progress bar
        with tqdm(
            total=total_iterations,
            disable=not verbose,
            desc="Adaptive MCMC",
            leave=True,
        ) as pbar:

            for i in range(total_iterations):
                step_info = self.step()

                # Store samples after warmup
                if i >= n_warmup and (i - n_warmup) % thin == 0:
                    all_samples["weights"].append(step_info["weights"])
                    all_samples["phi"].append(step_info["phi"])
                    all_samples["delta"].append(step_info["delta"])

                    # Compute sample statistics
                    log_like = self._log_likelihood(step_info["weights"])
                    log_prior = (
                        self._log_prior_weights(
                            step_info["weights"], step_info["phi"]
                        )
                        + self._log_prior_phi(
                            step_info["phi"], step_info["delta"]
                        )
                        + self._log_prior_delta(step_info["delta"])
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

                # Update progress bar with diagnostics
                if i % 10 == 0:  # Update every 10 iterations
                    phase = "Warmup" if i < n_warmup else "Sampling"
                    desc = (
                        f"{phase} | Accept: {step_info['acceptance_rate']:.3f} | "
                        f"LogPost: {step_info['log_posterior']:.1f} | "
                        f"StepSize: {np.mean(step_info['step_sizes']):.4f}"
                    )
                    pbar.set_description(desc)

                # Store detailed diagnostics less frequently
                if i % 100 == 0:
                    diagnostics["step_sizes_history"].append(
                        step_info["step_sizes"]
                    )
                    diagnostics["iteration_info"].append(
                        {
                            "iteration": i,
                            "acceptance_rate": step_info["acceptance_rate"],
                            "log_posterior": step_info["log_posterior"],
                            "phase": "warmup" if i < n_warmup else "sampling",
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
            print(f"Final acceptance rate: {final_accept:.3f}")
            print(
                f"Target acceptance rate: {self.config.target_accept_rate:.3f}"
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

        results = {
            "samples": all_samples,
            "sample_stats": sample_stats,
            "diagnostics": diagnostics,
            "runtime": runtime,
            "config": self.config,
            "spline_model": self.spline_model,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "chains": chains,
        }
        return self.to_arviz(results)

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
            "chain": range(results["chains"]),
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
        idata.attrs["sampling_method"] = "adaptive_metropolis_hastings"
        idata.attrs["runtime"] = results["runtime"]
        idata.attrs["n_warmup"] = results["n_warmup"]
        idata.attrs["target_accept_rate"] = self.config.target_accept_rate
        idata.attrs["adaptation_window"] = self.config.adaptation_window
        idata.attrs["n_weights"] = self.n_weights
        idata.attrs["spline_degree"] = self.spline_model.degree
        idata.attrs["n_knots"] = self.spline_model.n_knots

        return idata
