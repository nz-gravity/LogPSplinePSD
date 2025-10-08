"""
NUTS sampler for univariate PSD estimation.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ...logger import logger
from ..base_sampler import SamplerConfig
from ..utils import (
    build_log_density_fn,
    evaluate_log_density_batch,
    sample_pspline_block,
)
from ..vi_init import VIInitialisationMixin
from ..vi_init.adapters import compute_vi_artifacts_univar
from ..vi_init.defaults import default_init_values_univar
from .univar_base import UnivarBaseSampler, log_likelihood  # Updated import


@dataclass
class NUTSConfig(SamplerConfig):
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    save_nuts_diagnostics: bool = True
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: Optional[str] = None
    vi_posterior_draws: int = 256
    vi_progress_bar: Optional[bool] = None


def bayesian_model(
    log_pdgrm: jnp.ndarray,
    lnspline_basis: jnp.ndarray,
    penalty_matrix: jnp.ndarray,
    ln_parametric: jnp.ndarray,
    freq_weights: jnp.ndarray,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
):
    """NumPyro model for univariate PSD estimation."""
    block = sample_pspline_block(
        delta_name="delta",
        phi_name="phi",
        weights_name="weights",
        penalty_matrix=penalty_matrix,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        factor_name="ln_prior",
    )

    weights = block["weights"]
    lnl = log_likelihood(
        weights,
        log_pdgrm,
        lnspline_basis,
        ln_parametric,
        freq_weights,
    )
    numpyro.factor("ln_likelihood", lnl)


class NUTSSampler(VIInitialisationMixin, UnivarBaseSampler):
    """NUTS sampler for univariate PSD estimation."""

    def __init__(self, periodogram, spline_model, config: NUTSConfig = None):
        if config is None:
            config = NUTSConfig()
        super().__init__(periodogram, spline_model, config)
        self.config: NUTSConfig = config  # Type hint for IDE
        self._vi_diagnostics: Optional[Dict[str, np.ndarray]] = None

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
        vi_artifacts = compute_vi_artifacts_univar(self, model=bayesian_model)
        init_strategy = (
            vi_artifacts.init_strategy or self._default_init_strategy()
        )
        self.rng_key = vi_artifacts.rng_key
        self._vi_diagnostics = vi_artifacts.diagnostics
        self._save_vi_diagnostics()

        if (
            self.config.verbose
            and vi_artifacts.diagnostics
            and vi_artifacts.diagnostics.get("losses") is not None
        ):
            losses = np.asarray(vi_artifacts.diagnostics["losses"])
            if losses.size:
                guide = vi_artifacts.diagnostics.get("guide", "vi")
                logger.info(
                    f"VI init -> guide={guide}, final ELBO {float(losses[-1]):.3f}",
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
            logger.info(f"NUTS sampler [{self.device}] {self.rng_key}")

        # Run sampling
        start_time = time.time()
        mcmc.run(
            self.rng_key,
            self.log_pdgrm,
            self.basis_matrix,
            self.penalty_matrix,
            self.log_parametric,
            self.freq_weights,
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
            logger.info(f"Sampling completed in {self.runtime:.2f} seconds")

        # Extract samples and convert to ArviZ
        samples = mcmc.get_samples()
        stats = mcmc.get_extra_fields()

        params_batch = self._prepare_logpost_params(samples)
        stats["lp"] = evaluate_log_density_batch(
            self._logpost_fn, params_batch
        )

        if "phi" in samples:
            samples["phi"] = jnp.exp(samples["phi"])

        return self.to_arviz(samples, stats)

    def _default_init_strategy(self):
        default_values = default_init_values_univar(
            self.spline_model,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )
        return init_to_value(values=default_values)

    @property
    def _logp_kwargs(self):
        """Arguments passed to the NumPyro model / log-density helpers."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            lnspline_basis=self.basis_matrix,
            penalty_matrix=self.penalty_matrix,
            ln_parametric=self.log_parametric,
            freq_weights=self.freq_weights,
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
                logger.debug("Pre-compiling NumPyro model...")

            # Initialize model with dummy data to trigger compilation
            default_values = default_init_values_univar(
                self.spline_model,
                alpha_phi=self.config.alpha_phi,
                beta_phi=self.config.beta_phi,
                alpha_delta=self.config.alpha_delta,
                beta_delta=self.config.beta_delta,
            )

            init_params = initialize_model(
                self.rng_key,
                bayesian_model,
                model_kwargs=self._logp_kwargs,
                init_strategy=init_to_value(values=default_values),
            )

            if self.config.verbose:
                logger.debug("Model pre-compilation completed")
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Model pre-compilation failed: {e}")

    def _compute_log_posterior(
        self, weights: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """Compute log posterior for given parameters via the NumPyro model."""
        params = {
            "weights": jnp.asarray(weights),
            "phi": jnp.log(jnp.asarray(phi)),
            "delta": jnp.asarray(delta),
        }
        return float(self._logpost_fn(params))

    def _prepare_logpost_params(self, samples: Dict[str, jnp.ndarray]):
        """Stack posterior samples into a pytree for log-density evaluation."""
        return {
            "weights": jnp.asarray(samples["weights"]),
            "phi": jnp.asarray(samples["phi"]),
            "delta": jnp.asarray(samples["delta"]),
        }
