"""
NUTS sampler for univariate PSD estimation.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.linalg import solve_triangular
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
    penalty_chol: jnp.ndarray,
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
        penalty_chol=penalty_chol,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
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
        self,
        n_samples: int,
        n_warmup: int = 500,
        *,
        only_vi: bool = False,
        **kwargs,
    ) -> az.InferenceData:
        """Run NUTS sampling."""
        vi_artifacts = compute_vi_artifacts_univar(self, model=bayesian_model)
        init_strategy = (
            vi_artifacts.init_strategy or self._default_init_strategy()
        )
        self.rng_key = vi_artifacts.rng_key
        self._vi_diagnostics = vi_artifacts.diagnostics
        self._save_vi_diagnostics()

        vi_only_mode = bool(only_vi or getattr(self.config, "only_vi", False))
        if vi_only_mode:
            return self._vi_only_inference_data(vi_artifacts)

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
            chain_method=self.chain_method,
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
            self.penalty_chol,
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
        samples = mcmc.get_samples(group_by_chain=True)
        stats = mcmc.get_extra_fields(group_by_chain=True)

        params_batch = self._prepare_logpost_params(samples)
        stats["lp"] = evaluate_log_density_batch(
            self._logpost_fn, params_batch
        )
        lp_arr = jnp.asarray(stats.get("lp"))
        if lp_arr.ndim == 1:
            n_chains = int(self.config.num_chains)
            n_draws = int(samples["weights"].shape[1])
            if lp_arr.size == n_chains * n_draws:
                stats["lp"] = lp_arr.reshape((n_chains, n_draws))

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

    def _vi_only_inference_data(
        self, vi_artifacts: "VIInitialisationArtifacts"
    ) -> az.InferenceData:
        diagnostics = vi_artifacts.diagnostics or {}
        posterior_draws = vi_artifacts.posterior_draws or diagnostics.get(
            "vi_samples"
        )

        if posterior_draws:
            sample_dict = {
                name: jnp.asarray(array)
                for name, array in posterior_draws.items()
                if name in {"weights_z", "phi", "delta"}
            }
        else:
            means = vi_artifacts.means or {}
            sample_dict = {
                name: jnp.asarray(value)[None, ...]
                for name, value in means.items()
                if name in {"weights_z", "phi", "delta"}
            }

        missing = {"weights_z", "phi", "delta"} - set(sample_dict)
        if missing:
            raise ValueError(
                "Variational-only mode requires VI means for weights_z, phi, and delta."
            )

        params_batch = self._prepare_logpost_params(sample_dict)
        sample_stats = {}
        try:
            sample_stats["lp"] = evaluate_log_density_batch(
                self._logpost_fn, params_batch
            )
            lp_arr = jnp.asarray(sample_stats.get("lp"))
            n_chains = int(self.config.num_chains)
            if lp_arr.ndim == 1 and lp_arr.size % n_chains == 0:
                sample_stats["lp"] = lp_arr.reshape((n_chains, -1))
        except Exception:
            sample_stats = {}

        log_phi = jnp.asarray(sample_dict["phi"])
        phi = jnp.exp(log_phi)
        phi = jnp.clip(phi, a_min=1e-12, a_max=1e12)
        z = jnp.asarray(sample_dict["weights_z"])
        if z.ndim == 1:
            weights = solve_triangular(
                self.penalty_chol, z, lower=True, trans="T"
            ) / jnp.sqrt(phi)
        else:
            weights = jax.vmap(
                lambda zi, phii: solve_triangular(
                    self.penalty_chol, zi, lower=True, trans="T"
                )
                / jnp.sqrt(phii)
            )(z, phi)

        samples = dict(sample_dict)
        samples["phi"] = phi
        samples["weights"] = weights
        self.runtime = 0.0
        return self._create_vi_inference_data(
            samples, sample_stats, diagnostics
        )

    @property
    def _logp_kwargs(self):
        """Arguments passed to the NumPyro model / log-density helpers."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            lnspline_basis=self.basis_matrix,
            penalty_chol=self.penalty_chol,
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
        self, weights_z: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """Compute log posterior for given parameters via the NumPyro model."""
        params = {
            "weights_z": jnp.asarray(weights_z),
            "phi": jnp.log(jnp.asarray(phi)),
            "delta": jnp.asarray(delta),
        }
        return float(self._logpost_fn(params))

    def _prepare_logpost_params(self, samples: Dict[str, jnp.ndarray]):
        """Stack posterior samples into a pytree for log-density evaluation."""
        weights_z = jnp.asarray(samples["weights_z"])
        if weights_z.ndim == 1:
            weights_z = weights_z[None, :]
        elif weights_z.ndim >= 3:
            # Flatten chain and draw dimensions into a single batch axis
            weights_z = weights_z.reshape((-1, weights_z.shape[-1]))

        phi = jnp.asarray(samples["phi"])
        phi = phi.reshape(-1) if phi.ndim else phi.reshape(1)

        delta = jnp.asarray(samples["delta"])
        delta = delta.reshape(-1) if delta.ndim else delta.reshape(1)

        return {
            "weights_z": weights_z,
            "phi": phi,
            "delta": delta,
        }
