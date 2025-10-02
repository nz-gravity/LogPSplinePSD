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
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ...inference.vi import fit_vi
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
    lnl = log_likelihood(weights, log_pdgrm, lnspline_basis, ln_parametric)
    numpyro.factor("ln_likelihood", lnl)


class NUTSSampler(UnivarBaseSampler):
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
        init_strategy, run_key = self._select_initialisation_strategy()
        self.rng_key = run_key

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

        return self.to_arviz(samples, stats)

    def _select_initialisation_strategy(self):
        """Return the init strategy (and RNG key) for the upcoming run."""
        self._vi_diagnostics = None
        if not self.config.init_from_vi:
            return self._default_init_strategy(), self.rng_key

        key_vi, key_run = jax.random.split(self.rng_key)

        progress_bar = (
            self.config.vi_progress_bar
            if self.config.vi_progress_bar is not None
            else self.config.verbose
        )

        guide_spec = self.config.vi_guide or self._suggest_vi_guide()

        try:
            vi_result = fit_vi(
                model=bayesian_model,
                rng_key=key_vi,
                vi_steps=self.config.vi_steps,
                optimizer_lr=self.config.vi_lr,
                model_args=(
                    self.log_pdgrm,
                    self.basis_matrix,
                    self.penalty_matrix,
                    self.log_parametric,
                    self.config.alpha_phi,
                    self.config.beta_phi,
                    self.config.alpha_delta,
                    self.config.beta_delta,
                ),
                guide=guide_spec,
                posterior_draws=self.config.vi_posterior_draws,
                progress_bar=progress_bar,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self.config.verbose:
                print(
                    f"VI initialisation failed ({exc}) - using default init."
                )
            return self._default_init_strategy(), key_run

        init_values = {
            name: jnp.asarray(value) for name, value in vi_result.means.items()
        }
        init_strategy = init_to_value(values=init_values)

        losses = np.asarray(jax.device_get(vi_result.losses))
        means_np = {
            name: np.asarray(jax.device_get(value))
            for name, value in vi_result.means.items()
        }
        weights_mean = means_np.get("weights")
        true_psd = None
        if self.config.true_psd is not None:
            true_psd = np.asarray(jax.device_get(self.config.true_psd))

        vi_psd = None
        if weights_mean is not None:
            ln_psd = self.spline_model(vi_result.means["weights"])
            vi_psd = np.asarray(jax.device_get(jnp.exp(ln_psd)))

        self._vi_diagnostics = {
            "losses": losses,
            "guide": vi_result.guide_name,
            "weights": weights_mean,
            "psd": vi_psd,
            "true_psd": true_psd,
        }

        if self.config.verbose:
            final_loss = (
                float(vi_result.losses[-1])
                if vi_result.losses.size
                else float("nan")
            )
            print(
                "VI init -> guide=%s, final ELBO %.3f"
                % (vi_result.guide_name, final_loss)
            )

        return init_strategy, key_run

    def _default_init_strategy(self):
        delta_0, phi_0 = pspline_hyperparameter_initials(
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
            divide_phi_by_delta=True,
        )
        return init_to_value(
            values=dict(
                delta=delta_0,
                phi=jnp.log(phi_0),
                weights=self.spline_model.weights,
            )
        )

    def _suggest_vi_guide(self) -> str:
        n_latents = self.n_weights + 2  # weights plus phi/delta
        if n_latents <= 64:
            return "diag"
        rank = max(8, min(32, n_latents // 4))
        rank = min(rank, max(2, n_latents - 1))
        if rank < 2:
            return "diag"
        return f"lowrank:{rank}"

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
