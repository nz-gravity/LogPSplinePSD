"""NUTS sampler for multivariate PSD estimation."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import arviz as az
import jax
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
from ..vi_init.adapters import compute_vi_artifacts_multivar
from ..vi_init.defaults import default_init_values_multivar
from .multivar_base import MultivarBaseSampler


@dataclass
class MultivarNUTSConfig(SamplerConfig):
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


def multivariate_psplines_model(
    u_re: jnp.ndarray,  # Wishart replicates (real)
    u_im: jnp.ndarray,  # Wishart replicates (imag)
    nu: int,
    all_bases,
    all_penalties,
    freq_weights: jnp.ndarray,
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
):
    """
    NumPyro model for multivariate PSD estimation using P-splines and Cholesky parameterization.
    """
    # Extract dimensions from input arrays (these are static during compilation)
    n_freq, n_dim, _ = u_re.shape
    n_theta = int(n_dim * (n_dim - 1) / 2)

    component_idx = 0
    log_delta_components = []

    # Diagonal components (one per channel)
    for j in range(n_dim):  # Now n_dim is a concrete Python int
        penalty = all_penalties[component_idx]
        basis = all_bases[component_idx]
        block = sample_pspline_block(
            delta_name=f"delta_{j}",
            phi_name=f"phi_delta_{j}",
            weights_name=f"weights_delta_{j}",
            penalty_matrix=penalty,
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
        )
        component_eval = jnp.einsum("nk,k->n", basis, block["weights"])
        log_delta_components.append(component_eval)
        component_idx += 1

    log_delta_sq = jnp.stack(log_delta_components, axis=1)

    # Off-diagonal components (if multivariate)
    if n_theta > 0:
        penalty = all_penalties[component_idx]
        basis = all_bases[component_idx]
        theta_re_block = sample_pspline_block(
            delta_name="delta_theta_re",
            phi_name="phi_theta_re",
            weights_name="weights_theta_re",
            penalty_matrix=penalty,
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
        )
        theta_re_base = jnp.einsum("nk,k->n", basis, theta_re_block["weights"])
        component_idx += 1
        theta_re = jnp.tile(theta_re_base[:, None], (1, max(1, n_theta)))

        penalty = all_penalties[component_idx]
        basis = all_bases[component_idx]
        theta_im_block = sample_pspline_block(
            delta_name="delta_theta_im",
            phi_name="phi_theta_im",
            weights_name="weights_theta_im",
            penalty_matrix=penalty,
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
        )
        theta_im_base = jnp.einsum("nk,k->n", basis, theta_im_block["weights"])
        component_idx += 1
        theta_im = jnp.tile(theta_im_base[:, None], (1, max(1, n_theta)))
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

    # Likelihood using Wishart replicates
    theta_complex = theta_re + 1j * theta_im
    u_complex = u_re + 1j * u_im

    nu_scale = jnp.asarray(nu, dtype=log_delta_sq.dtype)
    u_resid = u_complex
    idx = 0
    for row in range(1, n_dim):
        count = row
        if count > 0:
            coeff = theta_complex[:, idx : idx + count]
            prev = u_resid[:, :row, :]
            contrib = jnp.einsum("fl,flr->fr", coeff, prev)
            u_resid = u_resid.at[:, row, :].set(u_resid[:, row, :] - contrib)
            idx += count

    residual_power = jnp.sum(jnp.abs(u_resid) ** 2, axis=2)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)
    # Scale log-determinant contribution by coarse-grain frequency weights
    fw = jnp.asarray(freq_weights, dtype=log_delta_sq.dtype)
    sum_log_det = -nu_scale * jnp.sum(fw[:, None] * log_delta_sq)
    log_likelihood = sum_log_det - jnp.sum(residual_power * exp_neg_log_delta)
    numpyro.factor("likelihood", log_likelihood)

    # Store deterministic quantities for diagnostics
    numpyro.deterministic("log_delta_sq", log_delta_sq)
    numpyro.deterministic("theta_re", theta_re)
    numpyro.deterministic("theta_im", theta_im)
    numpyro.deterministic("log_likelihood", log_likelihood)


class MultivarNUTSSampler(VIInitialisationMixin, MultivarBaseSampler):
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
        self._vi_diagnostics: Optional[Dict[str, Any]] = None

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
        self,
        n_samples: int,
        n_warmup: int = 500,
        *,
        only_vi: bool = False,
        **kwargs,
    ) -> az.InferenceData:
        """Run multivariate NUTS sampling."""
        vi_artifacts = compute_vi_artifacts_multivar(
            self, model=multivariate_psplines_model
        )
        init_strategy = (
            vi_artifacts.init_strategy or self._default_init_strategy()
        )
        self.rng_key = vi_artifacts.rng_key
        self._vi_diagnostics = vi_artifacts.diagnostics
        if self._vi_diagnostics and self.config.outdir is not None:
            self._save_vi_diagnostics(
                empirical_psd=self._compute_empirical_psd()
            )

        vi_only_mode = bool(only_vi or getattr(self.config, "only_vi", False))
        if vi_only_mode:
            return self._vi_only_inference_data(vi_artifacts)

        if (
            self.config.verbose
            and vi_artifacts.diagnostics is not None
            and vi_artifacts.diagnostics.get("losses") is not None
        ):
            losses = np.asarray(vi_artifacts.diagnostics["losses"])
            if losses.size:
                guide = vi_artifacts.diagnostics.get("guide", "vi")
                logger.info(
                    f"VI init (multivar) -> guide={guide}, final ELBO {float(losses[-1]):.3f}"
                )

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

        logger.info(
            f"Multivariate NUTS sampler [{self.device}] - {self.n_channels} channels"
        )

        # FIXED: Run sampling with individual arrays instead of MultivarFFT object
        start_time = time.time()
        mcmc.run(
            self.rng_key,
            self.u_re,
            self.u_im,
            self.nu,
            self.all_bases,
            self.all_penalties,
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

        logger.info(f"Sampling completed in {self.runtime:.2f} seconds")

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

        for key in list(samples):
            if key.startswith("phi"):
                samples[key] = jnp.exp(samples[key])

        return self.to_arviz(samples, stats)

    def _default_init_strategy(self):
        init_values = default_init_values_multivar(
            self.spline_model,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )
        return init_to_value(values=init_values)

    def _compile_model(self) -> None:
        """Pre-compile the NumPyro model to speed up warmup."""
        try:
            from numpyro.infer.util import initialize_model

            if self.config.verbose:
                logger.debug("Pre-compiling NumPyro model...")

            # Initialize model with dummy data to trigger compilation
            init_params = initialize_model(
                self.rng_key,
                multivariate_psplines_model,
                model_kwargs=dict(
                    u_re=self.u_re,
                    u_im=self.u_im,
                    nu=self.nu,
                    all_bases=self.all_bases,
                    all_penalties=self.all_penalties,
                    freq_weights=self.freq_weights,
                    alpha_phi=self.config.alpha_phi,
                    beta_phi=self.config.beta_phi,
                    alpha_delta=self.config.alpha_delta,
                    beta_delta=self.config.beta_delta,
                ),
                init_strategy=init_to_value(
                    values=default_init_values_multivar(
                        self.spline_model,
                        alpha_phi=self.config.alpha_phi,
                        beta_phi=self.config.beta_phi,
                        alpha_delta=self.config.alpha_delta,
                        beta_delta=self.config.beta_delta,
                    )
                ),
            )

            if self.config.verbose:
                logger.debug("Model pre-compilation completed")
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Model pre-compilation failed: {e}")

    def _get_lnz(
        self, samples: Dict[str, Any], sample_stats: Dict[str, Any]
    ) -> tuple[float, float]:
        """Compute log normalizing constant for multivariate case."""
        # Use the base class implementation which handles the multivariate case
        return super()._get_lnz(samples, sample_stats)

    @property
    def _logp_kwargs(self) -> Dict[str, Any]:
        return dict(
            u_re=self.u_re,
            u_im=self.u_im,
            nu=self.nu,
            all_bases=self.all_bases,
            all_penalties=self.all_penalties,
            freq_weights=self.freq_weights,
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
        transformed = {}
        for name, value in params.items():
            array = jnp.asarray(value)
            if name.startswith("phi"):
                transformed[name] = jnp.log(array)
            else:
                transformed[name] = array
        return float(self._logpost_fn(transformed))

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
                if name.startswith(("weights_", "phi_", "delta_"))
            }
        else:
            means = vi_artifacts.means or {}
            sample_dict = {
                name: jnp.asarray(value)[None, ...]
                for name, value in means.items()
                if name.startswith(("weights_", "phi_", "delta_"))
            }

        if not sample_dict:
            raise ValueError(
                "Variational-only mode requires VI means or draws for spline parameters."
            )

        params_batch = self._prepare_logpost_params(sample_dict)
        sample_stats = {}
        try:
            sample_stats["lp"] = evaluate_log_density_batch(
                self._logpost_fn, params_batch
            )
        except Exception:
            sample_stats = {}

        samples = dict(sample_dict)
        for key in list(samples):
            if key.startswith("phi"):
                samples[key] = jnp.exp(samples[key])

        self.runtime = 0.0
        return self._create_vi_inference_data(
            samples, sample_stats, diagnostics
        )
