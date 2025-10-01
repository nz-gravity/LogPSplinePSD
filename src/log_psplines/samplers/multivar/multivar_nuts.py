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
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..base_sampler import SamplerConfig
from ..utils import (
    build_log_density_fn,
    evaluate_log_density_batch,
    pspline_hyperparameter_initials,
    sample_pspline_block,
)
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
    all_bases,
    all_penalties,
    all_penalty_whiten,
    all_penalty_unwhiten_T,
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

    # Diagonal components (one per channel)
    for j in range(n_dim):  # Now n_dim is a concrete Python int
        penalty = all_penalties[component_idx]
        basis = all_bases[component_idx]
        penalty_whiten = all_penalty_whiten[component_idx]
        block = sample_pspline_block(
            delta_name=f"delta_{j}",
            phi_name=f"phi_delta_{j}",
            weights_name=f"weights_delta_{j}",
            penalty_whiten=penalty_whiten,
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
        penalty_whiten = all_penalty_whiten[component_idx]
        theta_re_block = sample_pspline_block(
            delta_name="delta_theta_re",
            phi_name="phi_theta_re",
            weights_name="weights_theta_re",
            penalty_whiten=penalty_whiten,
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
        penalty_whiten = all_penalty_whiten[component_idx]
        theta_im_block = sample_pspline_block(
            delta_name="delta_theta_im",
            phi_name="phi_theta_im",
            weights_name="weights_theta_im",
            penalty_whiten=penalty_whiten,
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

    # Likelihood using individual arrays
    log_likelihood = whittle_likelihood_arrays(
        y_re, y_im, Z_re, Z_im, log_delta_sq, theta_re, theta_im
    )
    numpyro.factor("likelihood", log_likelihood)

    # Store deterministic quantities for diagnostics
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
            self.all_penalty_whiten,
            self.all_penalty_unwhiten_T,
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

    def _get_initial_values(self) -> Dict[str, jnp.ndarray]:
        """Get initial values for all parameters."""
        init_values = {}

        delta_init, phi_init = pspline_hyperparameter_initials(
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

        # Initialize diagonal components
        for j in range(self.n_channels):
            init_values[f"delta_{j}"] = delta_init
            init_values[f"phi_delta_{j}"] = jnp.log(phi_init)
            init_values[f"weights_delta_{j}"] = (
                self.spline_model.diagonal_models[j].weights
            )
            init_values[
                f"weights_delta_{j}_latent"
            ] = self.all_penalty_unwhiten_T[j] @ (
                self.spline_model.diagonal_models[j].weights
                * jnp.sqrt(phi_init)
            )

        # Initialize off-diagonal components if needed
        if self.n_theta > 0:
            init_values["delta_theta_re"] = delta_init
            init_values["phi_theta_re"] = jnp.log(phi_init)
            init_values["weights_theta_re"] = (
                self.spline_model.offdiag_re_model.weights
            )
            latent_re_idx = self.n_channels
            init_values[
                "weights_theta_re_latent"
            ] = self.all_penalty_unwhiten_T[latent_re_idx] @ (
                self.spline_model.offdiag_re_model.weights * jnp.sqrt(phi_init)
            )

            init_values["delta_theta_im"] = delta_init
            init_values["phi_theta_im"] = jnp.log(phi_init)
            init_values["weights_theta_im"] = (
                self.spline_model.offdiag_im_model.weights
            )
            theta_im_idx = self.n_channels + 1
            theta_im_unwhiten_T = (
                self.all_penalty_unwhiten_T[theta_im_idx]
                if len(self.all_penalty_unwhiten_T) > theta_im_idx
                else self.all_penalty_unwhiten_T[self.n_channels]
            )
            init_values["weights_theta_im_latent"] = theta_im_unwhiten_T @ (
                self.spline_model.offdiag_im_model.weights * jnp.sqrt(phi_init)
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
                    all_penalty_whiten=self.all_penalty_whiten,
                    all_penalty_unwhiten_T=self.all_penalty_unwhiten_T,
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
            all_penalty_whiten=self.all_penalty_whiten,
            all_penalty_unwhiten_T=self.all_penalty_unwhiten_T,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _prepare_logpost_params(
        self, samples: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        params: Dict[str, jnp.ndarray] = {}
        for name, array in samples.items():
            if name.startswith(
                ("weights_", "phi_", "delta_")
            ) or name.endswith("_latent"):
                params[name] = jnp.asarray(array)

        def ensure_latent(
            weight_name: str, phi_name: str, unwhiten_T: jnp.ndarray
        ):
            latent_name = f"{weight_name}_latent"
            if latent_name in params:
                return
            if weight_name not in params or phi_name not in params:
                return
            weights_val = params[weight_name]
            phi_log = params[phi_name]
            sqrt_phi = jnp.exp(0.5 * phi_log)
            if weights_val.ndim == 1:
                latent = unwhiten_T @ (weights_val * sqrt_phi)
            else:
                latent = (weights_val * sqrt_phi[:, None]) @ unwhiten_T.T
            params[latent_name] = latent

        for j in range(self.n_channels):
            ensure_latent(
                f"weights_delta_{j}",
                f"phi_delta_{j}",
                self.all_penalty_unwhiten_T[j],
            )

        if self.n_theta > 0:
            ensure_latent(
                "weights_theta_re",
                "phi_theta_re",
                self.all_penalty_unwhiten_T[self.n_channels],
            )
            theta_im_idx = self.n_channels + 1
            theta_im_unwhiten_T = (
                self.all_penalty_unwhiten_T[theta_im_idx]
                if len(self.all_penalty_unwhiten_T) > theta_im_idx
                else self.all_penalty_unwhiten_T[self.n_channels]
            )
            ensure_latent(
                "weights_theta_im",
                "phi_theta_im",
                theta_im_unwhiten_T,
            )

        return params

    def _compute_log_posterior(self, params: Dict[str, jnp.ndarray]) -> float:
        working: Dict[str, jnp.ndarray] = {}
        for name, value in params.items():
            working[name] = jnp.asarray(value)

        def ensure_latent(
            weight_name: str, phi_name: str, unwhiten_T: jnp.ndarray
        ):
            latent_name = f"{weight_name}_latent"
            if latent_name in working and weight_name in working:
                return
            if weight_name not in working or phi_name not in working:
                return
            weights_val = working[weight_name]
            phi_val = working[phi_name]
            sqrt_phi = jnp.sqrt(phi_val)
            if weights_val.ndim == 1:
                latent = unwhiten_T @ (weights_val * sqrt_phi)
            else:
                latent = (weights_val * sqrt_phi[:, None]) @ unwhiten_T.T
            working[latent_name] = latent

        for j in range(self.n_channels):
            ensure_latent(
                f"weights_delta_{j}",
                f"phi_delta_{j}",
                self.all_penalty_unwhiten_T[j],
            )

        if self.n_theta > 0:
            ensure_latent(
                "weights_theta_re",
                "phi_theta_re",
                self.all_penalty_unwhiten_T[self.n_channels],
            )
            theta_im_idx = self.n_channels + 1
            theta_im_unwhiten_T = (
                self.all_penalty_unwhiten_T[theta_im_idx]
                if len(self.all_penalty_unwhiten_T) > theta_im_idx
                else self.all_penalty_unwhiten_T[self.n_channels]
            )
            ensure_latent(
                "weights_theta_im",
                "phi_theta_im",
                theta_im_unwhiten_T,
            )

        transformed = {}
        for name, value in working.items():
            if name.startswith("phi"):
                transformed[name] = jnp.log(value)
            else:
                transformed[name] = value
        return float(self._logpost_fn(transformed))
