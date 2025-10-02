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

from ..base_sampler import SamplerConfig
from ..utils import (
    build_log_density_fn,
    evaluate_log_density_batch,
    pspline_hyperparameter_initials,
    sample_pspline_block,
)
from ..vi_init import VIInitialisationArtifacts, VIInitialisationMixin
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
        self, n_samples: int, n_warmup: int = 500, **kwargs
    ) -> az.InferenceData:
        """Run multivariate NUTS sampling."""
        vi_artifacts = self._select_initialisation_strategy()
        init_strategy = (
            vi_artifacts.init_strategy or self._default_init_strategy()
        )
        self.rng_key = vi_artifacts.rng_key
        self._vi_diagnostics = vi_artifacts.diagnostics

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

    def _select_initialisation_strategy(self) -> VIInitialisationArtifacts:
        guide_spec = self.config.vi_guide or self._suggest_vi_guide()

        def _postprocess(vi_result):
            init_values = {
                name: jnp.asarray(value)
                for name, value in vi_result.means.items()
            }

            vi_psd_np = None
            try:
                log_delta_terms = []
                for channel_index in range(self.n_channels):
                    weights_name = f"weights_delta_{channel_index}"
                    weights = vi_result.means.get(weights_name)
                    if weights is None:
                        raise KeyError(weights_name)
                    basis = self.all_bases[channel_index]
                    component_eval = jnp.einsum("nk,k->n", basis, weights)
                    log_delta_terms.append(component_eval)
                log_delta_sq = jnp.stack(log_delta_terms, axis=1)

                if self.n_theta > 0:
                    basis_theta = self.all_bases[self.n_channels]
                    weights_theta_re = vi_result.means.get("weights_theta_re")
                    weights_theta_im = vi_result.means.get("weights_theta_im")
                    if weights_theta_re is None or weights_theta_im is None:
                        raise KeyError("theta weights")
                    theta_re_base = jnp.einsum(
                        "nk,k->n", basis_theta, weights_theta_re
                    )
                    theta_im_base = jnp.einsum(
                        "nk,k->n", basis_theta, weights_theta_im
                    )
                    theta_re = jnp.tile(
                        theta_re_base[:, None], (1, max(1, self.n_theta))
                    )
                    theta_im = jnp.tile(
                        theta_im_base[:, None], (1, max(1, self.n_theta))
                    )
                else:
                    theta_re = jnp.zeros((self.n_freq, 0))
                    theta_im = jnp.zeros((self.n_freq, 0))

                vi_psd = self.spline_model.reconstruct_psd_matrix(
                    log_delta_sq[None, ...],
                    theta_re[None, ...],
                    theta_im[None, ...],
                    n_samples_max=1,
                )[0]
                vi_psd_np = np.asarray(jax.device_get(vi_psd))
            except Exception as err:  # pragma: no cover - defensive fallback
                if self.config.verbose:
                    print(
                        f"Warning: could not build VI PSD diagnostics ({err})."
                    )

            true_psd = None
            if self.config.true_psd is not None:
                true_psd = np.asarray(jax.device_get(self.config.true_psd))

            diagnostics = {
                "psd_matrix": vi_psd_np,
                "true_psd": true_psd,
            }
            return init_values, diagnostics

        artifacts = self._run_vi_initialisation(
            model=multivariate_psplines_model,
            model_args=(
                self.y_re,
                self.y_im,
                self.Z_re,
                self.Z_im,
                self.all_bases,
                self.all_penalties,
                self.config.alpha_phi,
                self.config.beta_phi,
                self.config.alpha_delta,
                self.config.beta_delta,
            ),
            guide=guide_spec,
            postprocess=_postprocess,
        )

        if self.config.verbose and artifacts.diagnostics is not None:
            losses = artifacts.diagnostics.get("losses")
            if losses is not None:
                losses_arr = np.asarray(losses)
                if losses_arr.size:
                    print(
                        "VI init (multivar) -> guide=%s, final ELBO %.3f"
                        % (
                            artifacts.diagnostics.get("guide", "vi"),
                            float(losses_arr[-1]),
                        )
                    )

        return artifacts

    def _default_init_strategy(self):
        init_values = self._get_initial_values()
        return init_to_value(values=init_values)

    def _suggest_vi_guide(self) -> str:
        total_latents = 0
        for model in self.spline_model.diagonal_models:
            total_latents += model.n_basis + 2

        if self.n_theta > 0:
            total_latents += self.spline_model.offdiag_re_model.n_basis + 2
            total_latents += self.spline_model.offdiag_im_model.n_basis + 2

        if total_latents <= 120:
            return "diag"

        rank = max(16, min(64, total_latents // 6))
        rank = min(rank, max(2, total_latents - 1))
        if rank < 2:
            return "diag"
        return f"lowrank:{rank}"

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

        # Initialize off-diagonal components if needed
        if self.n_theta > 0:
            init_values["delta_theta_re"] = delta_init
            init_values["phi_theta_re"] = jnp.log(phi_init)
            init_values["weights_theta_re"] = (
                self.spline_model.offdiag_re_model.weights
            )

            init_values["delta_theta_im"] = delta_init
            init_values["phi_theta_im"] = jnp.log(phi_init)
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
        transformed = {}
        for name, value in params.items():
            array = jnp.asarray(value)
            if name.startswith("phi"):
                transformed[name] = jnp.log(array)
            else:
                transformed[name] = array
        return float(self._logpost_fn(transformed))
