"""Blocked NUTS sampler for multivariate PSD (Cholesky parameterisation).

This module implements the factorised form of the multivariate Whittle
likelihood when the inverse spectral density is parameterised via a unit lower
triangular Cholesky factor ``T(f)`` and diagonal matrix ``D(f)``:

    S(f)^{-1} = T(f)^H D(f)^{-1} T(f).

With this parameterisation the Whittle likelihood decouples across the rows of
``T``. Writing the j-th row parameters as

    - diagonal log-variances:  log_delta_sq_j(f) = log(δ_j(f)^2)
    - off–diagonal complex coefficients: θ_{j1}(f), …, θ_{j,j-1}(f)

the joint likelihood factorises as a product of p independent likelihoods
L_j(δ_j, θ_{j,<j}). Each factor has the form

    L_j ∝ ∏_k exp{-|d_j(f_k) - Σ_{l<j} θ_{jl}(f_k) d_l(f_k)|^2 / δ_j(f_k)^2}
           × δ_j(f_k)^{-2}.

The class :class:`MultivarBlockedNUTSSampler` exploits this by running a
separate NUTS chain for each block ``j = 0,…,p-1`` (0‑based indexing). Each
chain samples only the parameters that affect the j‑th likelihood factor and is
therefore independent of the other chains. The per‑block results are then
assembled into global arrays ``log_delta_sq`` with shape (draw, freq, p) and
``theta_re|im`` with shape (draw, freq, n_theta) matching the unified sampler.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from ..base_sampler import SamplerConfig
from ..utils import sample_pspline_block
from .multivar_base import MultivarBaseSampler


def _blocked_channel_model(
    channel_index: int,
    y_re: jnp.ndarray,
    y_im: jnp.ndarray,
    Z_re: jnp.ndarray,
    Z_im: jnp.ndarray,
    basis_delta: jnp.ndarray,
    penalty_delta: jnp.ndarray,
    basis_theta: jnp.ndarray,
    penalty_theta: jnp.ndarray,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> None:
    """NumPyro model for a single Cholesky block (row of ``T``).

    Parameters
    ----------
    channel_index
        0‑based row index ``j`` in the Cholesky factor (channel in the data).
    y_re, y_im
        Real/imaginary parts of the DFT for the j‑th channel, shape ``(n_freq,)``.
    Z_re, Z_im
        Real/imaginary parts of the design matrix for the off‑diagonal terms of
        row ``j``. Shape ``(n_freq, j)``. Column ``l`` corresponds to
        ``d_l(f)`` with ``l < j``.
    basis_delta, penalty_delta
        P‑spline basis/penalty for ``log δ_j(f)^2``.
    basis_theta, penalty_theta
        P‑spline basis/penalty shared by all θ_{jl}(f) in this row.
    alpha_phi, beta_phi, alpha_delta, beta_delta
        Hyperparameters for the hierarchical priors used in
        :func:`sample_pspline_block`.

    Notes
    -----
    - The likelihood implemented here corresponds to Eq. (likelihood_j) in your
      draft: the residual is ``u_j(f) = d_j(f) − Σ_{l<j} θ_{jl}(f) d_l(f)`` and
      the contribution to the log-likelihood is
      ``−Σ_k log δ_j(f_k)^2 − Σ_k |u_j(f_k)|^2 / δ_j(f_k)^2`` up to constants.
    - Deterministic nodes record the evaluated spline fields so downstream code
      can reconstruct the PSD matrix without re-evaluating the splines.
    """

    channel_label = f"{channel_index}"

    delta_block = sample_pspline_block(
        delta_name=f"delta_{channel_label}",
        phi_name=f"phi_delta_{channel_label}",
        weights_name=f"weights_delta_{channel_label}",
        penalty_matrix=penalty_delta,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
    )
    log_delta_sq = jnp.einsum("nk,k->n", basis_delta, delta_block["weights"])

    n_freq = y_re.shape[0]
    n_theta_block = Z_re.shape[1]

    if n_theta_block > 0:
        theta_re_components = []
        theta_im_components = []

        for theta_idx in range(n_theta_block):
            theta_prefix = f"theta_re_{channel_label}_{theta_idx}"
            theta_re_block = sample_pspline_block(
                delta_name=f"delta_{theta_prefix}",
                phi_name=f"phi_{theta_prefix}",
                weights_name=f"weights_{theta_prefix}",
                penalty_matrix=penalty_theta,
                alpha_phi=alpha_phi,
                beta_phi=beta_phi,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
            )
            theta_re_eval = jnp.einsum(
                "nk,k->n", basis_theta, theta_re_block["weights"]
            )
            theta_re_components.append(theta_re_eval)

            theta_im_prefix = f"theta_im_{channel_label}_{theta_idx}"
            theta_im_block = sample_pspline_block(
                delta_name=f"delta_{theta_im_prefix}",
                phi_name=f"phi_{theta_im_prefix}",
                weights_name=f"weights_{theta_im_prefix}",
                penalty_matrix=penalty_theta,
                alpha_phi=alpha_phi,
                beta_phi=beta_phi,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
            )

            theta_im_eval = jnp.einsum(
                "nk,k->n", basis_theta, theta_im_block["weights"]
            )
            theta_im_components.append(theta_im_eval)

        theta_re = jnp.stack(theta_re_components, axis=1)
        theta_im = jnp.stack(theta_im_components, axis=1)
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

    exp_neg_log_delta = jnp.exp(-log_delta_sq)
    sum_log_det = -jnp.sum(log_delta_sq)

    if n_theta_block > 0:
        z_theta_re = jnp.sum(Z_re * theta_re, axis=1) - jnp.sum(
            Z_im * theta_im, axis=1
        )
        z_theta_im = jnp.sum(Z_re * theta_im, axis=1) + jnp.sum(
            Z_im * theta_re, axis=1
        )
        u_re = y_re - z_theta_re
        u_im = y_im - z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    residual_power = u_re**2 + u_im**2
    log_likelihood = sum_log_det - jnp.sum(residual_power * exp_neg_log_delta)

    numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)

    numpyro.deterministic(f"log_delta_sq_{channel_label}", log_delta_sq)
    if n_theta_block > 0:
        numpyro.deterministic(f"theta_re_{channel_label}", theta_re)
        numpyro.deterministic(f"theta_im_{channel_label}", theta_im)

    numpyro.deterministic(
        f"log_likelihood_block_{channel_label}", log_likelihood
    )


@dataclass
class MultivarBlockedNUTSConfig(SamplerConfig):
    """Configuration for the blocked multivariate NUTS sampler.

    Attributes
    ----------
    target_accept_prob, max_tree_depth, dense_mass
        Standard NUTS controls forwarded to NumPyro.
    save_nuts_diagnostics
        When ``True``, store per‑block NUTS diagnostics (accept_prob, num_steps,
        etc.) under names like ``accept_prob_channel_j``.
    """

    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    save_nuts_diagnostics: bool = True


class MultivarBlockedNUTSSampler(MultivarBaseSampler):
    """Run independent NUTS chains for each row of the Cholesky factor.

    For ``p`` channels the sampler iterates ``j = 0,…,p-1`` and runs a separate
    NumPyro/NUTS inference for the j‑th block using :func:`_blocked_channel_model`.
    The block observes ``(y_re[:, j], y_im[:, j])`` and uses as regressors the
    previous channels' Fourier coefficients encoded in ``Z_{·, j, :j}``.

    The per‑block posterior draws are merged into global deterministic arrays:
    - ``log_delta_sq`` with shape ``(draw, freq, p)``
    - ``theta_re`` / ``theta_im`` with shape ``(draw, freq, n_theta)``

    These arrays match what the unified multivariate sampler produces, so all
    downstream plotting and diagnostics work identically.
    """

    def __init__(
        self,
        fft_data,
        spline_model,
        config: MultivarBlockedNUTSConfig | None = None,
    ):
        if config is None:
            config = MultivarBlockedNUTSConfig()
        super().__init__(fft_data, spline_model, config)
        self.config: MultivarBlockedNUTSConfig = config

        theta_basis_idx = self.n_channels
        self._theta_basis = (
            self.all_bases[theta_basis_idx]
            if self.n_theta > 0
            else jnp.zeros((self.n_freq, 0), dtype=jnp.float32)
        )
        self._theta_penalty = (
            self.all_penalties[theta_basis_idx]
            if self.n_theta > 0
            else jnp.zeros((0, 0))
        )

    @property
    def sampler_type(self) -> str:
        return "multivariate_blocked_nuts"

    def sample(
        self, n_samples: int, n_warmup: int = 500, **kwargs
    ) -> az.InferenceData:
        """Run the blocked inference and assemble results.

        Steps
        -----
        1. For each channel ``j`` build the design slice ``Z[:, j, :j]``
           (consistent with the triangular ordering used by
           :class:`~log_psplines.datatypes.MultivarFFT`), and run NUTS on the
           corresponding block model.
        2. Move per‑block deterministics out of the ``samples`` dict into
           ``sample_stats`` and, for diagnostics, optionally rename NumPyro’s
           NUTS fields with ``_channel_{j}`` suffixes.
        3. Stack ``log_delta_sq_j`` across channels and place the θ blocks into
           the correct positions of the global ``theta_re|im`` arrays.
        4. Sum the block log‑likelihoods to obtain the joint log‑likelihood.
        """
        if self.config.verbose:
            print(
                f"Blocked multivariate NUTS sampler [{self.device}] - {self.n_channels} channels"
            )

        combined_samples: Dict[str, np.ndarray] = {}
        combined_stats: Dict[str, np.ndarray] = {}

        channel_log_delta = []
        theta_re_total = None
        theta_im_total = None
        log_likelihood_total = None

        rng_key = self.rng_key
        total_runtime = 0.0

        for channel_index in range(self.n_channels):
            rng_key, subkey = jax.random.split(rng_key)

            delta_basis = self.all_bases[channel_index]
            delta_penalty = self.all_penalties[channel_index]

            theta_start = channel_index * (channel_index - 1) // 2
            theta_count = channel_index

            Z_re_block = (
                self.Z_re[
                    :, channel_index, theta_start : theta_start + theta_count
                ]
                if theta_count > 0
                else jnp.zeros((self.n_freq, 0))
            )
            Z_im_block = (
                self.Z_im[
                    :, channel_index, theta_start : theta_start + theta_count
                ]
                if theta_count > 0
                else jnp.zeros((self.n_freq, 0))
            )

            kernel = NUTS(
                _blocked_channel_model,
                target_accept_prob=self.config.target_accept_prob,
                max_tree_depth=self.config.max_tree_depth,
                dense_mass=self.config.dense_mass,
            )

            mcmc = MCMC(
                kernel,
                num_warmup=n_warmup,
                num_samples=n_samples,
                num_chains=self.config.num_chains,
                progress_bar=self.config.verbose,
                jit_model_args=False,
            )

            start_time = time.time()
            mcmc.run(
                subkey,
                channel_index,
                self.y_re[:, channel_index],
                self.y_im[:, channel_index],
                Z_re_block,
                Z_im_block,
                delta_basis,
                delta_penalty,
                self._theta_basis,
                self._theta_penalty,
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
            total_runtime += time.time() - start_time

            block_samples = mcmc.get_samples()
            for sample_key in list(block_samples):
                if sample_key.startswith("phi"):
                    block_samples[sample_key] = jnp.exp(
                        block_samples[sample_key]
                    )
            block_stats = mcmc.get_extra_fields()

            deterministic_keys = [
                f"log_delta_sq_{channel_index}",
                f"theta_re_{channel_index}",
                f"theta_im_{channel_index}",
                f"log_likelihood_block_{channel_index}",
            ]

            for det_key in deterministic_keys:
                if det_key in block_samples:
                    block_stats[det_key] = block_samples.pop(det_key)

            if self.config.save_nuts_diagnostics:
                for diag_key in [
                    "potential_energy",
                    "energy",
                    "num_steps",
                    "accept_prob",
                    "diverging",
                ]:
                    if diag_key in block_stats:
                        renamed = (
                            f"{diag_key}_channel_{channel_index}",
                            block_stats.pop(diag_key),
                        )
                        block_stats[renamed[0]] = renamed[1]

            combined_samples.update(block_samples)

            log_delta_channel = block_stats.pop(
                f"log_delta_sq_{channel_index}"
            )
            channel_log_delta.append(log_delta_channel)

            if theta_count > 0 and self.n_theta > 0:
                theta_re_block = block_stats.pop(f"theta_re_{channel_index}")
                theta_im_block = block_stats.pop(f"theta_im_{channel_index}")

                if theta_re_total is None:
                    n_draws = theta_re_block.shape[0]
                    theta_re_total = jnp.zeros(
                        (n_draws, self.n_freq, self.n_theta)
                    )
                    theta_im_total = jnp.zeros(
                        (n_draws, self.n_freq, self.n_theta)
                    )

                theta_slice = slice(theta_start, theta_start + theta_count)
                theta_re_total = theta_re_total.at[:, :, theta_slice].set(
                    theta_re_block
                )
                theta_im_total = theta_im_total.at[:, :, theta_slice].set(
                    theta_im_block
                )

            block_log_likelihood = block_stats.pop(
                f"log_likelihood_block_{channel_index}"
            )

            log_likelihood_total = (
                block_log_likelihood
                if log_likelihood_total is None
                else log_likelihood_total + block_log_likelihood
            )

            combined_stats.update(block_stats)

        self.runtime = total_runtime

        log_delta_sq = jnp.stack(channel_log_delta, axis=2)

        if theta_re_total is None:
            theta_re_total = jnp.zeros((log_delta_sq.shape[0], self.n_freq, 0))
            theta_im_total = jnp.zeros((log_delta_sq.shape[0], self.n_freq, 0))

        combined_stats.update(
            {
                "log_delta_sq": np.asarray(log_delta_sq),
                "theta_re": np.asarray(theta_re_total),
                "theta_im": np.asarray(theta_im_total),
                "log_likelihood": np.asarray(log_likelihood_total),
            }
        )

        return self.to_arviz(combined_samples, combined_stats)

    def _get_lnz(
        self, samples: Dict[str, jnp.ndarray], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """LnZ is currently not computed for the multivariate samplers."""
        return super()._get_lnz(samples, sample_stats)
