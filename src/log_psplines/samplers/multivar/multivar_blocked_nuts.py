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

Example
-------
Enable an innovation noise floor for Block 3 using a theory-scaled PSD:

>>> config = MultivarBlockedNUTSConfig(
...     use_noise_floor=True,
...     noise_floor_mode="theory_scaled",
...     noise_floor_scale=1e-4,
...     theory_psd=theory_psd,
... )
>>> sampler = MultivarBlockedNUTSSampler(fft_data, spline_model, config=config)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from ...logger import logger
from ..base_sampler import SamplerConfig
from ..utils import sample_pspline_block
from ..vi_init.adapters import prepare_block_vi
from .multivar_base import MultivarBaseSampler


def compute_noise_floor_sq(
    freqs: jnp.ndarray,
    block_j: int,
    mode: str,
    constant: float,
    scale: float,
    array: Optional[jnp.ndarray],
    theory_psd: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Compute innovation noise floor squared for a block.

    The constant value is interpreted as a variance-space floor (already
    squared).
    """
    n_freq = freqs.shape[0]
    if mode == "constant":
        floor_sq = jnp.full((n_freq,), float(constant), dtype=freqs.dtype)
    elif mode == "theory_scaled":
        if theory_psd is None:
            raise ValueError(
                "theory_psd is required when noise_floor_mode='theory_scaled'."
            )
        theory_psd = jnp.asarray(theory_psd, dtype=freqs.dtype)
        if theory_psd.shape[0] != n_freq:
            raise ValueError(
                f"theory_psd for block {block_j} has shape {theory_psd.shape}, expected ({n_freq},)."
            )
        floor_sq = jnp.asarray(scale, dtype=freqs.dtype) * theory_psd
    elif mode == "array":
        if array is None:
            raise ValueError(
                "noise_floor_array is required when noise_floor_mode='array'."
            )
        floor_sq = jnp.asarray(array, dtype=freqs.dtype)
        if floor_sq.shape[0] != n_freq:
            raise ValueError(
                f"noise_floor_array for block {block_j} has shape {floor_sq.shape}, expected ({n_freq},)."
            )
    else:
        raise ValueError(
            f"Unknown noise_floor_mode='{mode}'. Expected 'constant', 'theory_scaled', or 'array'."
        )
    return jnp.maximum(floor_sq, jnp.asarray(1e-30, dtype=freqs.dtype))


def _blocked_channel_model(
    channel_index: int,
    u_re_channel: jnp.ndarray,
    u_im_channel: jnp.ndarray,
    u_re_prev: jnp.ndarray,
    u_im_prev: jnp.ndarray,
    basis_delta: jnp.ndarray,
    penalty_delta: jnp.ndarray,
    basis_theta: jnp.ndarray,
    penalty_theta: jnp.ndarray,
    alpha_phi: float,
    beta_phi: float,
    alpha_phi_theta: float,
    beta_phi_theta: float,
    alpha_delta: float,
    beta_delta: float,
    nu: int,
    freq_weights: jnp.ndarray,
    apply_noise_floor: bool,
    noise_floor_sq: jnp.ndarray,
) -> None:
    """NumPyro model for a single Cholesky block (row of ``T``).

    Parameters
    ----------
    channel_index
        0‑based row index ``j`` in the Cholesky factor (channel in the data).
    u_re_channel, u_im_channel
        Real/imag parts of the eigenvector-weighted periodogram components for
        the active channel, shape ``(n_freq, n_rep)``.
    u_re_prev, u_im_prev
        Same components for the lower-triangular predecessors,
        shape ``(n_freq, channel_index, n_rep)``. The arrays can have zero size
        in the second dimension when ``channel_index == 0``.
    basis_delta, penalty_delta
        P‑spline basis/penalty for ``log δ_j(f)^2``.
    basis_theta, penalty_theta
        P‑spline basis/penalty shared by all θ_{jl}(f) in this row.
    alpha_phi, beta_phi, alpha_delta, beta_delta
        Hyperparameters for the hierarchical priors used in
        :func:`sample_pspline_block`.
    nu
        Degrees of freedom (number of averaged blocks) for determinant scaling.

    Notes
    -----
    - The likelihood implemented here corresponds to Eq. (likelihood_j) in your
      draft: the residual is ``u_j(f) = y_j(f) − Σ_{l<j} θ_{jl}(f) y_l(f)`` with
      ``y`` now replaced by the eigenvector-weighted replicates ``u``. The
      contribution to the log-likelihood is
      ``−ν Σ_k log δ_j(f_k)^2 − Σ_k ||u_j(f_k)||^2 / δ_j(f_k)^2`` up to constants.
    - When enabled, an innovation noise floor is added to ``δ_j(f)^2`` inside
      the likelihood terms to avoid variance collapse in near-null bands.
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
    # Numerical guard:
    # Very negative log_delta_sq implies extremely small variances, which can
    # overflow exp(-log_delta_sq) and yield inf-inf in the likelihood/ELBO.
    # Keep the range fairly tight to avoid exploding gradients during VI.
    log_delta_sq_safe = jnp.clip(log_delta_sq, a_min=-80.0, a_max=80.0)

    n_freq = u_re_channel.shape[0]
    n_theta_block = channel_index
    n_reps = u_re_channel.shape[1]

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
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
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
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
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

    delta_sq = jnp.exp(log_delta_sq_safe)
    if apply_noise_floor:
        # Innovation noise floor prevents variance collapse in near-null bands
        # and stabilizes the likelihood geometry.
        delta_eff_sq = delta_sq + noise_floor_sq
    else:
        delta_eff_sq = delta_sq
    fw = jnp.asarray(freq_weights, dtype=log_delta_sq.dtype)
    sum_log_det = -float(nu) * jnp.sum(fw * jnp.log(delta_eff_sq))

    if n_theta_block > 0:
        contrib_re = jnp.einsum(
            "fl,flr->fr", theta_re, u_re_prev
        ) - jnp.einsum("fl,flr->fr", theta_im, u_im_prev)
        contrib_im = jnp.einsum(
            "fl,flr->fr", theta_re, u_im_prev
        ) + jnp.einsum("fl,flr->fr", theta_im, u_re_prev)
        u_re_resid = u_re_channel - contrib_re
        u_im_resid = u_im_channel - contrib_im
    else:
        u_re_resid = u_re_channel
        u_im_resid = u_im_channel

    residual_power = u_re_resid**2 + u_im_resid**2
    # Sum across replicates then frequencies
    residual_power = jnp.sum(residual_power, axis=1)
    log_likelihood = sum_log_det - jnp.sum(residual_power / delta_eff_sq)

    numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)

    numpyro.deterministic(f"log_delta_sq_{channel_label}", log_delta_sq_safe)
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
    target_accept_prob_by_channel: Optional[Sequence[float]] = None
    max_tree_depth_by_channel: Optional[Sequence[int]] = None
    save_nuts_diagnostics: bool = True
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: Optional[str] = None
    vi_posterior_draws: int = 256
    vi_progress_bar: Optional[bool] = None

    # Optional separate hyperparameters for off-diagonal theta P-spline blocks.
    # When left as ``None`` they default to the diagonal hyperparameters
    # ``alpha_phi`` and ``beta_phi``.
    alpha_phi_theta: Optional[float] = None
    beta_phi_theta: Optional[float] = None

    # Innovation noise floor controls (variance-space floor added in likelihood).
    use_noise_floor: bool = False
    noise_floor_mode: str = "constant"
    noise_floor_constant: float = 0.0
    noise_floor_scale: float = 1e-4
    noise_floor_array: Optional[jnp.ndarray] = None
    theory_psd: Optional[jnp.ndarray] = None
    noise_floor_blocks: Sequence[int] | str = (3,)

    def __post_init__(self):
        super().__post_init__()
        if self.alpha_phi_theta is None:
            self.alpha_phi_theta = self.alpha_phi
        if self.beta_phi_theta is None:
            self.beta_phi_theta = self.beta_phi


class MultivarBlockedNUTSSampler(MultivarBaseSampler):
    """Run independent NUTS chains for each row of the Cholesky factor.

    For ``p`` channels the sampler iterates ``j = 0,…,p-1`` and runs a separate
    NumPyro/NUTS inference for the j‑th block using :func:`_blocked_channel_model`.
    The block observes the Wishart replicates ``(u_re[:, j, :], u_im[:, j, :])``
    and regresses against the corresponding lower-triangular components from
    previous channels.

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
        self._vi_diagnostics: Optional[Dict[str, Any]] = None

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

    def _get_channel_setting(
        self,
        name: str,
        channel_index: int,
        default_value: Any,
    ) -> Any:
        """Return per-channel override for a config field, with validation."""
        values = getattr(self.config, name, None)
        if values is None:
            return default_value
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"{name} must be a sequence of length {self.n_channels}, got {type(values).__name__}."
            )
        if len(values) != self.n_channels:
            raise ValueError(
                f"{name} must have length {self.n_channels}, got {len(values)}."
            )
        return values[channel_index]

    def _should_apply_noise_floor(self, channel_index: int) -> bool:
        if not self.config.use_noise_floor:
            return False
        blocks = self.config.noise_floor_blocks
        if isinstance(blocks, str):
            if blocks.lower() == "all":
                return True
            raise ValueError(
                f"noise_floor_blocks must be 'all' or a sequence of indices, got '{blocks}'."
            )
        return channel_index in set(blocks)

    def _get_noise_floor_args(
        self, channel_index: int
    ) -> tuple[bool, jnp.ndarray]:
        apply_noise_floor = self._should_apply_noise_floor(channel_index)
        if apply_noise_floor:
            noise_floor_sq = compute_noise_floor_sq(
                self.freq,
                channel_index,
                self.config.noise_floor_mode,
                self.config.noise_floor_constant,
                self.config.noise_floor_scale,
                self.config.noise_floor_array,
                self.config.theory_psd,
            )
        else:
            noise_floor_sq = jnp.zeros((self.n_freq,), dtype=self.freq.dtype)
        return apply_noise_floor, noise_floor_sq

    def sample(
        self,
        n_samples: int,
        n_warmup: int = 500,
        *,
        only_vi: bool = False,
        **kwargs,
    ) -> az.InferenceData:
        """Run the blocked inference and assemble results.

        Steps
        -----
        1. For each channel ``j`` provide the eigenvector replicates ``u`` to the
           single-row model and run an independent NUTS chain.
        2. Move per-block deterministics out of the ``samples`` dict into
           ``sample_stats`` and, for diagnostics, optionally rename NumPyro's
           NUTS fields with ``_channel_{j}`` suffixes.
        3. Stack ``log_delta_sq_j`` across channels and place the theta blocks into
           the correct positions of the global ``theta_re|im`` arrays.
        4. Sum the block log-likelihoods to obtain the joint log-likelihood.
        """
        logger.info(
            f"Blocked multivariate NUTS sampler [{self.device}] - {self.n_channels} channels"
        )

        combined_samples: Dict[str, np.ndarray] = {}
        combined_stats: Dict[str, np.ndarray] = {}

        channel_log_delta = []
        theta_re_total = None
        theta_im_total = None
        log_likelihood_total = None

        total_runtime = 0.0

        vi_only_mode = bool(only_vi or getattr(self.config, "only_vi", False))

        vi_setup = prepare_block_vi(
            self,
            rng_key=self.rng_key,
            block_model=_blocked_channel_model,
        )
        self._vi_diagnostics = vi_setup.diagnostics
        if self._vi_diagnostics:
            empirical_psd = (
                self._compute_empirical_psd()
                if self.config.outdir is not None
                else None
            )
            self._save_vi_diagnostics(empirical_psd=empirical_psd)
        self.rng_key = vi_setup.rng_key
        init_strategies = vi_setup.init_strategies
        mcmc_keys = vi_setup.mcmc_keys

        if vi_only_mode:
            return self._vi_only_inference_data(vi_setup.diagnostics)

        if (
            self.config.verbose
            and self._vi_diagnostics is not None
            and self._vi_diagnostics.get("losses") is not None
        ):
            losses = np.asarray(self._vi_diagnostics["losses"])
            if losses.size:
                guide = self._vi_diagnostics.get("guide", "vi")
                logger.info(
                    f"VI block init completed -> guide={guide}, final ELBO {float(losses[-1]):.3f}"
                )

        for channel_index in range(self.n_channels):
            delta_basis = self.all_bases[channel_index]
            delta_penalty = self.all_penalties[channel_index]

            theta_start = channel_index * (channel_index - 1) // 2
            theta_count = channel_index

            u_re_channel = self.u_re[:, channel_index, :]
            u_im_channel = self.u_im[:, channel_index, :]
            u_re_prev = self.u_re[:, :channel_index, :]
            u_im_prev = self.u_im[:, :channel_index, :]

            apply_noise_floor, noise_floor_sq = self._get_noise_floor_args(
                channel_index
            )

            target_accept_prob = self._get_channel_setting(
                "target_accept_prob_by_channel",
                channel_index,
                self.config.target_accept_prob,
            )
            max_tree_depth = self._get_channel_setting(
                "max_tree_depth_by_channel",
                channel_index,
                self.config.max_tree_depth,
            )
            kernel_kwargs = dict(
                target_accept_prob=float(target_accept_prob),
                max_tree_depth=int(max_tree_depth),
                dense_mass=bool(self.config.dense_mass),
            )
            init_strategy = init_strategies[channel_index]
            if init_strategy is not None:
                kernel_kwargs["init_strategy"] = init_strategy

            kernel = NUTS(_blocked_channel_model, **kernel_kwargs)

            mcmc = MCMC(
                kernel,
                num_warmup=n_warmup,
                num_samples=n_samples,
                num_chains=self.config.num_chains,
                chain_method=self.chain_method,
                progress_bar=self.config.verbose,
                jit_model_args=False,
            )

            start_time = time.time()
            mcmc.run(
                mcmc_keys[channel_index],
                channel_index,
                u_re_channel,
                u_im_channel,
                u_re_prev,
                u_im_prev,
                delta_basis,
                delta_penalty,
                self._theta_basis,
                self._theta_penalty,
                self.config.alpha_phi,
                self.config.beta_phi,
                self.config.alpha_phi_theta,
                self.config.beta_phi_theta,
                self.config.alpha_delta,
                self.config.beta_delta,
                self.nu,
                self.freq_weights,
                apply_noise_floor,
                noise_floor_sq,
                extra_fields=(
                    (
                        "potential_energy",
                        "energy",
                        "num_steps",
                        "adapt_state.step_size",
                        "accept_prob",
                        "diverging",
                    )
                    if self.config.save_nuts_diagnostics
                    else ()
                ),
            )
            total_runtime += time.time() - start_time

            block_samples = mcmc.get_samples(group_by_chain=True)
            for sample_key in list(block_samples):
                if sample_key.startswith("phi"):
                    block_samples[sample_key] = jnp.exp(
                        block_samples[sample_key]
                    )
            block_stats = mcmc.get_extra_fields(group_by_chain=True)

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
                diag_key_map = {
                    "potential_energy": "potential_energy",
                    "energy": "energy",
                    "num_steps": "num_steps",
                    "adapt_state.step_size": "step_size",
                    "accept_prob": "accept_prob",
                    "diverging": "diverging",
                }

                for diag_key, out_key in diag_key_map.items():
                    if diag_key in block_stats:
                        block_stats[f"{out_key}_channel_{channel_index}"] = (
                            block_stats.pop(diag_key)
                        )

            combined_samples.update(block_samples)

            log_delta_channel = block_stats.pop(
                f"log_delta_sq_{channel_index}"
            )
            channel_log_delta.append(log_delta_channel)

            if theta_count > 0 and self.n_theta > 0:
                theta_re_block = block_stats.pop(f"theta_re_{channel_index}")
                theta_im_block = block_stats.pop(f"theta_im_{channel_index}")

                if theta_re_total is None:
                    n_chains, n_draws = theta_re_block.shape[:2]
                    theta_re_total = jnp.zeros(
                        (n_chains, n_draws, self.n_freq, self.n_theta)
                    )
                    theta_im_total = jnp.zeros_like(theta_re_total)

                theta_slice = slice(theta_start, theta_start + theta_count)
                theta_re_total = theta_re_total.at[:, :, :, theta_slice].set(
                    theta_re_block
                )
                theta_im_total = theta_im_total.at[:, :, :, theta_slice].set(
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

        log_delta_sq = jnp.stack(channel_log_delta, axis=-1)

        if theta_re_total is None:
            theta_re_total = jnp.zeros(
                (log_delta_sq.shape[0], log_delta_sq.shape[1], self.n_freq, 0)
            )
            theta_im_total = jnp.zeros_like(theta_re_total)

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

    def _vi_only_inference_data(
        self, diagnostics: Optional[Dict[str, Any]]
    ) -> az.InferenceData:
        if not diagnostics:
            raise ValueError(
                "Variational-only mode is unavailable because VI diagnostics "
                "were not recorded. Ensure init_from_vi is True."
            )

        vi_samples = diagnostics.get("vi_samples")
        if vi_samples:
            sample_dict = {
                name: jnp.asarray(array)
                for name, array in vi_samples.items()
                if name.startswith(("weights_", "phi_", "delta_"))
            }
        else:
            raise ValueError(
                "Blocked VI diagnostics do not include posterior draws for the parameters."
            )

        if not sample_dict:
            raise ValueError(
                "No variational posterior draws were recorded for the model parameters."
            )

        samples = dict(sample_dict)
        for key in list(samples):
            if key.startswith("phi"):
                samples[key] = jnp.exp(samples[key])

        self.runtime = 0.0
        return self._create_vi_inference_data(samples, {}, diagnostics)
