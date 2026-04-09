from __future__ import annotations

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

import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from ...logger import logger
from ..base_sampler import SamplerConfig
from ..pspline_block import (
    build_log_density_fn,
    evaluate_log_density_batch,
    sample_pspline_block,
)
from ..vi_init.adapters import (
    _extract_multivar_design_psd,
    prepare_block_vi,
    prepare_coarse_block_vi,
)
from .multivar_base import MultivarBaseSampler


def _blocked_channel_model(
    channel_index: int,
    u_re_channel: jnp.ndarray,
    u_im_channel: jnp.ndarray,
    u_re_prev: jnp.ndarray,
    u_im_prev: jnp.ndarray,
    basis_delta: jnp.ndarray,
    penalty_delta: jnp.ndarray,
    basis_theta_re_by_component: tuple[jnp.ndarray, ...],
    penalty_theta_re_by_component: tuple[jnp.ndarray, ...],
    basis_theta_im_by_component: tuple[jnp.ndarray, ...],
    penalty_theta_im_by_component: tuple[jnp.ndarray, ...],
    alpha_phi: float,
    beta_phi: float,
    alpha_phi_theta: float,
    beta_phi_theta: float,
    alpha_delta: float,
    beta_delta: float,
    duration: float,
    Nb: int,
    Nh: int,
    design_weights: dict | None = None,
    tau: Optional[float] = None,
    enbw: float = 1.0,
    eta: float = 1.0,
) -> None:
    """NumPyro model for a single Cholesky block (row of ``T``).

    Parameters
    ----------
    channel_index
        0‑based row index ``j`` in the Cholesky factor (channel in the data).
    u_re_channel, u_im_channel
        Real/imag parts of the eigenvector-weighted periodogram components for
        the active channel, shape ``(N, n_rep)``.
    u_re_prev, u_im_prev
        Same components for the lower-triangular predecessors,
        shape ``(N, channel_index, n_rep)``. The arrays can have zero size
        in the second dimension when ``channel_index == 0``.
    basis_delta, penalty_delta
        P‑spline basis/penalty for ``log δ_j(f)^2``.
    basis_theta_re_by_component, penalty_theta_re_by_component
        P‑spline basis/penalty for each real theta component in this row.
    basis_theta_im_by_component, penalty_theta_im_by_component
        P‑spline basis/penalty for each imaginary theta component in this row.
    alpha_phi, beta_phi, alpha_delta, beta_delta
        Hyperparameters for the hierarchical priors used in
        :func:`sample_pspline_block`.
    Nb
        Degrees of freedom (number of averaged blocks) for determinant scaling.
    Nh
        Coarse-bin size multiplier for the determinant term.
    enbw
        Equivalent Noise Bandwidth of the analysis window (dimensionless,
        in units of frequency bins).  The full Whittle log-likelihood is
        divided by ``enbw`` so that the posterior is correctly calibrated
        regardless of window choice.  Rectangular window → ``enbw=1.0``
        (no change).  Hann window → ``enbw≈1.5``.
    eta
        Likelihood tempering exponent (η-posterior).  The log-likelihood
        is multiplied by ``eta`` before being passed to ``numpyro.factor``.
        ``eta=1.0`` recovers the standard (untempered) posterior.  Values
        ``0 < eta < 1`` flatten the likelihood surface, producing wider
        credible intervals.

        When the sampler config sets ``eta="auto"`` (the default), the
        resolved value passed here is ``min(1, n_basis * Nb * Nh / N_freq)``
        where ``n_basis`` is the number of B-spline basis functions,
        ``Nb`` the Wishart degrees of freedom, ``Nh`` the coarse-grain
        multiplicity, and ``N_freq`` the post-coarse-graining bin count.  This corrects for the Whittle pseudo-
        likelihood overstating Fisher information when ``N_freq`` >>
        ``n_basis`` (Grünwald "Safe Bayes" / generalized posterior).
        The correction is analogous to the ENBW factor: both account
        for the effective number of independent observations being
        smaller than the raw bin count.

    Notes
    -----
    - The likelihood implemented here corresponds to Eq. (likelihood_j) in paper
      draft: the residual is ``u_j(f) = y_j(f) − Σ_{l<j} θ_{jl}(f) y_l(f)`` The
      contribution to the log-likelihood is
      ``−ν Σ_k log δ_j(f_k)^2 − Σ_k ||u_j(f_k)||^2 / δ_j(f_k)^2`` up to constants.
    - Deterministic nodes record the evaluated spline fields so downstream code
      can reconstruct the PSD matrix without re-evaluating the splines.
    """

    channel_label = f"{channel_index}"
    _dw = design_weights or {}

    delta_block = sample_pspline_block(
        delta_name=f"delta_{channel_label}",
        phi_name=f"phi_delta_{channel_label}",
        weights_name=f"weights_delta_{channel_label}",
        penalty_matrix=penalty_delta,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        w_design=_dw.get(f"delta_{channel_index}"),
        tau=tau,
    )
    log_delta_sq = jnp.einsum("nk,k->n", basis_delta, delta_block["weights"])
    # Numerical guard:
    # Very negative log_delta_sq implies extremely small variances, which can
    # overflow exp(-log_delta_sq) and yield inf-inf in the likelihood/ELBO.
    # Keep the range fairly tight to avoid exploding gradients during VI.
    log_delta_sq_safe = jnp.clip(log_delta_sq, a_min=-80.0, a_max=80.0)

    N = u_re_channel.shape[0]
    n_theta_block = channel_index
    n_reps = u_re_channel.shape[1]

    if n_theta_block > 0:
        theta_re_components = []
        theta_im_components = []

        for theta_idx in range(n_theta_block):
            basis_theta_re = basis_theta_re_by_component[theta_idx]
            penalty_theta_re = penalty_theta_re_by_component[theta_idx]
            theta_prefix = f"theta_re_{channel_label}_{theta_idx}"
            theta_re_block = sample_pspline_block(
                delta_name=f"delta_{theta_prefix}",
                phi_name=f"phi_{theta_prefix}",
                weights_name=f"weights_{theta_prefix}",
                penalty_matrix=penalty_theta_re,
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
                w_design=_dw.get(f"theta_re_{channel_index}_{theta_idx}"),
                tau=tau,
            )
            theta_re_eval = jnp.einsum(
                "nk,k->n", basis_theta_re, theta_re_block["weights"]
            )
            theta_re_components.append(theta_re_eval)

            basis_theta_im = basis_theta_im_by_component[theta_idx]
            penalty_theta_im = penalty_theta_im_by_component[theta_idx]
            theta_im_prefix = f"theta_im_{channel_label}_{theta_idx}"
            theta_im_block = sample_pspline_block(
                delta_name=f"delta_{theta_im_prefix}",
                phi_name=f"phi_{theta_im_prefix}",
                weights_name=f"weights_{theta_im_prefix}",
                penalty_matrix=penalty_theta_im,
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
                w_design=_dw.get(f"theta_im_{channel_index}_{theta_idx}"),
                tau=tau,
            )

            theta_im_eval = jnp.einsum(
                "nk,k->n", basis_theta_im, theta_im_block["weights"]
            )
            theta_im_components.append(theta_im_eval)

        theta_re = jnp.stack(theta_re_components, axis=1)
        theta_im = jnp.stack(theta_im_components, axis=1)
    else:
        theta_re = jnp.zeros((N, 0))
        theta_im = jnp.zeros((N, 0))

    delta_eff_sq = jnp.exp(log_delta_sq_safe)
    nh = jnp.asarray(Nh, dtype=log_delta_sq.dtype)
    sum_log_det = -float(Nb) * nh * jnp.sum(jnp.log(delta_eff_sq))

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
    # Sum across Wishart replicates; for coarse bins, this reflects the sum of
    # sufficient statistics across all fine-grid frequencies in that bin.
    residual_power_sum = jnp.sum(residual_power, axis=1)
    duration_scale = jnp.asarray(duration, dtype=log_delta_sq.dtype)
    log_likelihood = sum_log_det - jnp.sum(
        residual_power_sum / (duration_scale * delta_eff_sq)
    )
    # ENBW correction: dividing the full log-likelihood by the Equivalent Noise
    # Bandwidth of the analysis window restores posterior calibration when a
    # tapered (e.g. Hann) window is used.  Adjacent DFT bins are correlated with
    # correlation ≈ 1 - 1/enbw, so the Whittle likelihood overstates the
    # effective Fisher information by a factor of enbw.  Scaling by 1/enbw
    # keeps the MLE unchanged (both terms scale equally) while widening the
    # posterior by enbw, recovering the correct frequentist coverage.
    # For a rectangular window enbw=1.0, so this is a strict no-op.
    log_likelihood = log_likelihood / jnp.asarray(
        enbw, dtype=log_delta_sq.dtype
    )
    # η-tempering: scale the log-likelihood by eta ∈ (0, 1] to produce a
    # generalized posterior.  eta=1.0 is the standard (untempered) case.
    log_likelihood = log_likelihood * jnp.asarray(
        eta, dtype=log_delta_sq.dtype
    )

    numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)

    # log_delta_sq and theta_re/im are NOT registered as deterministic sites:
    # doing so would cause JAX's lax.scan to preallocate (chains, draws, N_freq)
    # buffers for every channel, consuming several GB at N_freq=8192.
    # These fields are reconstructed from posterior weight samples in to_arviz.py.
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
    compute_lnz: bool = False
    target_accept_prob_by_channel: Optional[Sequence[float]] = None
    max_tree_depth_by_channel: Optional[Sequence[int]] = None
    save_nuts_diagnostics: bool = True
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: Optional[str] = None
    vi_posterior_draws: int = 50
    vi_progress_bar: Optional[bool] = None

    # Optional separate hyperparameters for off-diagonal theta P-spline blocks.
    # When left as ``None`` they default to the diagonal hyperparameters
    # ``alpha_phi`` and ``beta_phi``.
    alpha_phi_theta: Optional[float] = None
    beta_phi_theta: Optional[float] = None

    # Soft shrinkage toward a design PSD.
    # ``design_psd``: complex array (N, p, p) at model frequencies.  When
    # provided, each P-spline component is shrunk toward the spline fit to the
    # corresponding Cholesky component of the design.
    # ``tau``: isotropic Gaussian scale for level shrinkage.  ``None`` means no
    # additional L2 term beyond the smoothness penalty.
    design_psd: Optional[np.ndarray] = None
    tau: Optional[float] = None
    design_from_vi: bool = False
    design_from_vi_tau: float = 10.0

    # η-tempering (generalized posterior / Safe Bayes correction).
    # Corrects over-concentration of the Whittle pseudo-likelihood when the
    # smooth P-spline model has far fewer effective parameters than frequency
    # bins.  The Whittle likelihood treats each coarse-grained bin as
    # carrying Nb×Nh independent Wishart replications, inflating the Fisher
    # information far beyond what the smooth P-spline model can resolve.
    #
    # - ``"auto"`` (default): η = min(1, c / (Nb × Nh)), where c is set by
    #   ``eta_c``.  Empirically validated on VAR(2) simulations (coverage
    #   study, seeds 0-99, multiple Nb/Nh configurations).
    # - ``1.0``: no correction (standard Whittle posterior, legacy behaviour).
    # - Any float in (0, 1]: manual override for investigation.
    eta: float | str = "auto"
    # Scaling constant for the auto-eta formula: η = min(1, eta_c / (Nb×Nh)).
    # Default c=2 targets ~90% coverage.  Increase for wider CIs (more
    # conservative), decrease for tighter CIs.
    eta_c: float = 2.0

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

        self._design_weights: dict = {}
        if self.config.design_psd is not None:
            design_psd = self._align_design_psd(self.config.design_psd)
            self._design_weights = self.spline_model.compute_design_weights(
                design_psd
            )

    def _theta_component_arrays_for_channel(
        self,
        channel_index: int,
        *,
        part: str,
    ) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
        """Return per-theta basis/penalty tuples for one blocked channel."""
        if channel_index <= 0 or self.n_theta == 0:
            return tuple(), tuple()

        bases: list[jnp.ndarray] = []
        penalties: list[jnp.ndarray] = []
        for theta_idx in range(channel_index):
            model = self.spline_model.get_theta_model(
                part, channel_index, theta_idx
            )
            bases.append(jnp.asarray(model.basis, dtype=jnp.float32))
            penalties.append(jnp.asarray(model.penalty_matrix))
        return tuple(bases), tuple(penalties)

    def _align_design_psd(
        self,
        design_psd,
    ) -> np.ndarray:
        """Return design PSD interpolated to model frequencies.

        Accepts either:
        - ``np.ndarray`` of shape ``(N, p, p)`` already at model frequencies.
        - A tuple ``(freqs, psd)`` where ``freqs`` is shape ``(M,)`` and
          ``psd`` is shape ``(M, p, p)``; each element is interpolated to
          ``self.fft_data.freq``.
        """
        model_freq = np.asarray(self.fft_data.freq)
        if isinstance(design_psd, tuple):
            src_freq, src_psd = design_psd
            src_freq = np.asarray(src_freq)
            src_psd = np.asarray(src_psd, dtype=np.complex128)
            aligned = np.stack(
                [
                    [
                        np.interp(model_freq, src_freq, src_psd[:, j, l].real)
                        + 1j
                        * np.interp(
                            model_freq, src_freq, src_psd[:, j, l].imag
                        )
                        for l in range(self.p)
                    ]
                    for j in range(self.p)
                ],
                axis=0,
            )  # (p, p, N)
            design_psd = np.moveaxis(
                aligned, [0, 1, 2], [1, 2, 0]
            )  # (N, p, p)
        else:
            design_psd = np.asarray(design_psd, dtype=np.complex128)
            if design_psd.shape != (self.N, self.p, self.p):
                raise ValueError(
                    f"design_psd array must have shape ({self.N}, {self.p}, {self.p}), "
                    f"got {design_psd.shape}. "
                    "Pass a (freqs, psd) tuple to enable automatic interpolation."
                )

        # The model operates on channel-standardized data.  Divide the design
        # PSD by the outer product of channel stds so design_weights correspond
        # to the standardized parameterization that NUTS samples from.
        channel_stds = getattr(self.config, "channel_stds", None)
        if channel_stds is not None:
            stds = np.asarray(channel_stds, dtype=np.float64)
            scale_matrix = np.outer(stds, stds)  # (p, p)
            design_psd = design_psd / scale_matrix[np.newaxis, :, :]

        return design_psd

    def _set_design_weights_from_vi(self) -> None:
        """Use VI posterior mean PSD as the design prior."""
        vi_psd = _extract_multivar_design_psd(self._vi_diagnostics)
        if vi_psd is None:
            logger.warning(
                "design_from_vi enabled but VI did not produce a valid PSD; "
                "falling back to no design prior."
            )
            return

        self._design_weights = self.spline_model.compute_design_weights(vi_psd)
        if self.config.tau is None:
            self.config.tau = self.config.design_from_vi_tau
        logger.info(
            "design_from_vi: using VI posterior mean as design prior (tau=%.2f).",
            self.config.tau,
        )

    @property
    def sampler_type(self) -> str:
        return "multivariate_blocked_nuts"

    def _resolve_eta(self, channel_index: int) -> float:
        return self._resolve_eta_value(self.config.eta, channel_index)

    def _resolve_eta_value(
        self,
        eta_value: float | str | None,
        channel_index: int,
    ) -> float:
        """Resolve the eta config value to a concrete float for a channel.

        When ``eta="auto"``, compute the correction factor from model
        dimensions using the empirically validated scaling rule::

            η = min(1, c / (Nb × Nh))

        where ``c`` is ``config.eta_c`` (default 2), ``Nb`` is the Bartlett
        segment count (Wishart DOF), and ``Nh`` is the coarse-graining
        multiplicity.  The product ``Nb × Nh`` controls the effective sample
        size per coarse-grained frequency bin in the Whittle likelihood;
        dividing by it re-calibrates the posterior width to achieve nominal
        coverage.

        This formula was validated on VAR(2) simulations across multiple
        ``Nb``, ``Nh``, and ``n_basis`` configurations.  Notably, ``n_basis``
        is *not* a first-order term — the dominant over-concentration scales
        with the data replication factor ``Nb × Nh``, not the model dimension.
        """
        raw = self.config.eta if eta_value is None else eta_value
        if isinstance(raw, str) and raw == "auto":
            c = self.config.eta_c
            eta = min(1.0, c / float(self.Nb * self.Nh))
            if channel_index == 0 and eta_value in (None, self.config.eta):
                logger.info(
                    f"eta='auto': c={c}, Nb={self.Nb}, Nh={self.Nh} -> "
                    f"eta = min(1, {c}/({self.Nb}*{self.Nh})) = {eta:.4f}",
                )
            return eta
        return float(raw)

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
                f"{name} must be a sequence of length {self.p}, got {type(values).__name__}."
            )
        if len(values) != self.p:
            raise ValueError(
                f"{name} must have length {self.p}, got {len(values)}."
            )
        return values[channel_index]

    def _reset_lnz_details(self) -> None:
        super()._reset_lnz_details()

    def _channel_model(self):
        return _blocked_channel_model

    def _lnz_build_log_density_fn(self):
        return build_log_density_fn

    def _lnz_evaluate_log_density_batch(self):
        return evaluate_log_density_batch

    def _channel_model_kwargs(self, channel_index: int) -> Dict[str, Any]:
        theta_re_basis, theta_re_penalty = (
            self._theta_component_arrays_for_channel(channel_index, part="re")
        )
        theta_im_basis, theta_im_penalty = (
            self._theta_component_arrays_for_channel(channel_index, part="im")
        )
        return {
            "channel_index": channel_index,
            "u_re_channel": self.u_re[:, channel_index, :],
            "u_im_channel": self.u_im[:, channel_index, :],
            "u_re_prev": self.u_re[:, :channel_index, :],
            "u_im_prev": self.u_im[:, :channel_index, :],
            "basis_delta": self.all_bases[channel_index],
            "penalty_delta": self.all_penalties[channel_index],
            "basis_theta_re_by_component": theta_re_basis,
            "penalty_theta_re_by_component": theta_re_penalty,
            "basis_theta_im_by_component": theta_im_basis,
            "penalty_theta_im_by_component": theta_im_penalty,
            "alpha_phi": float(self.config.alpha_phi),
            "beta_phi": float(self.config.beta_phi),
            "alpha_phi_theta": float(self.config.alpha_phi_theta),
            "beta_phi_theta": float(self.config.beta_phi_theta),
            "alpha_delta": float(self.config.alpha_delta),
            "beta_delta": float(self.config.beta_delta),
            "duration": float(self.duration),
            "Nb": int(self.Nb),
            "Nh": int(self.Nh),
            "enbw": float(self.enbw),
            "eta": self._resolve_eta(channel_index),
            "design_weights": self._design_weights,
            "tau": self.config.tau,
        }

    def _channel_parameter_names(self, channel_index: int) -> list[str]:
        names = [
            f"weights_delta_{channel_index}",
            f"phi_delta_{channel_index}",
            f"delta_{channel_index}",
        ]
        for theta_idx in range(channel_index):
            for theta_prefix in ("theta_re", "theta_im"):
                names.extend(
                    [
                        f"weights_{theta_prefix}_{channel_index}_{theta_idx}",
                        f"phi_{theta_prefix}_{channel_index}_{theta_idx}",
                        f"delta_{theta_prefix}_{channel_index}_{theta_idx}",
                    ]
                )
        return names

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
          3. Sum the block log-likelihoods to obtain the joint log-likelihood.
        """
        logger.info(
            f"Blocked multivariate NUTS sampler [{self.device}] - {self.p} channels"
        )
        self._reset_lnz_details()

        combined_samples: Dict[str, np.ndarray] = {}
        combined_stats: Dict[str, np.ndarray] = {}
        warmup_attrs: Dict[str, float | int] = {}

        log_likelihood_total = None

        total_runtime = 0.0

        vi_only_mode = bool(only_vi or getattr(self.config, "only_vi", False))
        coarse_sampler = getattr(self, "_coarse_vi_sampler", None)
        if coarse_sampler is not None:
            vi_setup = prepare_coarse_block_vi(
                self,
                coarse_sampler=coarse_sampler,
                block_model=_blocked_channel_model,
            )
        else:
            vi_setup = prepare_block_vi(
                self,
                rng_key=cast(jax.Array, self.rng_key),
                block_model=_blocked_channel_model,
            )
        self._vi_diagnostics = vi_setup.diagnostics
        vi_diag = self._vi_diagnostics or {}
        self._extra_idata_attrs = {
            key: vi_diag[key]
            for key in (
                "coarse_vi_attempted",
                "coarse_vi_success",
                "coarse_vi_mode",
                "coarse_vi_full_nfreq",
                "coarse_vi_nfreq",
                "coarse_vi_target_nfreq",
            )
            if key in vi_diag
        }
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

        if self.config.design_from_vi and not self._design_weights:
            self._set_design_weights_from_vi()

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

        for channel_index in range(self.p):
            delta_basis = self.all_bases[channel_index]
            delta_penalty = self.all_penalties[channel_index]
            theta_re_basis, theta_re_penalty = (
                self._theta_component_arrays_for_channel(
                    channel_index, part="re"
                )
            )
            theta_im_basis, theta_im_penalty = (
                self._theta_component_arrays_for_channel(
                    channel_index, part="im"
                )
            )

            theta_start = channel_index * (channel_index - 1) // 2
            theta_count = channel_index

            u_re_channel = self.u_re[:, channel_index, :]
            u_im_channel = self.u_im[:, channel_index, :]
            u_re_prev = self.u_re[:, :channel_index, :]
            u_im_prev = self.u_im[:, :channel_index, :]

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
            kernel_kwargs: dict[str, Any] = dict(
                target_accept_prob=float(target_accept_prob),
                max_tree_depth=int(max_tree_depth),
                dense_mass=bool(self.config.dense_mass),
            )
            init_strategy = init_strategies[channel_index]
            if init_strategy is not None:
                kernel_kwargs["init_strategy"] = init_strategy

            model_args = (
                channel_index,
                u_re_channel,
                u_im_channel,
                u_re_prev,
                u_im_prev,
                delta_basis,
                delta_penalty,
                theta_re_basis,
                theta_re_penalty,
                theta_im_basis,
                theta_im_penalty,
                self.config.alpha_phi,
                self.config.beta_phi,
                self.config.alpha_phi_theta,
                self.config.beta_phi_theta,
                self.config.alpha_delta,
                self.config.beta_delta,
                self.duration,
                self.Nb,
                self.Nh,
                self._design_weights or None,
                self.config.tau,
                self.enbw,
            )

            extra_fields = (
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
            )

            sample_eta = self._resolve_eta(channel_index)
            warmup_attrs[f"sampling_eta_channel_{channel_index}"] = float(
                sample_eta
            )

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
                *model_args,
                sample_eta,
                extra_fields=extra_fields,
            )
            total_runtime += time.time() - start_time

            block_samples = mcmc.get_samples(group_by_chain=True)
            for sample_key in list(block_samples):
                if sample_key.startswith("phi"):
                    block_samples[sample_key] = jnp.exp(
                        block_samples[sample_key]
                    )
            block_stats = mcmc.get_extra_fields(group_by_chain=True)

            # Move only the scalar log_likelihood_block deterministic from
            # block_samples into block_stats.  The large per-frequency
            # deterministics (log_delta_sq, theta_re, theta_im) are no longer
            # registered in the model, so they won't appear here.
            ll_block_key = f"log_likelihood_block_{channel_index}"
            if ll_block_key in block_samples:
                block_stats[ll_block_key] = block_samples.pop(ll_block_key)

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

            block_log_likelihood = block_stats.pop(
                f"log_likelihood_block_{channel_index}"
            )
            combined_stats[f"log_likelihood_block_{channel_index}"] = (
                np.asarray(block_log_likelihood)
            )

            log_likelihood_total = (
                block_log_likelihood
                if log_likelihood_total is None
                else log_likelihood_total + block_log_likelihood
            )

            combined_stats.update(block_stats)

        self.runtime = total_runtime
        if warmup_attrs:
            existing_attrs = dict(
                getattr(self, "_extra_idata_attrs", {}) or {}
            )
            existing_attrs.update(warmup_attrs)
            self._extra_idata_attrs = existing_attrs

        combined_stats.update(
            {
                "log_likelihood": np.asarray(log_likelihood_total),
            }
        )

        return self.to_arviz(combined_samples, combined_stats)

    def _vi_only_inference_data(
        self, diagnostics: Optional[Dict[str, Any]]
    ) -> az.InferenceData:
        self._reset_lnz_details()
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
        idata = self._create_vi_inference_data(samples, {}, diagnostics)
        self._cache_full_diagnostics(idata)
        return idata
