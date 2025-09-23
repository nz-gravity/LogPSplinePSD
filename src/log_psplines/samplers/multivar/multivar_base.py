"""
Base class for multivariate PSD samplers.
"""

import tempfile
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray, Dataset
from dataclasses import asdict

from ...datatypes import MultivarFFT
from ...plotting import plot_psd_matrix, compute_empirical_psd
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def batch_spline_eval(basis: jnp.ndarray, weights_batch: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors.

    Args:
        basis: Basis matrix (n_freq, n_basis)
        weights_batch: Batch of weight vectors (n_samples, n_basis)

    Returns:
        Batch of spline evaluations (n_samples, n_freq)
    """
    return jnp.sum(basis[None, :, :] * weights_batch[:, None, :], axis=-1)


@jax.jit
def whittle_likelihood(
    y_re: jnp.ndarray,  # (n_freq, n_dim)
    y_im: jnp.ndarray,  # (n_freq, n_dim)
    Z_re: jnp.ndarray,  # (n_freq, n_dim, n_theta)
    Z_im: jnp.ndarray,  # (n_freq, n_dim, n_theta)
    log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
    theta_re: jnp.ndarray,      # (n_freq, n_theta)
    theta_im: jnp.ndarray       # (n_freq, n_theta)
) -> jnp.ndarray:
    """Multivariate Whittle likelihood for Cholesky PSD parameterization."""
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)

    if Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum('fij,fj->fi', Z_re, theta_re) - jnp.einsum('fij,fj->fi', Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', Z_re, theta_im) + jnp.einsum('fij,fj->fi', Z_im, theta_re)
        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    numerator = u_re ** 2 + u_im ** 2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2


class MultivarBaseSampler(BaseSampler):
    """
    Base class for multivariate PSD samplers.

    Handles multi-channel FFT data with MultivariateLogPSplines models using
    Cholesky parameterization for cross-spectral density estimation.
    """

    def __init__(
        self,
        fft_data: MultivarFFT,
        spline_model: MultivariateLogPSplines,
        config: SamplerConfig
    ):
        # Type hints for clarity
        self.fft_data: MultivarFFT = fft_data
        self.spline_model: MultivariateLogPSplines = spline_model

        super().__init__(fft_data, spline_model, config)

    def _setup_data(self) -> None:
        """Setup multivariate-specific data attributes."""
        self.n_freq = self.fft_data.n_freq
        self.n_channels = self.fft_data.n_dim
        self.n_theta = self.spline_model.n_theta

        # Get all bases and penalties for NumPyro model
        self.all_bases, self.all_penalties = self.spline_model.get_all_bases_and_penalties()

        # FFT data arrays for JAX operations
        self.y_re = jnp.array(self.fft_data.y_re)
        self.y_im = jnp.array(self.fft_data.y_im)
        self.Z_re = jnp.array(self.fft_data.Z_re)
        self.Z_im = jnp.array(self.fft_data.Z_im)
        self.freq = jnp.array(self.fft_data.freq)

        # Create MultivarFFT object for JAX functions
        self.data_jax = MultivarFFT(
            y_re=self.y_re,
            y_im=self.y_im,
            Z_re=self.Z_re,
            Z_im=self.Z_im,
            freq=self.freq,
            n_freq=self.n_freq,
            n_dim=self.n_channels
        )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _create_inference_data(
        self,
        samples: Dict[str, jnp.ndarray],
        sample_stats: Dict[str, Any],
        lnz: float,
        lnz_err: float
    ) -> az.InferenceData:
        """Create InferenceData for multivariate case."""
        
        # Ensure all arrays have chain dimension - FIXED VERSION
        def add_chain_dim(data_dict):
            result = {}
            for k, v in data_dict.items():
                v_array = np.array(v)
                # Always add chain dimension as first dimension if not present
                # NumPyro with 1 chain returns samples without chain dimension
                if v_array.ndim == 1:  # Scalar parameters: (n_draws,) -> (1, n_draws)
                    result[k] = v_array[None, :]
                elif v_array.ndim == 2:  # Vector parameters: (n_draws, dim) -> (1, n_draws, dim)
                    result[k] = v_array[None, :, :]
                elif v_array.ndim == 3:  # Matrix parameters: (n_draws, dim1, dim2) -> (1, n_draws, dim1, dim2)
                    result[k] = v_array[None, :, :, :]
                elif v_array.ndim == 4:  # Already has chain dim: (n_chains, n_draws, dim1, dim2)
                    result[k] = v_array
                else:
                    # Handle higher dimensional cases
                    result[k] = v_array[None, ...]
            return result

        samples = add_chain_dim(samples)
        sample_stats = add_chain_dim(sample_stats)

        # Extract dimensions from a standard sample
        first_sample_key = next((k for k, v in samples.items() if "weights_" in k), next(iter(samples.keys())))
        sample_shape = samples[first_sample_key].shape
        n_chains, n_draws = sample_shape[:2]

        # Create posterior predictive samples
        psd_samples = self._compute_posterior_predictive(samples)
        n_pp = psd_samples.shape[0]

        # Coordinates
        coords = {
            "chain": range(n_chains),
            "draw": range(n_draws),
            "pp_draw": range(n_pp),
            "freq": np.array(self.freq),
            "channels": range(self.n_channels),
        }

        # Dimensions for each data group
        dims = {}

        # Posterior samples - handle weights with proper dimensions
        for key, array in samples.items():
            array_shape = array.shape
            if key.startswith("weights_"):
                # Extract the component type from the key (e.g., "delta_0", "theta_re")
                component = key[8:]  # Remove "weights_" prefix
                dims[key] = ["chain", "draw", f"{component}_basis_dim"]
                coords[f"{component}_basis_dim"] = range(array_shape[-1])
            elif key in ["phi", "delta"] or key.startswith(("phi_", "delta_")):
                # Scalar hyperparameters
                dims[key] = ["chain", "draw"]

        # Sample stats - handle multivariate-specific variables properly
        for key, array in sample_stats.items():
            array_shape = array.shape
            
            if key == "log_delta_sq":
                # Should now have shape: (1, n_draws, n_freq, n_channels)
                if len(array_shape) == 4:
                    dims[key] = ["chain", "draw", "freq", "channels"]
                else:
                    # Fallback for unexpected shapes
                    dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(len(array_shape)-2)]
                    for i in range(2, len(array_shape)):
                        coords[f"{key}_dim_{i-2}"] = range(array_shape[i])
                        
            elif key in ["theta_re", "theta_im"]:
                # Should have shape: (1, n_draws, n_freq, n_theta)
                if len(array_shape) == 4:
                    dims[key] = ["chain", "draw", "freq", f"{key}_theta_dim"]
                    coords[f"{key}_theta_dim"] = range(array_shape[-1])
                else:
                    # Fallback
                    dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(len(array_shape)-2)]
                    for i in range(2, len(array_shape)):
                        coords[f"{key}_dim_{i-2}"] = range(array_shape[i])
                        
            elif key == "log_likelihood":
                # Scalar likelihood: should be (1, n_draws)
                dims[key] = ["chain", "draw"]
            else:
                # Generic handling for other sample stats
                nd = len(array_shape)
                if nd == 2:
                    dims[key] = ["chain", "draw"]
                else:
                    dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(nd-2)]
                    for i in range(2, nd):
                        coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

        # Observed data
        dims.update({
            "fft_re": ["freq", "channels"],
            "fft_im": ["freq", "channels"],
            # Posterior predictive
            "psd_matrix": ["pp_draw", "freq", "channels", "channels"],
        })

        # Add log posterior if both likelihood and prior exist
        if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
            sample_stats["lp"] = (
                sample_stats["log_likelihood"] + sample_stats["log_prior"]
            )
            dims["lp"] = ["chain", "draw"]

        # Attributes
        attributes = dict(
            device=str(self.device),
            runtime=self.runtime,
            lnz=lnz,
            lnz_err=lnz_err,
            sampler_type=self.sampler_type,
            data_type=self.data_type,
            n_channels=self.n_channels,
            n_freq=self.n_freq,
            n_theta=self.n_theta,
            frequencies=np.array(self.freq),
        )

        # Convert config to attributes (handle booleans)
        config_attrs = {
            k: int(v) if isinstance(v, bool) else v
            for k, v in asdict(self.config).items()
        }
        attributes.update(config_attrs)

        # Add ESS calculation
        try:
            attributes.update(dict(ess=az.ess(samples).to_array().values.flatten()))
        except:
            attributes.update(dict(ess=[]))

        # Create InferenceData
        idata = az.from_dict(
            posterior=samples,
            sample_stats=sample_stats,
            observed_data={
                "fft_re": np.array(self.fft_data.y_re),
                "fft_im": np.array(self.fft_data.y_im)
            },
            dims=dims,
            coords=coords,
            attrs=attributes,
        )

        # Add posterior predictive samples
        idata.add_groups(
            posterior_psd=Dataset(
                {
                    "psd_matrix": DataArray(psd_samples, dims=["pp_draw", "freq", "channels", "channels"]),
                },
                coords={
                    "pp_draw": coords["pp_draw"],
                    "freq": coords["freq"],
                    "channels": coords["channels"],
                },
            )
        )

        # Add spline model info
        idata.add_groups(spline_model=self._pack_spline_model())
        return idata

    def _pack_spline_model(self) -> Dataset:
        """Pack multivariate spline model parameters into xarray Dataset."""
        data = {
            "degree": self.spline_model.degree,
            "diffMatrixOrder": self.spline_model.diffMatrixOrder,
            "n_freq": self.spline_model.n_freq,
            "n_channels": self.spline_model.n_channels,
            "n_theta": self.spline_model.n_theta,
        }

        coords = {}

        # Diagonal models
        for i, diag_model in enumerate(self.spline_model.diagonal_models):
            prefix = f"diag_{i}"
            data.update({
                f"{prefix}_knots": ([f"{prefix}_knots_dim"], np.array(diag_model.knots)),
                f"{prefix}_basis": ([f"{prefix}_freq", f"{prefix}_weights_dim"], np.array(diag_model.basis)),
                f"{prefix}_penalty_matrix": ([f"{prefix}_weights_dim_row", f"{prefix}_weights_dim_col"], np.array(diag_model.penalty_matrix)),
                f"{prefix}_parametric_model": ([f"{prefix}_freq"], np.array(diag_model.parametric_model)),
            })
            coords.update({
                f"{prefix}_knots_dim": np.arange(len(diag_model.knots)),
                f"{prefix}_weights_dim": np.arange(diag_model.basis.shape[1]),
                f"{prefix}_weights_dim_row": np.arange(diag_model.penalty_matrix.shape[0]),
                f"{prefix}_weights_dim_col": np.arange(diag_model.penalty_matrix.shape[1]),
                f"{prefix}_freq": np.arange(diag_model.basis.shape[0]),
            })

        # Off-diagonal models
        if self.spline_model.offdiag_re_model is not None:
            data.update({
                "offdiag_re_knots": (["offdiag_knots_dim"], np.array(self.spline_model.offdiag_re_model.knots)),
                "offdiag_re_basis": (["offdiag_freq", "offdiag_weights_dim"], np.array(self.spline_model.offdiag_re_model.basis)),
                "offdiag_re_penalty_matrix": (["offdiag_weights_dim_row", "offdiag_weights_dim_col"], np.array(self.spline_model.offdiag_re_model.penalty_matrix)),
                "offdiag_re_parametric_model": (["offdiag_freq"], np.array(self.spline_model.offdiag_re_model.parametric_model)),
            })
            coords.update({
                "offdiag_knots_dim": np.arange(len(self.spline_model.offdiag_re_model.knots)),
                "offdiag_weights_dim": np.arange(self.spline_model.offdiag_re_model.basis.shape[1]),
                "offdiag_weights_dim_row": np.arange(self.spline_model.offdiag_re_model.penalty_matrix.shape[0]),
                "offdiag_weights_dim_col": np.arange(self.spline_model.offdiag_re_model.penalty_matrix.shape[1]),
                "offdiag_freq": np.arange(self.spline_model.offdiag_re_model.basis.shape[0]),
            })

        if self.spline_model.offdiag_im_model is not None:
            data.update({
                "offdiag_im_knots": (["offdiag_knots_dim"], np.array(self.spline_model.offdiag_im_model.knots)),
                "offdiag_im_basis": (["offdiag_freq", "offdiag_weights_dim"], np.array(self.spline_model.offdiag_im_model.basis)),
                "offdiag_im_penalty_matrix": (["offdiag_weights_dim_row", "offdiag_weights_dim_col"], np.array(self.spline_model.offdiag_im_model.penalty_matrix)),
                "offdiag_im_parametric_model": (["offdiag_freq"], np.array(self.spline_model.offdiag_im_model.parametric_model)),
            })
            coords.update({
                "offdiag_knots_dim": np.arange(len(self.spline_model.offdiag_im_model.knots)),
                "offdiag_weights_dim": np.arange(self.spline_model.offdiag_im_model.basis.shape[1]),
                "offdiag_weights_dim_row": np.arange(self.spline_model.offdiag_im_model.penalty_matrix.shape[0]),
                "offdiag_weights_dim_col": np.arange(self.spline_model.offdiag_im_model.penalty_matrix.shape[1]),
                "offdiag_freq": np.arange(self.spline_model.offdiag_im_model.basis.shape[0]),
            })

        return Dataset(
            {
                k: (
                    DataArray(v[1], dims=v[0])
                    if isinstance(v, tuple)
                    else DataArray(v)
                )
                for k, v in data.items()
            },
            coords=coords,
        )

    def _compute_posterior_predictive(self, samples: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute posterior predictive PSD matrices from samples."""
        # Extract samples - handle different possible sample structures
        if "log_delta_sq" in samples:
            log_delta_sq = samples["log_delta_sq"]
            # Remove chain dimension if present (take first chain only for posterior predictive)
            if log_delta_sq.ndim == 4:  # (n_chains, n_draws, n_freq, n_channels)
                log_delta_sq = log_delta_sq[0]  # Take first chain: (n_draws, n_freq, n_channels)
            theta_re = samples.get("theta_re")
            theta_im = samples.get("theta_im")
            # Default to zeros if not present
            if theta_re is None:
                theta_re = jnp.zeros((log_delta_sq.shape[0], self.n_freq, 0))
            else:
                if theta_re.ndim == 4:
                    theta_re = theta_re[0]
            if theta_im is None:
                theta_im = jnp.zeros((log_delta_sq.shape[0], self.n_freq, 0))
            else:
                if theta_im.ndim == 4:
                    theta_im = theta_im[0]
        else:
            # Reconstruct from individual component samples
            log_delta_sq = self._reconstruct_log_delta_sq(samples)
            theta_re = self._reconstruct_theta_params(samples, "re")
            theta_im = self._reconstruct_theta_params(samples, "im")

        # Use spline model's reconstruction method
        return self.spline_model.reconstruct_psd_matrix(
            log_delta_sq, theta_re, theta_im, n_samples_max=50
        )

    def _reconstruct_log_delta_sq(self, samples: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Reconstruct log_delta_sq from individual diagonal component samples."""
        first_sample = next(iter(samples.values()))
        n_chains = first_sample.shape[0]  # Assume chain dim added
        n_samples = first_sample.shape[0] * first_sample.shape[1]  # Total draws across chains
        log_delta_components = []

        for j in range(self.n_channels):
            weights_key = f"weights_delta_{j}"
            if weights_key in samples:
                weights_full = samples[weights_key]  # Shape: (n_chains, n_draws, n_weights)
                # For posterior predictive, use first chain only
                weights = weights_full[0]  # Shape: (n_draws, n_weights)
                # Vectorized spline evaluation using JAX
                log_delta_j = batch_spline_eval(self.all_bases[j], weights)
                log_delta_components.append(log_delta_j)

        if log_delta_components:
            return jnp.stack(log_delta_components, axis=2)  # (n_samples, n_freq, n_channels)
        else:
            # Fallback
            return jnp.zeros((n_samples, self.n_freq, self.n_channels))

    def _reconstruct_theta_params(self, samples: Dict[str, jnp.ndarray], param_type: str) -> jnp.ndarray:
        """Reconstruct theta parameters from samples."""
        key = f"weights_theta_{param_type}"
        if key in samples and self.n_theta > 0:
            weights_full = samples[key]  # Shape: (n_chains, n_draws, n_weights)
            # For posterior predictive, use first chain only
            weights = weights_full[0]  # Shape: (n_draws, n_weights)
            basis_idx = self.n_channels + (0 if param_type == "re" else 1)
            # Vectorized spline evaluation using JAX
            theta_base = batch_spline_eval(self.all_bases[basis_idx], weights)
            # Tile to match expected shape
            return jnp.tile(theta_base[:, :, None], (1, 1, max(1, self.n_theta)))
        else:
            first_sample = next(iter(samples.values()))
            n_samples = first_sample.shape[1]  # n_draws
            return jnp.zeros((n_samples, self.n_freq, max(1, self.n_theta)))

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            self._plot_psd_matrix(idata)
            self._plot_summary_diagnostics(idata)
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not create plots: {e}")
            # Create basic fallback plot
            self._create_fallback_plot()

    def _plot_psd_matrix(self, idata: az.InferenceData) -> None:
        """Plot the reconstructed PSD matrix components."""
        # Create empirical PSD matrix for comparison
        empirical_psd = compute_empirical_psd(
            fft_data_re=np.array(self.fft_data.y_re),
            fft_data_im=np.array(self.fft_data.y_im),
            n_channels=self.n_channels
        )

        plot_psd_matrix(
            idata=idata,
            n_channels=self.n_channels,
            freq=np.array(self.freq),
            empirical_psd=empirical_psd,
            outdir=self.config.outdir
        )



    def _plot_summary_diagnostics(self, idata: az.InferenceData) -> None:
        """Plot summary diagnostics for multivariate case."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Log likelihood trace
        if "log_likelihood" in idata.sample_stats:
            axes[0, 0].plot(idata.sample_stats["log_likelihood"].values.flatten())
            axes[0, 0].set_title("Log Likelihood Trace")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Log Likelihood")

        # Plot 2: Sample summary
        try:
            summary_df = az.summary(idata)
            axes[0, 1].text(0.1, 0.9, f"Parameters: {len(summary_df)}", transform=axes[0, 1].transAxes)
            axes[0, 1].text(0.1, 0.8, f"Channels: {self.n_channels}", transform=axes[0, 1].transAxes)
            axes[0, 1].text(0.1, 0.7, f"Frequencies: {self.n_freq}", transform=axes[0, 1].transAxes)
            axes[0, 1].text(0.1, 0.6, f"Runtime: {self.runtime:.2f}s", transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Summary Statistics")
            axes[0, 1].axis('off')
        except:
            axes[0, 1].text(0.5, 0.5, "Summary unavailable", ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Summary Statistics")
            axes[0, 1].axis('off')

        # Plot 3: Parameter count by type
        param_counts = {}
        for param in idata.posterior.data_vars:
            param_type = param.split('_')[0]  # Extract prefix (delta, phi, weights)
            param_counts[param_type] = param_counts.get(param_type, 0) + 1

        if param_counts:
            axes[1, 0].bar(param_counts.keys(), param_counts.values())
            axes[1, 0].set_title("Parameter Count by Type")
            axes[1, 0].set_ylabel("Count")

        # Plot 4: ESS summary if available
        try:
            ess = az.ess(idata)
            ess_values = ess.to_array().values.flatten()
            ess_values = ess_values[~np.isnan(ess_values)]
            if len(ess_values) > 0:
                axes[1, 1].hist(ess_values, bins=20, alpha=0.7)
                axes[1, 1].set_title("Effective Sample Size Distribution")
                axes[1, 1].set_xlabel("ESS")
                axes[1, 1].set_ylabel("Count")
            else:
                axes[1, 1].text(0.5, 0.5, "ESS unavailable", ha='center', va='center', transform=axes[1, 1].transAxes)
        except:
            axes[1, 1].text(0.5, 0.5, "ESS unavailable", ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(f"{self.config.outdir}/multivariate_diagnostics.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_fallback_plot(self) -> None:
        """Create basic fallback plot when main plotting fails."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5,
                f"Multivariate PSD Analysis Complete\n"
                f"Channels: {self.n_channels}\n"
                f"Frequencies: {self.n_freq}\n"
                f"Sampler: {self.sampler_type}\n"
                f"Runtime: {self.runtime:.2f}s",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.set_title("Multivariate Analysis Summary")
        ax.axis('off')
        plt.savefig(f"{self.config.outdir}/multivariate_summary.png", bbox_inches='tight')
        plt.close(fig)

    def _get_lnz(
        self,
        samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Multivariate LnZ computation using morphZ."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        try:
            # Flatten all samples into a single array
            all_samples = []
            param_sizes = []
            param_names = []

            for name, sample_array in samples.items():
                if name.startswith(('weights_', 'delta_', 'phi_')):
                    flat_samples = sample_array.reshape(sample_array.shape[0], -1)
                    all_samples.append(flat_samples)
                    param_sizes.append(flat_samples.shape[1])
                    param_names.append(name)

            if not all_samples:
                return np.nan, np.nan

            posterior_samples = np.concatenate(all_samples, axis=1)
            lposterior = sample_stats.get("log_likelihood", sample_stats.get("lp", None))

            if lposterior is None:
                return np.nan, np.nan

            def log_posterior_fn(params):
                """Reconstruct log posterior from flattened parameters."""
                # This is a simplified version - you'd need to implement the full
                # multivariate log posterior computation here
                return 0.0  # Placeholder

            # Compute evidence using morphZ
            lnz_res = morphZ.evidence(
                posterior_samples,
                lposterior,
                log_posterior_fn,
                morph_type='pair',
                kde_bw="cv_iso",
                output_path=tempfile.gettempdir()
            )[0]

            return float(lnz_res.lnz), float(lnz_res.uncertainty)

        except Exception as e:
            if self.config.verbose:
                print(f"Warning: LnZ computation failed: {e}")
            return np.nan, np.nan
