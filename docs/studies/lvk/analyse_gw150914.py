"""
Gravitational Wave Event Data Analysis with P-Splines PSD Estimation

This module provides functionality for loading, processing, and analyzing 
gravitational wave event data, with particular focus on Power Spectral Density 
(PSD) estimation using P-splines methods and comparison with catalog PSDs.
"""

import os
from typing import Dict, Tuple
import argparse
import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import kstest, norm
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from log_psplines.datatypes import Periodogram
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.psplines import LogPSplines
from log_psplines.arviz_utils import get_periodogram, get_spline_model, get_weights

# Configuration constants
DEFAULT_FFT_LENGTH = 4
DEFAULT_OVERLAP = 0.5
DEFAULT_WINDOW = 'hann'
DEFAULT_N_KNOTS = 100
DEFAULT_SPLINE_DEGREE = 3
DEFAULT_DIFF_ORDER = 2
DEFAULT_N_SAMPLES = 2000
DEFAULT_N_WARMUP = 2000

# Detector color mapping for consistent plotting
DETECTOR_COLORS = {'H1': 'red', 'L1': 'blue', 'V1': 'green'}


def compute_normality_pvalue(data_fd: np.ndarray, psd_values: np.ndarray,
                             frequencies_data: np.ndarray, frequencies_psd: np.ndarray) -> float:
    """Compute p-value for normality test of whitened data."""
    # Find matching frequencies
    freq_mask = np.isin(frequencies_data, frequencies_psd)
    common_freqs = frequencies_data[freq_mask]
    psd_matched = psd_values[np.isin(frequencies_psd, common_freqs)]

    # Compute normalized ratios (whitened data)
    whitened_data = data_fd[freq_mask] / np.sqrt(psd_matched)

    # Combine real and imaginary parts for normality test
    ratio_combined = np.concatenate([
        np.real(whitened_data),
        np.imag(whitened_data)
    ])

    # Kolmogorov-Smirnov test against standard normal
    _, pvalue = kstest(ratio_combined, norm.cdf)

    return pvalue


class GWEventData:
    """Container for gravitational wave event data and metadata."""

    def __init__(self, event_name: str):
        self.event_name = event_name
        self.analysis_group = None

        # Data containers
        self.configs = {}
        self.psds = {}
        self.strain_data = {}
        self.welch_psds = {}
        self.postevent_fd = {}

    @classmethod
    def from_hdf5(cls, filepath: str) -> 'GWEventData':
        """Load event data from HDF5 file."""
        with h5py.File(filepath, "r") as f:
            # Load event metadata
            event_name = cls._decode_if_bytes(f.attrs['event_name'])
            instance = cls(event_name)
            instance.analysis_group = cls._decode_if_bytes(
                f.attrs.get('analysis_group', 'unknown')
            )

            # Load each data section
            instance._load_configs(f)
            instance._load_psds(f)
            instance._load_welch_psds(f)
            instance._load_postevent_fd(f)
            instance._load_strain_data(f)

        return instance

    @staticmethod
    def _decode_if_bytes(data) -> str:
        """Decode bytes to string if necessary."""
        return data.decode('utf-8') if isinstance(data, bytes) else data

    def _load_configs(self, h5_file):
        """Load configuration data from HDF5 file."""
        if "configs" in h5_file:
            config_group = h5_file["configs"]
            for key in config_group.keys():
                self.configs[key] = self._decode_if_bytes(config_group[key][()])

    def _load_psds(self, h5_file):
        """Load PSD data from HDF5 file."""
        if "psds" in h5_file:
            psd_group = h5_file["psds"]
            for detector in psd_group.keys():
                det_group = psd_group[detector]
                self.psds[detector] = {
                    'freqs': det_group["freqs"][:],
                    'psd': det_group["psd"][:]
                }

    def _load_welch_psds(self, h5_file):
        """Load Welch PSD data from HDF5 file."""
        if "welch_psds" in h5_file:
            welch_group = h5_file["welch_psds"]
            for detector in welch_group.keys():
                det_group = welch_group[detector]
                self.welch_psds[detector] = {
                    'freqs': det_group["freqs"][:],
                    'psd': det_group["psd"][:]
                }

    def _load_postevent_fd(self, h5_file):
        """Load post-event frequency domain data from HDF5 file."""
        if "postevent_fd" in h5_file:
            fd_group = h5_file["postevent_fd"]
            for detector in fd_group.keys():
                det_group = fd_group[detector]
                self.postevent_fd[detector] = {
                    'freqs': det_group["freqs"][:],
                    'datafd': det_group["datafd"][:]
                }

    def _load_strain_data(self, h5_file):
        """Load strain time series data from HDF5 file."""
        if "strain_data" in h5_file:
            strain_group = h5_file["strain_data"]
            for detector in strain_group.keys():
                det_group = strain_group[detector]
                self.strain_data[detector] = {}

                for data_type in det_group.keys():
                    type_group = det_group[data_type]
                    times = type_group["times"][:]
                    strain = type_group["strain"][:]
                    self.strain_data[detector][data_type] = (times, strain)

    def get_detectors(self) -> list:
        """Get list of available detectors."""
        return sorted(self.psds.keys())

    def __repr__(self):
        return f"<GWEventData.{self.event_name}>"


def create_psd_comparison_plot(event_data: GWEventData,
                               posterior_quantiles: Dict[str, np.ndarray],
                               output_dir: str = "plots") -> str:
    """Create comparison plots of different PSD estimates."""
    detectors = event_data.get_detectors()
    n_detectors = len(detectors)

    # Create subplot layout
    fig, axes = plt.subplots(n_detectors, 1, figsize=(10, 4 * n_detectors),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, detector in enumerate(detectors):
        ax = axes[i]

        # Extract data
        psd_data = event_data.psds[detector]
        welch_data = event_data.welch_psds[detector]
        fd_data = event_data.postevent_fd[detector]
        pspline_data = posterior_quantiles[detector]

        # Calculate p-values for different PSD estimates
        pval_gwtc = compute_normality_pvalue(
            fd_data['datafd'], psd_data['psd'],
            fd_data['freqs'], psd_data['freqs']
        )
        pval_welch = compute_normality_pvalue(
            fd_data['datafd'], welch_data['psd'],
            fd_data['freqs'], welch_data['freqs']
        )
        pval_pspline = compute_normality_pvalue(
            fd_data['datafd'], pspline_data['quantiles'][1],
            fd_data['freqs'], pspline_data['freqs']
        )

        color = DETECTOR_COLORS.get(detector, 'black')

        # Plot data and PSDs
        ax.loglog(fd_data['freqs'], np.abs(fd_data['datafd']) ** 2,
                  color='lightgray', alpha=0.4, label='Post-event Data')

        ax.loglog(psd_data['freqs'], psd_data['psd'], color=color,
                  linewidth=1.5, label=f'GWTC PSD (p={pval_gwtc:.3f})')

        ax.loglog(welch_data['freqs'], welch_data['psd'], color=color,
                  linewidth=2, linestyle='--', alpha=0.3,
                  label=f'Welch PSD (p={pval_welch:.3f})')

        ax.loglog(pspline_data['freqs'], pspline_data['quantiles'][1],
                  color='tab:green', linewidth=2)
        ax.fill_between(pspline_data['freqs'],
                        pspline_data['quantiles'][0],
                        pspline_data['quantiles'][2],
                        color='tab:green', alpha=0.3,
                        label=f'P-splines 90% CI (p={pval_pspline:.3f})')

        # Formatting
        ax.set_ylabel("PSD [strain²/Hz]", fontsize=12)
        ax.set_title(f"{detector} Power Spectral Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set reasonable axis limits
        if len(psd_data['psd']) > 0:
            ymin = np.min(psd_data['psd']) * 0.1
            ymax = np.max(psd_data['psd']) * 10
            ax.set_ylim(ymin, ymax)

    # Final formatting
    axes[-1].set_xlabel("Frequency [Hz]", fontsize=12)
    fig.suptitle(f"{event_data.event_name} Power Spectral Densities",
                 fontsize=14, y=0.98)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{event_data.event_name}_psd.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PSD comparison plot saved to: {plot_path}")
    return plot_path


def process_detector_data(event_data: GWEventData, detector: str,
                          fmin: float, fmax: float, output_dir: str) -> Dict[str, np.ndarray]:
    """Process strain data for a single detector using P-splines."""
    print(f"Processing {detector} detector data...")

    # Create detector-specific output directory
    det_output_dir = os.path.join(output_dir, detector)
    os.makedirs(det_output_dir, exist_ok=True)

    # Extract and standardize strain data
    times, strain = event_data.strain_data[detector]['analysis']
    strain_std = np.std(strain)
    strain_normalized = (strain - np.mean(strain)) / strain_std

    # Create GWpy TimeSeries and compute PSD
    ts = GwpyTimeSeries(strain_normalized, t0=times[0], dt=times[1] - times[0])
    avg_pdgrm = ts.psd(fftlength=DEFAULT_FFT_LENGTH, overlap=DEFAULT_OVERLAP,
                       window=DEFAULT_WINDOW, method='welch')

    # Convert to Periodogram object and apply frequency cuts
    pdgrm = Periodogram(
        freqs=avg_pdgrm.frequencies.value,
        power=avg_pdgrm.value
    )
    pdgrm = pdgrm.cut(fmin, fmax)

    # Check for existing inference data
    idata_path = os.path.join(det_output_dir, "inference_data.nc")

    if os.path.exists(idata_path):
        print(f"Loading existing inference data from {idata_path}")
        idata = az.from_netcdf(idata_path)
    else:
        print(f"Running MCMC inference for {detector}...")

        # Create and fit initial spline model

        # Run MCMC sampling
        idata = run_mcmc(
            pdgrm, sampler='mh',
            n_samples=DEFAULT_N_SAMPLES, n_warmup=DEFAULT_N_WARMUP,
            outdir=det_output_dir, rng_key=42,
            knot_kwargs=dict(
                method='lvk',
                knots_plotfn=os.path.join(det_output_dir, "knots.png"),
                extra_thresh_multiplier=4.0,
                max_extra_per_peak=10,
                d=15,
            )
        )

    # Extract posterior samples and compute quantiles
    spline_model = get_spline_model(idata)
    periodogram = get_periodogram(idata)
    weights = get_weights(idata)

    # Compute log-splines for all posterior samples
    ln_splines = np.array([spline_model(w) for w in weights], dtype=np.float64)

    # Get quantiles and transform back to linear scale
    posterior_quantiles = np.exp(np.quantile(ln_splines, [0.05, 0.5, 0.95], axis=0))

    # Apply scaling correction
    posterior_quantiles *= strain_std ** 2

    return {
        'quantiles': posterior_quantiles,
        'freqs': periodogram.freqs
    }


def analyze_event(data_file: str, output_dir: str = None):
    """Analyze a single gravitational wave event."""
    # Set default output directory based on event name
    if output_dir is None:
        event_name = os.path.splitext(os.path.basename(data_file))[0]
        output_dir = f"out_lvk_{event_name.split('_')[0]}"  # e.g., "out_lvk_GW150914"

    # Load event data
    print(f"Loading event data from {data_file}")
    event_data = GWEventData.from_hdf5(data_file)
    print(f"Loaded data for event: {event_data.event_name}")

    # Set frequency bounds (using L1 as reference, fallback to first available detector)
    ref_detector = 'L1' if 'L1' in event_data.psds else list(event_data.psds.keys())[0]
    fmin = event_data.psds[ref_detector]['freqs'][0]
    fmax = event_data.psds[ref_detector]['freqs'][-1]
    print(f"Frequency range: {fmin:.1f} - {fmax:.1f} Hz (using {ref_detector} as reference)")

    # Process each detector
    quantiles = {}
    for detector in event_data.get_detectors():
        print(f"\nDetector: {detector}, PSD points: {len(event_data.psds[detector]['freqs'])}")

        quantiles[detector] = process_detector_data(
            event_data, detector, fmin, fmax, output_dir
        )

    # Create comparison plots
    print(f"\nCreating PSD comparison plots...")
    create_psd_comparison_plot(event_data, quantiles, output_dir=output_dir)
    print("Analysis complete!")


def main():
    """Main analysis pipeline with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze GW event data with P-splines PSD estimation')
    parser.add_argument('-d', '--data-file', help='Path to HDF5 data file', default=None)
    parser.add_argument('-o', '--output', help='Output directory (default: auto-generated from event name)')

    args = parser.parse_args()
    data_file = args.data_file
    if data_file is None:
        data_file = glob.glob("*.h5")[0]

    analyze_event(data_file, args.output)


if __name__ == "__main__":
    main()
