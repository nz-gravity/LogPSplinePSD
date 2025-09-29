"""
Read the results for the multivariate PSD simulation study

1. Varying N (data size) vs performance (runtime, ESS, RIAE, coverage probabilities) for fixed K (number of knots)
2. Varying K (number of knots) vs performance (runtime, ESS, RIAE, coverage probabilities) for fixed N (data size)

This script reads InferenceData objects from simulation studies and extracts relevant metrics,
saving them to a CSV file for later plotting.

The plots generated are:
1. Runtime vs N (log-log)
2. ESS vs N (log-log)
3. RIAE vs N (log-log)
4. Coverage probabilities vs N (linear)
5. Runtime vs K (log-log)
6. ESS vs K (log-log)
7. RIAE vs K (log-log)
8. Coverage probabilities vs K (linear)
"""

import glob
import os
from typing import Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

RESULTS_DIR = "out_changing_n/"
CSV_FILE = "summary_metrics.csv"
CLEAN = True  # If True, re-extract metrics even if CSV exists


def extract_metrics_from_inferencedata(
    idata: az.InferenceData,
) -> Tuple[float, float, float, float]:
    """
    Extract performance metrics from InferenceData object.

    Uses the attributes stored in the InferenceData object to extract:
    - runtime: Total sampling time
    - ess: Effective sample size (bulk method)
    - riae: Relative integrated absolute error
    - coverage: Coverage probability of true PSD values (ci_coverage)

    Args:
        idata: InferenceData object containing posterior samples and attributes

    Returns:
        Tuple of (runtime, ess, riae, coverage)
    """
    # Extract runtime directly from attributes
    runtime = idata.attrs.get("runtime", np.nan)
    ess = np.median(idata.attrs.get("ess", [np.nan]))

    # Extract RIAE (check both multiar and univar attribute names)
    riae = idata.attrs.get("riae_matrix", idata.attrs.get("riae", np.nan))

    # Extract CI coverage from attributes if available (computed during MCMC runs)
    ci_coverage = idata.attrs.get("ci_coverage", np.nan)

    # If ci_coverage not found in attributes, compute it (for backward compatibility)
    if np.isnan(ci_coverage):
        print(
            "Warning: ci_coverage not found in attributes, computing on-the-fly"
        )
        # Get true PSD from attributes
        true_psd = idata.attrs.get("true_psd", None)
        if true_psd is None:
            print("Warning: true_psd not found in attributes")
            return runtime, ess, riae, np.nan

        try:
            # Compute coverage for multivariate case
            if "psd_matrix" in idata.posterior:
                posterior_psds = idata.posterior["psd_matrix"].values
                n_samples, n_freq = posterior_psds.shape[:2]
                posterior_psds_real = np.zeros(
                    (n_samples, n_freq, *posterior_psds.shape[2:]), dtype=float
                )
                for i in range(n_samples):
                    for j in range(n_freq):
                        posterior_psds_real[i, j] = _complex_to_real(
                            posterior_psds[i, j]
                        )
                posterior_lower = np.percentile(
                    posterior_psds_real, 2.5, axis=0
                )
                posterior_upper = np.percentile(
                    posterior_psds_real, 97.5, axis=0
                )
                true_psd_real = np.zeros_like(true_psd, dtype=float)
                for j in range(n_freq):
                    true_psd_real[j] = _complex_to_real(true_psd[j])
                ci_coverage = np.mean(
                    (true_psd_real >= posterior_lower)
                    & (true_psd_real <= posterior_upper)
                )
            # Compute coverage for univariate case
            elif "psd" in idata.posterior_psd:
                psd_samples = idata.posterior_psd[
                    "psd"
                ].values  # (pp_draw, freq)
                posterior_lower = np.percentile(psd_samples, 2.5, axis=0)
                posterior_upper = np.percentile(psd_samples, 97.5, axis=0)
                ci_coverage = np.mean(
                    (true_psd >= posterior_lower)
                    & (true_psd <= posterior_upper)
                )
            else:
                print(
                    "Warning: PSD data not found in posterior for coverage calculation"
                )
                ci_coverage = np.nan
        except Exception as e:
            print(f"Warning: Could not calculate coverage: {e}")
            ci_coverage = np.nan

    return runtime, ess, riae, ci_coverage


def extract_study_parameters(
    idata: az.InferenceData,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract study parameters (N, K) from InferenceData attributes.

    Args:
        idata: InferenceData object containing study parameters in attributes

    Returns:
    """
    N = idata.attrs["n_freq"] + 1 * 2
    K = idata.spline_model["diag_0_knots"].shape[0]
    return N, K


def read_or_create_summary(
    results_dir: str = RESULTS_DIR, csv_file: str = CSV_FILE
) -> pd.DataFrame:
    """
    Read summary metrics from CSV or create from InferenceData files.

    Args:
        results_dir: Directory containing .nc files
        csv_file: Path to save/load CSV file

    Returns:
        DataFrame with extracted metrics
    """
    if os.path.exists(csv_file) and not CLEAN:
        print(f"Reading metrics from {csv_file}")
        return pd.read_csv(csv_file)
    else:
        print("Extracting metrics from InferenceData files...")
        rows = []

        # Find all .nc files in results directory
        pattern = os.path.join(results_dir, "**", "*.nc")
        nc_files = glob.glob(pattern, recursive=True)

        if not nc_files:
            print(f"No .nc files found in {results_dir}")
            return pd.DataFrame()

        for fname in tqdm(nc_files):
            try:
                print(f"Processing {os.path.basename(fname)}...")
                idata = az.from_netcdf(fname)

                # Extract metrics
                runtime, ess, riae, coverage = (
                    extract_metrics_from_inferencedata(idata)
                )

                # Extract study parameters
                N, K = extract_study_parameters(idata)

                # Extract additional useful attributes
                sampler_type = idata.attrs.get("sampler_type", "unknown")
                n_channels = idata.attrs.get("n_channels", np.nan)
                n_freq = idata.attrs.get("n_freq", np.nan)

                rows.append(
                    {
                        "filename": os.path.basename(fname),
                        "N": N,
                        "K": K,
                        "runtime": runtime,
                        "ess": ess,
                        "riae": riae,
                        "coverage": coverage,
                        "sampler_type": sampler_type,
                        "n_channels": n_channels,
                        "n_freq": n_freq,
                    }
                )

            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue

        if not rows:
            print("No valid data extracted from InferenceData files")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"Saved metrics to {csv_file}")
        return df


def plot_with_errorbars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Create errorbar plot with mean and standard deviation.

    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        group_col: Column name to group by for error bars
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the plot
    """
    # Group by x_col and calculate statistics
    grouped = df.groupby(x_col)
    x_values = []
    y_means = []
    y_stds = []

    for name, group in grouped:
        x_values.append(name)
        y_means.append(group[y_col].mean())
        y_stds.append(group[y_col].std())

    plt.figure(figsize=(5, 3.5))
    plt.errorbar(
        x_values, y_means, yerr=y_stds, fmt="o-", capsize=4, markersize=6
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    # Removed grid for PhysRevD style
    plt.tight_layout()

    if save_path:
        # Save as PDF for publication quality
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved to {pdf_path}")
    else:
        plt.show()


def create_all_plots(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create all performance plots for the simulation study.

    Args:
        df: DataFrame with extracted metrics
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter out any rows with missing data for plotting
    plot_df = df.dropna(
        subset=["N", "K", "runtime", "ess", "riae", "coverage"]
    )

    if plot_df.empty:
        print("No valid data for plotting")
        return

    print(f"Creating plots for {len(plot_df)} data points...")

    # ESS vs N (log-log)
    plot_with_errorbars(
        plot_df,
        x_col="N",
        y_col="ess",
        group_col="K",
        xlabel="N (Data Size)",
        ylabel="Effective Sample Size (ESS)",
        save_path=os.path.join(output_dir, "ess_vs_N.png"),
    )

    # RIAE vs N (log-log)
    plot_with_errorbars(
        plot_df,
        x_col="N",
        y_col="riae",
        group_col="K",
        xlabel="N (Data Size)",
        ylabel="RIAE",
        save_path=os.path.join(output_dir, "riae_vs_N.png"),
    )

    # Coverage vs N (linear)
    plot_with_errorbars(
        plot_df,
        x_col="N",
        y_col="coverage",
        group_col="K",
        xlabel="N (Data Size)",
        ylabel="Coverage Probability",
        save_path=os.path.join(output_dir, "coverage_vs_N.png"),
    )

    # ESS vs K (log-log)
    plot_with_errorbars(
        plot_df,
        x_col="K",
        y_col="ess",
        group_col="N",
        xlabel="K (Number of Knots)",
        ylabel="Effective Sample Size (ESS)",
        save_path=os.path.join(output_dir, "ess_vs_K.png"),
    )

    # RIAE vs K (log-log)
    plot_with_errorbars(
        plot_df,
        x_col="K",
        y_col="riae",
        group_col="N",
        xlabel="K (Number of Knots)",
        ylabel="RIAE",
        save_path=os.path.join(output_dir, "riae_vs_K.png"),
    )

    # Coverage vs K (linear)
    plot_with_errorbars(
        plot_df,
        x_col="K",
        y_col="coverage",
        group_col="N",
        xlabel="K (Number of Knots)",
        ylabel="Coverage Probability",
        save_path=os.path.join(output_dir, "coverage_vs_K.png"),
    )


def main(
    results_dir: str = RESULTS_DIR,
    csv_file: str = CSV_FILE,
    create_plots: bool = True,
    output_dir: str = "plots",
) -> pd.DataFrame:
    """
    Main function to extract metrics and optionally create plots.

    Args:
        results_dir: Directory containing .nc files
        csv_file: Path to save/load CSV file
        create_plots: Whether to create plots
        output_dir: Directory to save plots

    Returns:
        DataFrame with extracted metrics
    """
    df = read_or_create_summary(results_dir, csv_file)

    if df.empty:
        print("No data to process")
        return df

    print(f"\nSummary statistics:")
    print(f"Total files processed: {len(df)}")
    print(f"N values: {sorted(df['N'].dropna().unique())}")
    print(f"K values: {sorted(df['K'].dropna().unique())}")

    print("\nDataFrame info:")
    print(df.info())

    print("\nFirst few rows:")
    print(df.head())

    if create_plots:
        create_all_plots(df, output_dir)

    return df


if __name__ == "__main__":
    # Run the analysis
    df = main()

    print("\nAnalysis complete!")
