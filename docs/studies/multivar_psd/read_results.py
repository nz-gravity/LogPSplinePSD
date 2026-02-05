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
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

RESULTS_DIRS = ["out_changing_n/", "out_changing_k/"]
CSV_FILES = ["summary_metrics_n.csv", "summary_metrics_k.csv"]
CLEAN = False  # If True, re-extract metrics even if CSV exists


def extract_data(idata: az.InferenceData):
    return dict(
        runtime=idata.attrs.get("runtime", np.nan),
        ess=np.median(idata.attrs.get("ess", [np.nan])),
        riae=idata.attrs.get("riae_matrix", idata.attrs.get("riae", np.nan)),
        coverage=idata.attrs.get("ci_coverage", np.nan),
        N=idata.attrs["N"] + 1 * 2,
        K=idata.spline_model["diag_0_knots"].shape[0],
    )


def read_or_create_summary(results_dir: str, csv_file: str) -> pd.DataFrame:
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
                rows.append(
                    {
                        "filename": os.path.basename(fname),
                        **extract_data(idata),
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


def plot_with_errorbars_no_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Create errorbar plot with mean and standard deviation, but no connecting line between points.
    This preserves the correct x-axis spacing.

    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
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

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x_values,
        y_means,
        yerr=y_stds,
        fmt="o",
        capsize=4,
        markersize=8,
        color="black",
        ecolor="black",
        elinewidth=2,
        capthick=2,
        alpha=0.8,
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        # Save as PDF for publication quality
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved to {pdf_path}")
    else:
        plt.show()


def create_n_study_plots(
    csv_file: str = "docs/studies/multivar_psd/summary_metrics_n.csv",
    output_dir: str = "docs/studies/multivar_psd/plots_n",
) -> None:
    """
    Create plots for the changing N study.

    Args:
        csv_file: CSV file containing metrics for N study
        output_dir: Directory to save plots
    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("No data for N study")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating N study plots ({len(df)} data points)...")

    # Filter out any rows with missing data for plotting
    plot_df = df.dropna(
        subset=["N", "K", "runtime", "ess", "riae", "coverage"]
    )

    if plot_df.empty:
        print("No valid data for N study plotting")
        return

    # ESS vs N
    plot_with_errorbars_no_line(
        plot_df,
        x_col="N",
        y_col="ess",
        xlabel="N (Data Size)",
        ylabel="Effective Sample Size (ESS)",
        save_path=os.path.join(output_dir, "ess_vs_N.png"),
    )

    # RIAE vs N
    plot_with_errorbars_no_line(
        plot_df,
        x_col="N",
        y_col="riae",
        xlabel="N (Data Size)",
        ylabel="RIAE",
        save_path=os.path.join(output_dir, "riae_vs_N.png"),
    )

    # Coverage vs N
    plot_with_errorbars_no_line(
        plot_df,
        x_col="N",
        y_col="coverage",
        xlabel="N (Data Size)",
        ylabel="Coverage Probability",
        save_path=os.path.join(output_dir, "coverage_vs_N.png"),
    )

    # Runtime vs N
    plot_with_errorbars_no_line(
        plot_df,
        x_col="N",
        y_col="runtime",
        xlabel="N (Data Size)",
        ylabel="Runtime (seconds)",
        save_path=os.path.join(output_dir, "runtime_vs_N.png"),
    )


def create_k_study_plots(
    csv_file: str = "docs/studies/multivar_psd/summary_metrics_k.csv",
    output_dir: str = "docs/studies/multivar_psd/plots_k",
) -> None:
    """
    Create plots for the changing K study.

    Args:
        csv_file: CSV file containing metrics for K study
        output_dir: Directory to save plots
    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("No data for K study")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating K study plots ({len(df)} data points)...")

    # Filter out any rows with missing data for plotting
    plot_df = df.dropna(
        subset=["N", "K", "runtime", "ess", "riae", "coverage"]
    )

    if plot_df.empty:
        print("No valid data for K study plotting")
        return

    # ESS vs K
    plot_with_errorbars_no_line(
        plot_df,
        x_col="K",
        y_col="ess",
        xlabel="K (Number of Knots)",
        ylabel="Effective Sample Size (ESS)",
        save_path=os.path.join(output_dir, "ess_vs_K.png"),
    )

    # RIAE vs K
    plot_with_errorbars_no_line(
        plot_df,
        x_col="K",
        y_col="riae",
        xlabel="K (Number of Knots)",
        ylabel="RIAE",
        save_path=os.path.join(output_dir, "riae_vs_K.png"),
    )

    # Coverage vs K
    plot_with_errorbars_no_line(
        plot_df,
        x_col="K",
        y_col="coverage",
        xlabel="K (Number of Knots)",
        ylabel="Coverage Probability",
        save_path=os.path.join(output_dir, "coverage_vs_K.png"),
    )

    # Runtime vs K
    plot_with_errorbars_no_line(
        plot_df,
        x_col="K",
        y_col="runtime",
        xlabel="K (Number of Knots)",
        ylabel="Runtime (seconds)",
        save_path=os.path.join(output_dir, "runtime_vs_K.png"),
    )


def main() -> None:
    """
    Main function to create plots for both N and K studies.
    """
    print("Creating plots for changing N study...")
    create_n_study_plots()

    print("\nCreating plots for changing K study...")
    create_k_study_plots()


if __name__ == "__main__":
    # Run the analysis
    main()

    print("\nAnalysis complete!")
