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
CLEAN = True  # If True, re-extract metrics even if CSV exists


def extract_data(idata: az.InferenceData):
    return dict(
        runtime=idata.attrs.get("runtime", np.nan),
        ess=np.median(idata.attrs.get("ess", [np.nan])),
        riae=idata.attrs.get("riae_matrix", idata.attrs.get("riae", np.nan)),
        coverage=idata.attrs.get("ci_coverage", np.nan),
        N=idata.attrs["n_freq"] + 1 * 2,
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
    results_dirs: list = RESULTS_DIRS,
    csv_files: list = CSV_FILES,
    create_plots: bool = True,
    output_dir: str = "plots",
    combine_studies: bool = True,
) -> pd.DataFrame:
    """
    Main function to extract metrics and optionally create plots.

    Args:
        results_dirs: List of directories containing .nc files (change_n and change_k studies)
        csv_files: List of paths to save/load CSV files
        create_plots: Whether to create plots
        output_dir: Directory to save plots
        combine_studies: Whether to combine results from all directories

    Returns:
        DataFrame with extracted metrics
    """
    dfs = []
    for results_dir, csv_file in zip(results_dirs, csv_files):
        df = read_or_create_summary(results_dir, csv_file)
        dfs.append(df)

    if combine_studies:
        df = pd.concat(dfs, ignore_index=True)
    else:
        if not dfs:
            df = pd.DataFrame()
        else:
            df = dfs[0]

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
