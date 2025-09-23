import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional






def create_fallback_plot(
    outdir: str,
    n_channels: Optional[int] = None,
    n_freq: Optional[int] = None,
    sampler_type: str = "Unknown",
    runtime: Optional[float] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Create basic fallback plot when main plotting fails.

    Parameters
    ----------
    outdir : str
        Output directory for saving plot
    n_channels : int, optional
        Number of channels (for multivariate)
    n_freq : int, optional
        Number of frequencies
    sampler_type : str
        Type of sampler used
    runtime : float, optional
        Runtime in seconds
    figsize : tuple
        Figure size for plot
    """
    if outdir is None:
        return

    fig, ax = plt.subplots(figsize=figsize)
    text_parts = ["Analysis Complete"]
    if n_channels is not None:
        text_parts.append(f"Channels: {n_channels}")
    if n_freq is not None:
        text_parts.append(f"Frequencies: {n_freq}")
    text_parts.append(f"Sampler: {sampler_type}")
    if runtime is not None:
        text_parts.append(f"Runtime: {runtime:.2f}s")

    ax.text(0.5, 0.5, "\n".join(text_parts), ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title("Analysis Summary")
    ax.axis('off')
    plt.savefig(f"{outdir}/analysis_summary.png", bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics(
    idata: az.InferenceData,
    outdir: str,
    n_channels: Optional[int] = None,
    n_freq: Optional[int] = None,
    runtime: Optional[float] = None,
    figsize: tuple = (12, 8),
) -> None:
    """
    Plot comprehensive MCMC diagnostics including trace plots, summary statistics,
    and sampler-specific metrics.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data from MCMC analysis
    outdir : str
        Output directory for saving plots
    n_channels : int, optional
        Number of channels (for multivariate)
    n_freq : int, optional
        Number of frequencies
    runtime : float, optional
        Runtime in seconds
    figsize : tuple
        Figure size for plots
    """
    if outdir is None:
        return

    try:
        # Trace plots
        az.plot_trace(idata, figsize=figsize)
        plt.suptitle("Trace plots")
        plt.tight_layout()
        plt.savefig(f"{outdir}/trace_plots.png")
        plt.close()

        # Acceptance rate plot with enhanced interpretation
        if "acceptance_rate" in idata.sample_stats:
            fig, ax = plt.subplots(figsize=(10, 4))
            accept_rates = idata.sample_stats.acceptance_rate.values.flatten()

            # Determine if this is NUTS or MH based on target acceptance rate
            target_rate = idata.attrs.get("target_accept_rate", 0.44)
            sampler_type = "NUTS" if target_rate > 0.5 else "MH"
            good_range = (0.7, 0.9) if target_rate > 0.5 else (0.2, 0.5)

            ax.plot(accept_rates, alpha=0.7, color='blue')

            # Add target line
            ax.axhline(y=target_rate, color="red", linestyle="--", linewidth=2,
                      label=f"Target ({target_rate})")

            # Add optimal range shading
            ax.axhspan(good_range[0], good_range[1], alpha=0.1, color='green',
                      label='.1f')

            ax.axhspan(0, good_range[0], alpha=0.1, color='red', label='Too low')
            if good_range[1] < 1.0:
                ax.axhspan(good_range[1], 1.0, alpha=0.1, color='orange', label='Too high')

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Acceptance Rate")
            ax.set_title(f"{sampler_type} Acceptance Rate Over Time")
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, alpha=0.3)

            # Add interpretation text
            interpretation = f"{sampler_type} target: {target_rate}. "
            if target_rate > 0.5:
                interpretation += "NUTS aims for 0.7-0.9"
            else:
                interpretation += "MH aims for 0.2-0.5"
            ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
                   fontsize='small', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.savefig(f"{outdir}/acceptance_rate.png")
            plt.close()

        # Step size evolution
        if "step_size_mean" in idata.sample_stats:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            step_means = idata.sample_stats.step_size_mean.values.flatten()
            step_stds = idata.sample_stats.step_size_std.values.flatten()

            ax1.plot(step_means, alpha=0.7)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Mean Step Size")
            ax1.set_title("Step Size Evolution")
            ax1.grid(True, alpha=0.3)

            ax2.plot(step_stds, alpha=0.7, color="orange")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Step Size Std")
            ax2.set_title("Step Size Variability")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{outdir}/step_size_evolution.png")
            plt.close()

        # Summary diagnostics plot (2x2)
        summary_fig, axes = plt.subplots(2, 2, figsize=figsize)

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
            if n_channels is not None:
                axes[0, 1].text(0.1, 0.8, f"Channels: {n_channels}", transform=axes[0, 1].transAxes)
            if n_freq is not None:
                axes[0, 1].text(0.1, 0.7, f"Frequencies: {n_freq}", transform=axes[0, 1].transAxes)
            if runtime is not None:
                axes[0, 1].text(0.1, 0.6, f"Runtime: {runtime:.2f}s", transform=axes[0, 1].transAxes)
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

        # Plot 4: ESS summary with reference lines and interpretation
        try:
            ess = az.ess(idata)
            ess_values = ess.to_array().values.flatten()
            ess_values = ess_values[~np.isnan(ess_values)]
            if len(ess_values) > 0:
                n, bins, patches = axes[1, 1].hist(ess_values, bins=20, alpha=0.7)

                # Add reference lines for ESS interpretation
                ess_thresholds = [
                    (400, 'red', '--', 'Minimum reliable ESS'),
                    (1000, 'orange', '--', 'Good ESS'),
                    (np.max(ess_values), 'green', ':', f'Max ESS = {np.max(ess_values):.0f}')
                ]

                for threshold, color, style, label in ess_thresholds:
                    axes[1, 1].axvline(x=threshold, color=color, linestyle=style,
                                     linewidth=2, alpha=0.8, label=label)

                axes[1, 1].set_title("Effective Sample Size Distribution")
                axes[1, 1].set_xlabel("ESS")
                axes[1, 1].set_ylabel("Count")
                axes[1, 1].legend(loc='upper right', fontsize='x-small')

                # Add summary stats text
                min_ess = np.min(ess_values)
                mean_ess = np.mean(ess_values)
                pct_good = np.mean(ess_values >= 400) * 100
                axes[1, 1].text(0.02, 0.98, f'Min ESS: {min_ess:.0f}\nMean ESS: {mean_ess:.0f}\n≥400: {pct_good:.1f}%',
                               transform=axes[1, 1].transAxes, fontsize='small',
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            else:
                axes[1, 1].text(0.5, 0.5, "ESS unavailable", ha='center', va='center', transform=axes[1, 1].transAxes)
        except:
            axes[1, 1].text(0.5, 0.5, "ESS unavailable", ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(f"{outdir}/summary_diagnostics.png", dpi=150, bbox_inches='tight')
        plt.close(summary_fig)

    except Exception as e:
        print(f"Error generating MCMC diagnostics: {e}")

    # NUTS-specific diagnostics
    try:
        # Divergences over time (cumulative count)
        if "diverging" in idata.sample_stats:
            divergences = idata.sample_stats.diverging.values.flatten()
            cumulative_divergences = np.cumsum(divergences.astype(int))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Cumulative divergences
            ax1.plot(cumulative_divergences, alpha=0.7)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Cumulative Divergences")
            ax1.set_title("Cumulative Divergent Transitions")
            ax1.grid(True, alpha=0.3)

            # Divergence rate over time (rolling window)
            window_size = min(100, len(divergences) // 10)
            if window_size > 0:
                rolling_divergence_rate = []
                for i in range(window_size, len(divergences)):
                    rate = np.mean(divergences[i-window_size:i])
                    rolling_divergence_rate.append(rate)

                ax2.plot(np.arange(window_size, len(divergences)), rolling_divergence_rate, alpha=0.7)
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel(f"Divergence Rate (last {window_size} steps)")
                ax2.set_title("Rolling Divergence Rate")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{outdir}/divergences.png")
            plt.close()

        # Energy diagnostics (Hamiltonian and Potential energy)
        energy_available = "energy" in idata.sample_stats
        potential_energy_available = "potential_energy" in idata.sample_stats

        if energy_available or potential_energy_available:
            n_rows = 1 if (energy_available and potential_energy_available) else 1
            n_cols = 4 if (energy_available and potential_energy_available) else 2

            if energy_available and potential_energy_available:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            elif energy_available:
                fig, axes = plt.subplots(2, 2, figsize=figsize)
            else:  # potential_energy only
                fig, axes = plt.subplots(2, 2, figsize=figsize)

            plot_col = 0

            # Hamiltonian Energy plots
            if energy_available:
                energy = idata.sample_stats.energy.values.flatten()

                # Energy trace
                if energy_available and potential_energy_available:
                    ax1, ax2 = axes[0, plot_col], axes[0, plot_col+1]
                    ax3, ax4 = axes[1, plot_col], axes[1, plot_col+1]
                else:
                    ax1, ax2 = axes[0, 0], axes[0, 1]
                    ax3, ax4 = axes[1, 0], axes[1, 1]

                ax1.plot(energy, alpha=0.7)
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Hamiltonian Energy")
                ax1.set_title("Hamiltonian Energy Trace")
                ax1.grid(True, alpha=0.3)

                # Energy histogram
                ax2.hist(energy, bins=30, alpha=0.7, density=True)
                ax2.set_xlabel("Hamiltonian Energy")
                ax2.set_ylabel("Density")
                ax2.set_title("Energy Distribution")
                ax2.grid(True, alpha=0.3)

                # Energy change per step
                energy_diffs = np.diff(energy)
                ax3.plot(energy_diffs, alpha=0.7)
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Energy Change")
                ax3.set_title("Energy Changes Between Steps")
                ax3.grid(True, alpha=0.3)

                # Energy change distribution
                ax4.hist(energy_diffs, bins=30, alpha=0.7, density=True)
                ax4.set_xlabel("Energy Change")
                ax4.set_ylabel("Density")
                ax4.set_title("Energy Change Distribution")
                ax4.grid(True, alpha=0.3)

                if energy_available and potential_energy_available:
                    plot_col += 2

            # Potential Energy plots
            if potential_energy_available:
                potential_energy = idata.sample_stats.potential_energy.values.flatten()

                if energy_available and potential_energy_available:
                    ax1, ax2 = axes[0, plot_col], axes[0, plot_col+1]
                    ax3, ax4 = axes[1, plot_col], axes[1, plot_col+1]
                else:
                    ax1, ax2 = axes[0, 0], axes[0, 1]
                    ax3, ax4 = axes[1, 0], axes[1, 1]

                # Potential energy trace
                ax1.plot(potential_energy, alpha=0.7, color='orange')
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Potential Energy")
                ax1.set_title("Potential Energy Trace")
                ax1.grid(True, alpha=0.3)

                # Potential energy histogram
                ax2.hist(potential_energy, bins=30, alpha=0.7, density=True, color='orange')
                ax2.set_xlabel("Potential Energy")
                ax2.set_ylabel("Density")
                ax2.set_title("Potential Energy Distribution")
                ax2.grid(True, alpha=0.3)

                # Potential energy change per step
                potential_energy_diffs = np.diff(potential_energy)
                ax3.plot(potential_energy_diffs, alpha=0.7, color='orange')
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Potential Energy Change")
                ax3.set_title("Potential Energy Changes Between Steps")
                ax3.grid(True, alpha=0.3)

                # Potential energy change distribution
                ax4.hist(potential_energy_diffs, bins=30, alpha=0.7, density=True, color='orange')
                ax4.set_xlabel("Potential Energy Change")
                ax4.set_ylabel("Density")
                ax4.set_title("Potential Energy Change Distribution")
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{outdir}/energy_diagnostics.png")
            plt.close()

        # Tree depth and step count diagnostics
        if "tree_depth" in idata.sample_stats and "n_steps" in idata.sample_stats:
            tree_depth = idata.sample_stats.tree_depth.values.flatten()
            n_steps = idata.sample_stats.n_steps.values.flatten()

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # Tree depth trace
            ax1.plot(tree_depth, alpha=0.7)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Tree Depth")
            ax1.set_title("NUTS Tree Depth")
            ax1.grid(True, alpha=0.3)

            # Steps per trajectory
            ax2.plot(n_steps, alpha=0.7)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Leapfrog Steps")
            ax2.set_title("Leapfrog Steps Per Trajectory")
            ax2.grid(True, alpha=0.3)

            # Tree depth distribution
            unique_depths, counts_depth = np.unique(tree_depth, return_counts=True)
            ax3.bar(unique_depths, counts_depth, alpha=0.7)
            ax3.set_xlabel("Tree Depth")
            ax3.set_ylabel("Count")
            ax3.set_title("Tree Depth Distribution")
            ax3.grid(True, alpha=0.3)

            # Steps distribution
            unique_steps, counts_steps = np.unique(n_steps, return_counts=True)
            ax4.bar(unique_steps, counts_steps, alpha=0.7)
            ax4.set_xlabel("Leapfrog Steps")
            ax4.set_ylabel("Count")
            ax4.set_title("Leapfrog Steps Distribution")
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{outdir}/nuts_trajectory_stats.png")
            plt.close()

        # Energy error diagnostics
        if "energy_error" in idata.sample_stats:
            energy_error = idata.sample_stats.energy_error.values.flatten()
            abs_energy_error = np.abs(energy_error)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # Energy error trace
            ax1.plot(energy_error, alpha=0.7)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Energy Error")
            ax1.set_title("Energy Error Trace")
            ax1.grid(True, alpha=0.3)

            # Absolute energy error trace
            ax2.plot(abs_energy_error, alpha=0.7)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Absolute Energy Error")
            ax2.set_title("Absolute Energy Error Trace")
            ax2.grid(True, alpha=0.3)

            # Energy error histogram
            ax3.hist(energy_error, bins=30, alpha=0.7, density=True)
            ax3.set_xlabel("Energy Error")
            ax3.set_ylabel("Density")
            ax3.set_title("Energy Error Distribution")
            ax3.grid(True, alpha=0.3)

            # Large energy errors (> 0.1) with divergences
            error_threshold = 0.1
            large_errors = abs_energy_error > error_threshold

            if "diverging" in idata.sample_stats:
                divergences = idata.sample_stats.diverging.values.flatten()
                ax4.scatter(abs_energy_error, divergences.astype(int),
                          alpha=0.5, s=2, c='red', label='Divergent')
                ax4.scatter(abs_energy_error[~divergences], np.zeros(sum(~divergences)),
                          alpha=0.5, s=2, c='blue', label='Non-divergent')
                ax4.set_xlabel("Absolute Energy Error")
                ax4.set_ylabel("Divergence (0/1)")
                ax4.set_title(f"Energy Error vs Divergence (> {error_threshold})")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.hist(abs_energy_error, bins=30, alpha=0.7, density=True, log=True)
                ax4.set_xlabel("Absolute Energy Error")
                ax4.set_ylabel("Log Density")
                ax4.set_title("Energy Error Distribution (log scale)")
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{outdir}/energy_error_diagnostics.png")
            plt.close()

        # NUTS extra fields diagnostics: potential_energy, num_steps, accept_prob
        nuts_fields_available = []
        if "potential_energy" in idata.sample_stats:
            nuts_fields_available.append("potential_energy")
        if "num_steps" in idata.sample_stats:
            nuts_fields_available.append("num_steps")
        if "accept_prob" in idata.sample_stats:
            nuts_fields_available.append("accept_prob")

        if nuts_fields_available:
            n_fields = len(nuts_fields_available)
            if n_fields == 1:
                fig, axes = plt.subplots(2, 2, figsize=figsize)
                axes = axes.reshape(1, 4) if n_fields == 1 else axes
            elif n_fields == 2:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            else:  # 3 fields
                fig, axes = plt.subplots(2, 6, figsize=(24, 8))

            plot_idx = 0
            for field_name in nuts_fields_available:
                if field_name == "potential_energy":
                    field_data = idata.sample_stats.potential_energy.values.flatten()
                    field_title = "Potential Energy"
                    color = 'orange'
                    # Add rolling mean line for stability reference
                    window_size = min(50, len(field_data) // 4)
                    if window_size > 1:
                        rolling_mean = np.convolve(field_data, np.ones(window_size)/window_size, mode='valid')
                        rolling_start = window_size // 2
                        reference_values = [(f'Rolling Mean (w={window_size})', 'blue', '--',
                                           rolling_mean, rolling_start, 'Stability reference')]
                    else:
                        reference_values = []
                    # Color zones for potential energy (relative to mean)
                    data_mean = np.mean(field_data)
                    data_std = np.std(field_data)
                    zones = [
                        (data_mean - 2*data_std, data_mean + 2*data_std, 'green', 'Normal range'),
                        (data_mean - 3*data_std, data_mean - 2*data_std, 'yellow', 'Concerning'),
                        (data_mean + 2*data_std, data_mean + 3*data_std, 'yellow', 'Concerning'),
                    ]

                elif field_name == "num_steps":
                    field_data = idata.sample_stats.num_steps.values.flatten()
                    field_title = "Number of Steps"
                    color = 'green'
                    # Reference line at max possible steps (2^max_tree_depth)
                    max_steps = 2 ** 10  # 1024, assuming max_tree_depth=10
                    reference_values = [(f'Max Steps (2^{10})', 'red', '--', max_steps, None, 'Hitting max = inefficient')]
                    # Color zones for steps
                    zones = [
                        (1, max_steps * 0.5, 'green', 'Efficient'),
                        (max_steps * 0.5, max_steps * 0.8, 'yellow', 'Moderate'),
                        (max_steps * 0.8, max_steps, 'red', 'Inefficient'),
                    ]

                elif field_name == "accept_prob":
                    field_data = idata.sample_stats.accept_prob.values.flatten()
                    field_title = "Acceptance Probability"
                    color = 'purple'
                    # Target acceptance probability
                    target_accept = 0.8
                    reference_values = [(f'Target (0.8)', 'red', '--', target_accept, None, 'NUTS target acceptance')]
                    # Color zones for acceptance probability
                    zones = [
                        (0.7, 0.9, 'green', 'Good range'),
                        (0.6, 0.7, 'yellow', 'Borderline'),
                        (0.9, 1.0, 'yellow', 'Too high'),
                        (0.0, 0.6, 'red', 'Too low'),
                    ]

                # Trace plot
                row = plot_idx // 2
                col_start = (plot_idx % 2) * 2
                ax_trace = axes[row, col_start]

                # Add color zones as background
                if 'zones' in locals():
                    for zone_min, zone_max, zone_color, label in zones:
                        ax_trace.axhspan(zone_min, zone_max, alpha=0.1, color=zone_color, label=label)

                ax_trace.plot(field_data, alpha=0.7, color=color, linewidth=1)
                ax_trace.set_xlabel("Iteration")
                ax_trace.set_ylabel(field_title)
                ax_trace.set_title(f"{field_title} Trace")
                ax_trace.grid(True, alpha=0.3)

                # Add reference lines/values to trace plot
                if 'reference_values' in locals():
                    for ref_label, ref_color, ref_style, ref_value, ref_start, ref_desc in reference_values:
                        if ref_start is None:
                            # Horizontal reference line
                            ax_trace.axhline(y=ref_value, color=ref_color, linestyle=ref_style,
                                           linewidth=2, alpha=0.8, label=ref_label)
                        else:
                            # Rolling reference line
                            ax_trace.plot(range(ref_start, ref_start + len(ref_value)),
                                        ref_value, color=ref_color, linestyle=ref_style,
                                        linewidth=2, alpha=0.8, label=ref_label)

                # Add legend if we have reference lines
                if 'reference_values' in locals() and reference_values:
                    ax_trace.legend(loc='best', fontsize='small')

                # Histogram
                ax_hist = axes[row, col_start+1]
                n, bins, patches = ax_hist.hist(field_data, bins=30, alpha=0.7, density=True, color=color)

                # Add vertical reference lines to histogram
                if 'reference_values' in locals():
                    for ref_label, ref_color, ref_style, ref_value, ref_start, ref_desc in reference_values:
                        if ref_start is None and isinstance(ref_value, (int, float)):
                            # Vertical reference line on histogram
                            ax_hist.axvline(x=ref_value, color=ref_color, linestyle=ref_style,
                                          linewidth=2, alpha=0.8, label=ref_label)

                # Add color zones to histogram background
                if 'zones' in locals():
                    for zone_min, zone_max, zone_color, label in zones:
                        # Fill histogram background in zones
                        ax_hist.axvspan(zone_min, zone_max, alpha=0.05, color=zone_color)

                ax_hist.set_xlabel(field_title)
                ax_hist.set_ylabel("Density")
                ax_hist.set_title(f"{field_title} Distribution")
                ax_hist.grid(True, alpha=0.3)

                # Add histogram legend if we have reference lines
                if 'reference_values' in locals() and reference_values:
                    ax_hist.legend(loc='best', fontsize='small')

                # Add summary statistics as text on histogram
                stats_text = ".2e" if field_name == "potential_energy" else ".3f"
                mean_val = np.mean(field_data)
                std_val = np.std(field_data)
                ax_hist.text(0.02, 0.98, f'Mean: {mean_val:{stats_text}}\nStd: {std_val:{stats_text}}',
                           transform=ax_hist.transAxes, fontsize='small',
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plot_idx += 1

                # Clean up local variables for next iteration
                if 'reference_values' in locals():
                    del reference_values
                if 'zones' in locals():
                    del zones

            # Fill unused subplots if any
            total_subplots = axes.size
            used_subplots = plot_idx * 2
            for i in range(used_subplots, total_subplots):
                row = i // axes.shape[1]
                col = i % axes.shape[1]
                axes[row, col].axis('off')

            plt.tight_layout()
            plt.savefig(f"{outdir}/nuts_extra_fields_diagnostics.png", dpi=150, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error generating NUTS-specific diagnostics: {e}")
