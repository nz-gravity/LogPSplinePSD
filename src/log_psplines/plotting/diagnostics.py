import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostics(
    idata: az.InferenceData,
    outdir: str,
    variables: list = ["phi", "delta", "weights"],
    figsize: tuple = (12, 8),
) -> None:
    """
    Plot MCMC diagnostics using arviz.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data from adaptive MCMC
    variables : list
        Variables to plot
    figsize : tuple
        Figure size
    """

    # Trace plots
    az.plot_trace(idata, var_names=variables, figsize=figsize)
    plt.suptitle("Trace plots - Adaptive MCMC")
    plt.tight_layout()
    plt.savefig(f"{outdir}/trace_plots.png")

    # Summary statistics
    print("Summary Statistics:")
    print(az.summary(idata, var_names=variables))

    # Acceptance rate plot
    if "acceptance_rate" in idata.sample_stats:
        fig, ax = plt.subplots(figsize=(10, 4))
        accept_rates = idata.sample_stats.acceptance_rate.values.flatten()
        ax.plot(accept_rates, alpha=0.7)
        ax.axhline(
            idata.attrs.get("target_accept_rate", 0.44),
            color="red",
            linestyle="--",
            label="Target",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title("Acceptance Rate Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/acceptance_rate.png")

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

    # NUTS-specific plots
    if "tree_depth" in idata.sample_stats:
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        tree_depths = idata.sample_stats.tree_depth.values.flatten()
        axes[0, 0].hist(
            tree_depths, bins=range(int(tree_depths.max()) + 2), alpha=0.7
        )
        axes[0, 0].set_xlabel("Tree Depth")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Distribution of Tree Depths")

        if "num_steps" in idata.sample_stats:
            num_steps = idata.sample_stats.num_steps.values.flatten()
            axes[0, 1].hist(num_steps, bins=30, alpha=0.7)
            axes[0, 1].set_xlabel("Number of Steps")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Distribution of Leapfrog Steps")

        if "energy" in idata.sample_stats:
            energy = idata.sample_stats.energy.values.flatten()
            axes[1, 0].plot(energy, alpha=0.7)
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Energy")
            axes[1, 0].set_title("Energy Over Time")

        if "diverging" in idata.sample_stats:
            diverging = idata.sample_stats.diverging.values.flatten()
            n_divergent = np.sum(diverging)
            total_samples = len(diverging)
            divergent_rate = n_divergent / total_samples

            axes[1, 1].bar(
                ["Non-divergent", "Divergent"],
                [total_samples - n_divergent, n_divergent],
            )
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title(
                f"Divergent Transitions ({divergent_rate:.1%})"
            )

        plt.tight_layout()
        plt.savefig(f"{outdir}/nuts_diagnostics.png")
