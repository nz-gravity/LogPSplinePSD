"""
Script to visualize delta and phi prior distributions.

The hierarchical prior structure is:
  - delta ~ Gamma(alpha_delta, beta_delta)
  - phi | delta ~ Gamma(alpha_phi, delta * beta_phi)

This script shows how delta and phi marginal priors behave under different
alpha_delta and beta_delta values.


# Delta/Phi Prior Visualization - Summary

## Overview

The script `test_plot_delta_phi_priors.py` creates comprehensive visualizations of how the delta (δ) and phi (φ) priors behave under different hyperparameter choices.

The hierarchical prior structure used in this project is:
- **δ ~ Gamma(α_δ, β_δ)**
- **φ | δ ~ Gamma(α_φ, δ·β_φ)**

## Key Insights

### When α_δ and β_δ are **SMALL** (default: 1e-4)

**Delta prior:**
- Highly dispersed with most mass near 0
- Mean(δ) = α_δ / β_δ = 1e-4 / 1e-4 = 1
- But the variance is huge: Var(δ) = α_δ / β_δ² = 1e4
- Coefficient of variation: CV = 100 (very noisy)
- Results in **weakly informative** priors on the spline smoothing parameters
- Allows the data to dominate the prior
- Typical for exploratory analyses

**Phi (conditional on δ):**
- When δ is small, the conditional distribution φ|δ ~ Gamma(α_φ, δ·β_φ) becomes more dispersed
- The rate parameter δ·β_φ becomes very small
- This creates a long-tailed distribution in φ
- Effectively places weak constraints on the spline flexibility

### When α_δ and β_δ are **LARGE** (e.g., 1.0)

**Delta prior:**
- Much more concentrated distribution
- Mean(δ) = 1.0
- Variance(δ) = 1.0
- CV ≈ 1.0 (well-defined prior belief)
- Results in **more informative** priors
- Regularizes the smoothing parameters more strongly
- Useful when you have prior information about smoothness

**Phi (conditional on δ):**
- When δ is larger, the conditional distribution φ|δ becomes more concentrated
- The rate parameter δ·β_φ is larger, leading to smaller expected values of φ
- More restrictive on spline flexibility

## Generated Plot Files

1. **priors_small_alpha_delta,_beta_delta_(default_1e-4).png**
   - Shows default behavior (α_δ = 1e-4, β_δ = 1e-4)
   - 4 subplots: delta prior, marginal phi, conditional phi, summary statistics

2. **priors_large_alpha_delta,_beta_delta_(1.0).png**
   - Shows highly informative setting (α_δ = 1.0, β_δ = 1.0)
   - Same 4 subplots structure

3. **priors_comparison_small_vs_large.png**
   - Direct side-by-side comparison of both extremes
   - Clearly shows the qualitative difference in prior behavior

4. **priors_intermediate_values.png**
   - Shows intermediate settings (1e-3, 0.01, 0.1)
   - Useful for understanding the full spectrum

5. **priors_mean_variance_behavior.png**
   - Log-log plots showing how mean and variance scale with α_δ
   - For various fixed β_δ values
   - Shows that:
     - Mean scales linearly with α_δ: E[δ] ∝ α_δ / β_δ
     - Variance scales with α_δ but inversely with β_δ²: Var[δ] ∝ α_δ / β_δ²

## How to Use This Information

### Choose small α_δ, β_δ (like 1e-4) if:
- You want weakly informative priors
- You have limited prior knowledge about smoothness
- You're doing exploratory data analysis
- You want the data to dominate

### Choose large α_δ, β_δ (like 1.0) if:
- You have strong prior beliefs about smoothness
- You want to regularize against overfitting
- You're using empirical Bayes methods
- You want stable estimates with small sample sizes

### Tune intermediate values:
- Use the mean/variance behavior plot to understand scaling
- Balance prior informativeness with data-driven estimation
- Consider sensitivity analysis: run inference with a range of values

## Technical Notes

The marginal prior on φ is computed by numerical integration:
$$p(\phi) = \int p(\phi|\delta) p(\delta) d\delta$$

The conditional priors show how φ|δ changes as δ varies, which illustrates the coupling between the two hyperparameters.

This hierarchical structure enables:
1. **Adaptive regularization**: δ controls the overall smoothing strength
2. **Flexibility**: φ provides additional local flexibility given δ
3. **Stable MCMC**: The separation of scales helps with sampling



"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

# matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")


def plot_delta_priors(
    alpha_delta_values, beta_delta_values, ax, alpha_phi=1.0, beta_phi=1.0
):
    """Plot delta prior distributions for different hyperparameter values."""
    ax.set_title(
        r"$\delta$ prior: $\Gamma(\alpha_\delta, \beta_\delta)$",
        fontsize=12,
        fontweight="bold",
    )

    delta_range = np.linspace(0, 0.1, 500)

    for alpha_d in alpha_delta_values:
        for beta_d in beta_delta_values:
            label = rf"$\alpha_\delta={alpha_d}$, $\beta_\delta={beta_d}$"
            prior = stats.gamma(a=alpha_d, scale=1 / beta_d)
            ax.plot(
                delta_range, prior.pdf(delta_range), label=label, linewidth=2
            )

    ax.set_xlabel(r"$\delta$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_phi_marginal_priors(
    alpha_delta_values, beta_delta_values, ax, alpha_phi=1.0, beta_phi=1.0
):
    """
    Plot marginal phi prior distributions.

    Since phi | delta ~ Gamma(alpha_phi, delta * beta_phi), the marginal is obtained
    by integrating over delta. For visualization, we sample delta and conditional phi.
    """
    ax.set_title(
        r"Marginal $\phi$ prior (sampled from hierarchical structure)",
        fontsize=12,
        fontweight="bold",
    )

    # Reduced resolution for speed
    phi_range = np.linspace(0, 50, 200)
    delta_samples = np.linspace(1e-6, 0.1, 100)

    for alpha_d in alpha_delta_values:
        for beta_d in beta_delta_values:
            label = rf"$\alpha_\delta={alpha_d}$, $\beta_\delta={beta_d}$"

            # Prior on delta (vectorized)
            delta_prior_pdf = stats.gamma.pdf(
                delta_samples, a=alpha_d, scale=1 / beta_d
            )

            # Vectorized computation: shape (phi_range, delta_samples)
            phi_pdf = np.zeros_like(phi_range)
            for i, phi_val in enumerate(phi_range):
                rate_param = delta_samples * beta_phi
                conditional_phi_pdf = stats.gamma.pdf(
                    phi_val, a=alpha_phi, scale=1 / rate_param
                )
                phi_pdf[i] = np.trapz(
                    conditional_phi_pdf * delta_prior_pdf, delta_samples
                )

            ax.plot(phi_range, phi_pdf, label=label, linewidth=2)

    ax.set_xlabel(r"$\phi$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_conditional_phi_for_fixed_delta(
    alpha_delta_values, beta_delta_values, ax, alpha_phi=1.0, beta_phi=1.0
):
    """
    Plot conditional phi prior for representative delta values.

    Shows how phi | delta changes with different delta values, for different
    hyperparameter settings.
    """
    ax.set_title(
        r"Conditional $\phi$ prior: $\Gamma(\alpha_\phi, \delta \beta_\phi)$ at fixed $\delta$ values",
        fontsize=12,
        fontweight="bold",
    )

    phi_range = np.linspace(0, 20, 500)

    # Representative delta values
    delta_values = [0.001, 0.01, 0.05]

    for alpha_d in alpha_delta_values:
        for beta_d in beta_delta_values:
            for delta_val in delta_values:
                label = rf"$\alpha_\delta={alpha_d}$, $\beta_\delta={beta_d}$, $\delta={delta_val}$"
                conditional_phi = stats.gamma(
                    a=alpha_phi, scale=1 / (delta_val * beta_phi)
                )
                ax.plot(
                    phi_range,
                    conditional_phi.pdf(phi_range),
                    label=label,
                    linewidth=1.5,
                    alpha=0.8,
                )

    ax.set_xlabel(r"$\phi$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_mean_and_variance_behavior(ax):
    """
    Plot the mean and variance of delta prior as a function of alpha_delta, beta_delta.

    For Gamma(a, b): mean = a/b, variance = a/b^2
    """
    alpha_delta_range = np.logspace(-4, 0, 50)  # 1e-4 to 1

    fig_inner, axes_inner = plt.subplots(1, 2, figsize=(12, 4))

    for beta_d in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        mean_delta = alpha_delta_range / beta_d
        axes_inner[0].loglog(
            alpha_delta_range,
            mean_delta,
            marker="o",
            label=rf"$\beta_\delta={beta_d}$",
            linewidth=2,
        )

    axes_inner[0].set_xlabel(r"$\alpha_\delta$", fontsize=11)
    axes_inner[0].set_ylabel(
        r"$E[\delta] = \alpha_\delta / \beta_\delta$", fontsize=11
    )
    axes_inner[0].set_title(
        r"Mean of $\delta$ prior", fontsize=12, fontweight="bold"
    )
    axes_inner[0].legend(fontsize=9)
    axes_inner[0].grid(True, alpha=0.3, which="both")

    for beta_d in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        var_delta = alpha_delta_range / (beta_d**2)
        axes_inner[1].loglog(
            alpha_delta_range,
            var_delta,
            marker="s",
            label=rf"$\beta_\delta={beta_d}$",
            linewidth=2,
        )

    axes_inner[1].set_xlabel(r"$\alpha_\delta$", fontsize=11)
    axes_inner[1].set_ylabel(
        r"$\mathrm{Var}[\delta] = \alpha_\delta / \beta_\delta^2$", fontsize=11
    )
    axes_inner[1].set_title(
        r"Variance of $\delta$ prior", fontsize=12, fontweight="bold"
    )
    axes_inner[1].legend(fontsize=9)
    axes_inner[1].grid(True, alpha=0.3, which="both")

    return fig_inner


def main(outdir=None):
    """Create comprehensive plots of delta and phi priors.

    Parameters
    ----------
    outdir : str or Path, optional
        Output directory for saving plots. Defaults to a 'test_hyperpriors' subdirectory.
    """
    if outdir is None:
        outdir = Path.cwd() / "test_hyperpriors"
    else:
        outdir = Path(outdir) / "test_hyperpriors"

    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # Define hyperparameter scenarios
    scenarios = [
        {
            "name": "Small alpha_delta, beta_delta (default: 1e-4)",
            "alpha_delta": [1e-4],
            "beta_delta": [1e-4],
        },
        {
            "name": "Large alpha_delta, beta_delta (1.0)",
            "alpha_delta": [1.0],
            "beta_delta": [1.0],
        },
        {
            "name": "Comparison: Small vs Large",
            "alpha_delta": [1e-4, 1.0],
            "beta_delta": [1e-4, 1.0],
        },
        {
            "name": "Intermediate values",
            "alpha_delta": [1e-3, 0.01, 0.1],
            "beta_delta": [1e-3, 0.01, 0.1],
        },
    ]

    # Create plots for each scenario
    for scenario in scenarios:
        print(f"Creating plots for: {scenario['name']}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"{scenario['name']}\n"
            r"Hierarchical Prior: $\delta \sim \Gamma(\alpha_\delta, \beta_\delta)$, "
            r"$\phi | \delta \sim \Gamma(\alpha_\phi, \delta \beta_\phi)$",
            fontsize=13,
            fontweight="bold",
            y=1.00,
        )

        # Plot 1: Delta priors
        plot_delta_priors(
            scenario["alpha_delta"], scenario["beta_delta"], axes[0, 0]
        )

        # Plot 2: Marginal phi priors
        plot_phi_marginal_priors(
            scenario["alpha_delta"], scenario["beta_delta"], axes[0, 1]
        )

        # Plot 3: Conditional phi priors
        plot_conditional_phi_for_fixed_delta(
            scenario["alpha_delta"], scenario["beta_delta"], axes[1, 0]
        )

        # Plot 4: Summary statistics
        axes[1, 1].axis("off")
        summary_text = "Summary Statistics:\n\n"
        for alpha_d in scenario["alpha_delta"]:
            for beta_d in scenario["beta_delta"]:
                mean_delta = alpha_d / beta_d
                var_delta = alpha_d / (beta_d**2)
                cv_delta = (
                    np.sqrt(var_delta) / mean_delta
                    if mean_delta > 0
                    else np.inf
                )

                summary_text += (
                    rf"$\alpha_\delta = {alpha_d}$, $\beta_\delta = {beta_d}$:"
                    + "\n"
                    rf"  Mean($\delta$) = {mean_delta:.6f}" + "\n"
                    rf"  Var($\delta$) = {var_delta:.6f}" + "\n"
                    rf"  CV($\delta$) = {cv_delta:.3f}" + "\n\n"
                )

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        # Save figure
        safe_name = scenario["name"].lower().replace(" ", "_").replace(":", "")
        filename = f"priors_{safe_name}.png"
        filepath = outdir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved to {filepath}")
        plt.close()

    # Create mean/variance behavior plot
    print("\nCreating mean/variance behavior plots...")
    fig_mv = plot_mean_and_variance_behavior(None)
    mv_path = outdir / "priors_mean_variance_behavior.png"
    fig_mv.savefig(mv_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {mv_path}")
    plt.close(fig_mv)

    print("\n✓ All plots created successfully!")


if __name__ == "__main__":
    import sys

    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    main(outdir=outdir)


def test_plot_hyperpriors(outdir):
    """Pytest test to generate prior visualization plots.

    Uses the conftest.py outdir fixture to save plots to test output directory.
    """
    main(outdir=outdir)
