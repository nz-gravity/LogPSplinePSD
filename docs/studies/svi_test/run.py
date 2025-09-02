import jax.numpy as jnp
from svi import get_posterior_predictions, run_svi

from log_psplines.example_datasets import ARData

ar_data = ARData(order=4, duration=4.0, fs=128.0, sigma=1.0, seed=42)
results = run_svi(
    pdgrm=ar_data.periodogram,
    n_knots=50,
    degree=3,
    diffMatrixOrder=2,
    num_steps=50_000,
    learning_rate=1e-4,
    guide_type="diag",
    rng_key=42,
)

### plot metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results["losses"][0:500], label="Loss", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SVI Loss Over Iterations")
plt.savefig("loss_plot.png")

### plot PPC
import matplotlib.pyplot as plt

posterior_predictions = get_posterior_predictions(results, n_samples=1000)
plt.figure(figsize=(10, 6))
plt.plot(
    ar_data.periodogram.freqs,
    ar_data.periodogram.power,
    label="Observed PSD",
    color="blue",
)
plt.yscale("log")
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

plt.plot(
    ar_data.periodogram.freqs,
    jnp.mean(posterior_predictions["psd_samples"], axis=0),
    label="Posterior Mean PSD",
    color="orange",
)
plt.fill_between(
    ar_data.periodogram.freqs,
    jnp.percentile(posterior_predictions["psd_samples"], 5, axis=0),
    jnp.percentile(posterior_predictions["psd_samples"], 95, axis=0),
    color="lightgray",
    alpha=0.5,
    label="90% Credible Interval",
)

# plot true
plt.plot(
    ar_data.periodogram.freqs,
    ar_data.psd_theoretical,
    label="True PSD",
    color="green",
    linestyle="--",
)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.savefig("posterior_predictions_plot.png")


### plot posteriors
posterior = posterior_predictions["posterior_samples"]
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
# Plot delta posterior
axes[0].hist(
    posterior["delta"], bins=30, density=True, color="blue", alpha=0.7
)
axes[0].set_title("Posterior of Delta")
axes[0].set_xlabel("Delta")
axes[1].hist(
    posterior["phi"], bins=30, density=True, color="orange", alpha=0.7
)
axes[1].set_title("Posterior of Phi")
axes[1].set_xlabel("Phi")
# Plot weights posterior
for i in range(posterior["weights"].shape[1]):
    axes[2].hist(
        posterior["weights"][:, i],
        bins=30,
        density=True,
        alpha=0.5,
        color=f"C{i}",
    )
    axes[2].set_title("Posterior of Weights")
plt.savefig("posteriors_plot.png")
