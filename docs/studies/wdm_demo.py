"""Public-facing WDM time-varying PSD demo.

Simulates the Tang LS2 signal (an MA(1) whose coefficient drifts smoothly in
time, so its spectral peak moves too), fits it with the scalar
time-frequency ``LogPSpline`` model through the normal ``fit`` entry point,
and saves two documentation figures:

``wdm-demo-fit.png``
    Input series, WDM periodogram, and posterior median log-PSD surface.
``wdm-demo-precision.png``
    The tensor-product roughness precision ``Q`` used by the prior.

Run with ``.venv/bin/python docs/studies/wdm_demo.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from log_psplines import (
    LogPSpline,
    PowerSplineConfig,
    SplineBasis,
    TimeSeries,
    fit,
)
from log_psplines.preprocessing.wdm import wdm_periodogram

DOCS = Path(__file__).resolve().parents[1]
STATIC = DOCS / "_static"


def simulate_ls2(rng: np.random.Generator, n: int = 2048) -> np.ndarray:
    """Tang LS2 MA(1) with a sinusoidally time-varying coefficient."""
    noise = rng.normal(size=n + 2)
    time = np.arange(n) / n
    coefficient = 1.1 * np.cos(1.5 - np.cos(4 * np.pi * time))
    return noise[1 : n + 1] + coefficient * noise[:n]


def plot_fit(series: TimeSeries, data, result, *, path: Path) -> None:
    import matplotlib.pyplot as plt

    median = np.median(result.psd, axis=(0, 1))
    fig, axes = plt.subplots(
        3, 1, figsize=(7.5, 8), sharex=False,
        gridspec_kw={"height_ratios": [1, 1.4, 1.4]},
    )

    axes[0].plot(series.t, series.data[:, 0], lw=0.5, color="0.3")
    axes[0].set(title="Simulated time-varying signal", xlabel="time [s]")
    axes[0].set_xlim(series.t[0], series.t[-1])

    mesh0 = axes[1].pcolormesh(
        data.time, data.frequency,
        np.log10(np.maximum(data.power.T, 1e-12)),
        shading="nearest", cmap="magma",
    )
    axes[1].set(title="WDM periodogram", ylabel="frequency [Hz]")
    fig.colorbar(mesh0, ax=axes[1], pad=0.02, label="log10 power")

    mesh1 = axes[2].pcolormesh(
        result.time, result.frequency, np.log(median).T, shading="auto",
        cmap="viridis",
    )
    axes[2].set(
        title="Posterior median log-PSD (log-P-spline fit)",
        xlabel="rescaled time",
        ylabel="frequency [Hz]",
    )
    fig.colorbar(mesh1, ax=axes[2], pad=0.02, label="log PSD")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_precision(*, path: Path) -> None:
    """Visualise Q = phi_time * (I_Kf ⊗ Qt) + phi_freq * (Qf ⊗ I_Kt)."""
    import matplotlib.pyplot as plt

    time_basis = SplineBasis.from_grid(np.linspace(0.0, 1.0, 64), 4)
    freq_basis = SplineBasis.from_grid(np.linspace(0.0, 1.0, 96), 6)
    Qt, Qf = np.asarray(time_basis.penalty), np.asarray(freq_basis.penalty)
    Kt, Kf = Qt.shape[0], Qf.shape[0]

    phi_time, phi_freq = 1.0, 0.3
    Q = phi_time * np.kron(np.eye(Kf), Qt) + phi_freq * np.kron(Qf, np.eye(Kt))

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    mesh = ax.imshow(np.log10(np.abs(Q) + 1e-8), cmap="cividis")
    ax.set(
        title=r"Roughness precision $Q$ (log$_{10}|Q_{ij}|$)",
        xlabel=r"$\mathrm{vec}(W)$ index",
        ylabel=r"$\mathrm{vec}(W)$ index",
    )
    fig.colorbar(mesh, ax=ax, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    values = simulate_ls2(rng)
    series = TimeSeries(values, np.arange(len(values)) * 0.1)
    data = wdm_periodogram(series, nt=32)

    model = LogPSpline(
        frequency=SplineBasis.from_grid(data.frequency / data.frequency[-1], 4),
        time=SplineBasis.from_grid(data.time, 4),
    )
    config = PowerSplineConfig(
        n_warmup=250, n_samples=250, seed=4, progress_bar=False
    )
    result = fit(data, config, model=model)

    plot_fit(series, data, result, path=STATIC / "wdm-demo-fit.png")
    plot_precision(path=STATIC / "wdm-demo-precision.png")


if __name__ == "__main__":
    main()
