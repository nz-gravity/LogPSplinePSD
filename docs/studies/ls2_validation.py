"""Bounded multi-realization LS2 check in native WDM variance units.

Run from the repo root with .venv/bin/python docs/studies/ls2_validation.py.
Three independent 512-sample realizations, two centered chains, 250 warmup
and 250 draws. An independent 512-realization Monte Carlo mean supplies the
transform-domain reference, avoiding an assumed PSD-to-WDM normalization.
Pointwise interval coverage here is descriptive, not a calibration claim.
"""

from pathlib import Path

import jax
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary

from log_psplines import (
    LogPSpline,
    PowerSplineConfig,
    SplineBasis,
    TimeSeries,
    fit,
)
from log_psplines.preprocessing.wdm import wdm_periodogram


def simulate_ls2(rng: np.random.Generator, n: int = 512) -> np.ndarray:
    """Tang LS2 MA(1), matching the sibling WDM source's indexing."""
    noise = rng.normal(size=n + 2)
    time = np.arange(n) / n
    coefficient = 1.1 * np.cos(1.5 - np.cos(4 * np.pi * time))
    return noise[1 : n + 1] + coefficient * noise[:n]


def observations(rng: np.random.Generator):
    values = simulate_ls2(rng)
    return wdm_periodogram(
        TimeSeries(values, np.arange(len(values)) * 0.1), nt=32
    )


def main() -> None:
    output = (
        Path(__file__).resolve().parents[2]
        / "tests/test_output/ls2_validation"
    )
    output.mkdir(parents=True, exist_ok=True)
    # Independent reference realizations are never used to initialize or fit.
    rng = np.random.default_rng(901)
    references = np.asarray([observations(rng).power for _ in range(512)])
    target = references.mean(axis=0)
    mc_standard_error = references.std(axis=0, ddof=1) / np.sqrt(
        len(references)
    )
    np.savez_compressed(
        output / "reference.npz", mean=target, standard_error=mc_standard_error
    )
    rows = []
    with jax.enable_x64(True):
        for seed in (0, 1, 2):
            data = observations(np.random.default_rng(seed))
            model = LogPSpline(
                SplineBasis.from_grid(data.frequency / data.frequency[-1], 4),
                time=SplineBasis.from_grid(data.time, 4),
            )
            config = PowerSplineConfig(
                centered=True,
                num_chains=2,
                n_warmup=250,
                n_samples=250,
                max_tree_depth=8,
                seed=seed + 100,
                progress_bar=False,
            )
            result = fit(data, config, model=model)
            result.save(str(output / f"seed_{seed}"))
            latent = {
                name: result.posterior[name].values
                for name in ("s", "phi_time", "phi_freq")
            }
            diagnostics = summary(latent, group_by_chain=True)
            rhats = np.concatenate(
                [d["r_hat"].ravel() for d in diagnostics.values()]
            )
            ess = np.concatenate(
                [d["n_eff"].ravel() for d in diagnostics.values()]
            )
            lower, median, upper = np.quantile(
                result.psd, [0.05, 0.5, 0.95], axis=(0, 1)
            )
            stats = result.idata["sample_stats"]
            rows.append(
                dict(
                    seed=seed,
                    max_split_rhat=float(np.max(rhats)),
                    min_ess=float(np.min(ess)),
                    divergences=int(stats["diverging"].sum()),
                    depth_hits=int((stats["num_steps"] >= 255).sum()),
                    log_rmse=float(
                        np.sqrt(np.mean(np.log(median / target) ** 2))
                    ),
                    pointwise_90_coverage=float(
                        np.mean((lower <= target) & (target <= upper))
                    ),
                    median_reference_relative_se=float(
                        np.median(mc_standard_error / target)
                    ),
                )
            )
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
            print(rows[-1], flush=True)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
