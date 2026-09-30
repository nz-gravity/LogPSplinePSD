"""Reproducible TVVAR -> complex FFT -> blocked GridTV comparison.

Run from the checkout with its .venv, e.g.:
JAX_ENABLE_X64=1 .venv/bin/python docs/examples/multivariate_gridtv.py --outdir tests/test-output/tvvar-gridtv
Short chains exercise feasibility; use repeated longer runs for convergence.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer.util import init_to_value, initialize_model

from log_psplines import (
    ANOVALogPSpline,
    PowerConfig,
    PSDResult,
    SpectralMatrix,
    SplineBasis,
    WishartGridData,
    coarse_grain_wishart_grid,
    fit,
    local_wishart_grid,
)
from log_psplines.diagnostics import sampling_diagnostics, spectrum_diagnostics
from log_psplines.example_datasets.tvvar_data import TVVARData
from log_psplines.inference.wishart_grid import prepare_wishart_grid_row
from log_psplines.plotting.results import plot_posterior_spectrum
from log_psplines.preprocessing.knot_locator import (
    wishart_grid_knots,
)


def profile_largest_row(
    data: WishartGridData, spline: ANOVALogPSpline, config: PowerConfig
) -> dict:
    """Explicit compiled log-density/gradient cost, synchronized on each call."""
    model, init = prepare_wishart_grid_row(data, spline, config, data.p - 1)
    start = perf_counter()
    info = initialize_model(
        jax.random.PRNGKey(config.seed),
        model,
        init_strategy=init_to_value(values=init),
    )
    jax.block_until_ready(info.param_info)
    initialization = perf_counter() - start
    value_gradient = jax.jit(jax.value_and_grad(info.potential_fn))
    start = perf_counter()
    compiled = value_gradient.lower(info.param_info.z).compile()
    compilation = perf_counter() - start
    # The timed calls include Python dispatch and synchronization.
    jax.block_until_ready(compiled(info.param_info.z))
    costs = []
    for _ in range(20):
        start = perf_counter()
        value, gradient = jax.block_until_ready(compiled(info.param_info.z))
        costs.append(perf_counter() - start)
    if not np.isfinite(value) or any(
        not np.isfinite(array).all() for array in jax.tree.leaves(gradient)
    ):
        raise ValueError("compiled log density/gradient is not finite")
    return {
        "initialization_seconds": initialization,
        "logdensity_gradient_compile_seconds": compilation,
        "logdensity_gradient_median_seconds": float(np.median(costs)),
        "largest_row_latent_dimension": sum(
            array.size for array in jax.tree.leaves(info.param_info.z)
        ),
    }


def metrics(result: PSDResult, truth: np.ndarray) -> dict:
    """Equal-cell descriptive errors; a single realization is not calibration."""
    median = result.quantiles((50.0,)).values[0]
    diagonal = np.diagonal(truth, axis1=-2, axis2=-1).real
    estimate = np.diagonal(median, axis1=-2, axis2=-1).real
    upper = np.triu_indices(diagonal.shape[-1], 1)
    scale = np.sqrt(diagonal[..., upper[0]] * diagonal[..., upper[1]])
    residual = (median - truth)[..., upper[0], upper[1]] / scale
    coherence = (
        result.spectrum_summary["coherence_quantiles"]
        .sel(percentile=50)
        .values
    )
    return {
        "log_diagonal_rmse": float(
            np.sqrt(np.mean(np.log(estimate / diagonal) ** 2))
        ),
        "normalized_cross_real_rmse": float(
            np.sqrt(np.mean(residual.real**2))
        ),
        "normalized_cross_imag_rmse": float(
            np.sqrt(np.mean(residual.imag**2))
        ),
        "coherence_rmse": float(
            np.sqrt(
                np.mean(
                    (coherence - SpectralMatrix.coherence(truth))[
                        ..., upper[0], upper[1]
                    ]
                    ** 2
                )
            )
        ),
        **spectrum_diagnostics(result),
    }


def plot_comparison(
    results: dict[str, PSDResult],
    truth: np.ndarray,
    directory: Path,
    *,
    subtitle: str | None = None,
) -> None:
    """One middle-time slice of diagonals and the first complex channel pair."""
    first = next(iter(results.values()))
    middle = len(first.time) // 2
    channels = first.spectrum.sizes["channel"]
    fig, axes = plt.subplots(
        channels + 3, 1, figsize=(9, 3 * (channels + 3)), sharex=True
    )
    plt.rcParams.update({"font.size": 12})
    targets = [truth[middle, :, c, c].real for c in range(channels)] + [
        truth[middle, :, 0, 1].real,
        truth[middle, :, 0, 1].imag,
        SpectralMatrix.coherence(truth)[middle, :, 0, 1],
    ]
    labels = [f"S{c + 1}{c + 1} [data²/Hz]" for c in range(channels)] + [
        "Re S12 [data²/Hz]",
        "Im S12 [data²/Hz]",
        "Squared coherence 12",
    ]
    for axis, target, label in zip(axes, targets, labels, strict=True):
        axis.plot(
            first.frequency,
            target,
            color="black",
            label="Pointwise local truth",
        )
        axis.set_ylabel(label)
    for name, result in results.items():
        quantiles = result.quantiles((5.0, 50.0, 95.0)).values
        median = quantiles[1]
        values = [median[middle, :, c, c].real for c in range(channels)] + [
            median[middle, :, 0, 1].real,
            median[middle, :, 0, 1].imag,
            result.spectrum_summary["coherence_quantiles"]
            .sel(percentile=50)
            .values[middle, :, 0, 1],
        ]
        for axis, value in zip(axes, values, strict=True):
            axis.plot(result.frequency, value, label=name)
        if name == next(iter(results)):
            bounds = [
                quantiles[[0, 2], middle, :, c, c].real
                for c in range(channels)
            ] + [
                quantiles[[0, 2], middle, :, 0, 1].real,
                quantiles[[0, 2], middle, :, 0, 1].imag,
                result.spectrum_summary["coherence_quantiles"]
                .sel(percentile=[5, 95])
                .values[:, middle, :, 0, 1],
            ]
            for axis, bound in zip(axes, bounds, strict=True):
                axis.fill_between(
                    result.frequency,
                    *bound,
                    color="C0",
                    alpha=0.15,
                    label=f"{name} 90% interval",
                )
    for axis in axes[:channels]:
        axis.set_yscale("log")
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Frequency [Hz]")
    title = f"Time = {first.time[middle]:.3g} s"
    fig.suptitle(title if subtitle is None else f"{title}\n{subtitle}")
    fig.tight_layout()
    fig.savefig(directory / "pooling-comparison.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    directory = Path(args.outdir)
    directory.mkdir(parents=True, exist_ok=True)
    # Use the existing simulator's parameter defaults, restricted to C channels.
    defaults = TVVARData(n_samples=64, seed=args.seed)
    simulation = TVVARData(
        n_samples=args.segments * args.segment_length,
        fs=16.0,
        seed=args.seed,
        a1=defaults.a1[: args.channels, : args.channels],
        a2_base=defaults.a2_base[: args.channels, : args.channels],
        sigma=defaults.sigma[: args.channels, : args.channels],
    )
    native = local_wishart_grid(simulation.ts, args.segment_length)
    config = PowerConfig(
        structure="anova",
        degree_time=2,
        degree_freq=2,
        penalty_order_time=2,
        penalty_order_freq=2,
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.frequency_knots,
        n_warmup=args.warmup,
        n_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        progress_bar=False,
        max_tree_depth=10,
        target_accept_prob=args.target_accept,
        centered=args.centered,
        spectrum_draws=min(args.samples, args.spectrum_draws),
        spectrum_chunk_size=args.spectrum_chunk_size,
    )
    modes = {
        "native": (1, 1),
        "frequency-only": (1, args.frequency_bin),
        "time-frequency": (args.time_bin, args.frequency_bin),
    }
    modes = {name: modes[name] for name in args.modes}
    knots = {"time": None, "frequency": None}
    if args.knot_placement == "quantile":
        knots = wishart_grid_knots(
            native,
            args.time_knots,
            args.frequency_knots,
            time_spacing=max(value[0] for value in modes.values())
            * np.diff(native.time).min(),
            frequency_spacing=max(value[1] for value in modes.values())
            * np.diff(native.frequency).min(),
        )
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(
            native.reference_frequency,
            config.n_interior_knots_freq
            if knots["frequency"] is None
            else None,
            interior_knots=knots["frequency"],
            degree=2,
            penalty_order=2,
        ),
        SplineBasis.from_grid(
            native.reference_time,
            config.n_interior_knots_time if knots["time"] is None else None,
            interior_knots=knots["time"],
            degree=2,
            penalty_order=2,
        ),
        sigma_eta_prior=config.interaction_scale,
    )
    truth = simulation.get_true_psd(
        time_grid=native.time / simulation.duration, freq_grid=native.frequency
    )
    np.save(directory / "pointwise-local-truth.npy", truth)
    np.savez(
        directory / "knots.npz",
        time=spline.time.knots,
        frequency=spline.frequency.knots,
    )
    results, report = {}, {}
    for name, (time_bin, frequency_bin) in modes.items():
        data = coarse_grain_wishart_grid(
            native, time_bin=time_bin, frequency_bin=frequency_bin
        )
        benchmark = profile_largest_row(data, spline, config)
        start = perf_counter()
        result = fit(data, config, model=spline, true_psd=truth)
        elapsed = perf_counter() - start
        if np.linalg.eigvalsh(result.spectrum).min() <= 0:
            raise ValueError(
                "reconstruction failed positive-definiteness check"
            )
        results[name] = result
        output = directory / name
        output.mkdir(exist_ok=True)
        result.to_netcdf(output / "result.nc")
        plot_posterior_spectrum(result, output)
        # Label the arithmetic bin average separately from pointwise truth.
        ts = np.arange(0, len(native.time), time_bin)
        fs = np.arange(0, len(native.frequency), frequency_bin)
        sizes = (
            np.diff(np.r_[ts, len(native.time)])[:, None]
            * np.diff(np.r_[fs, len(native.frequency)])[None, :]
        )
        averaged = (
            np.add.reduceat(np.add.reduceat(truth, ts, axis=0), fs, axis=1)
            / sizes[..., None, None]
        )
        np.save(output / "bin-averaged-local-truth.npy", averaged)
        np.save(
            output / "bin-centre-local-truth.npy",
            simulation.get_true_psd(
                time_grid=data.time / simulation.duration,
                freq_grid=data.frequency,
            ),
        )
        rows = sampling_diagnostics(result)["nuts"]
        for row, diagnostic in enumerate(rows):
            sampling_seconds = result.metadata[
                "row_sampling_including_compilation_seconds"
            ][row]
            diagnostic["ess_bulk_min_per_sampling_second"] = (
                diagnostic["ess_bulk_min"] / sampling_seconds
            )
            steps = result.sample_stats[f"n_steps_channel_{row}"].values
            diagnostic["gradient_steps_per_retained_draw"] = float(
                steps.mean()
            )
        report[name] = {
            **benchmark,
            "fit_total_seconds": elapsed,
            "likelihood_grid": list(data.counts.shape),
            "observation_count": float(data.counts.sum()),
            "factor_bytes": data.u_re.nbytes + data.u_im.nbytes,
            "counts_bytes": data.counts.nbytes,
            "coefficient_basis_bytes": spline.time_basis.nbytes
            + np.asarray(spline.frequency.basis).nbytes,
            "posterior_bytes": sum(
                var.nbytes for var in result.posterior.values()
            ),
            "summary_bytes": sum(
                var.nbytes for var in result.spectrum_summary.values()
            ),
            "invariants": {
                "minimum_eigenvalue": float(
                    np.linalg.eigvalsh(result.spectrum).min()
                ),
                "finite": bool(np.isfinite(result.spectrum).all()),
            },
            "metrics": metrics(result, truth),
            "nuts": rows,
            **result.metadata,
        }
        print(
            f"{name}: {elapsed:.2f}s; grid={data.counts.shape}; metrics={report[name]['metrics']}",
            flush=True,
        )
    plot_comparison(results, truth, directory)
    payload = {
        "settings": vars(args),
        "jax_version": jax.__version__,
        "jax_x64": jax.config.x64_enabled,
        "device": str(jax.devices()[0]),
        "interpretation": "One realization; descriptive errors and feasibility, not convergence or calibration. HMC phase timings include compilation. Compiled gradient timing is separate.",
        "results": report,
    }

    # ArviZ emits NaN for unavailable short-chain diagnostics; JSON uses null.
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (directory / "benchmark.json").write_text(
        json.dumps(clean(payload), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="tests/test-output/tvvar-gridtv")
    parser.add_argument("--channels", type=int, choices=(2, 3), default=2)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--time-knots", type=int, default=5)
    parser.add_argument("--frequency-knots", type=int, default=10)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument(
        "--knot-placement", choices=("uniform", "quantile"), default="uniform"
    )
    parser.add_argument("--time-bin", type=int, default=2)
    parser.add_argument("--frequency-bin", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("native", "frequency-only", "time-frequency"),
        default=["native", "frequency-only", "time-frequency"],
    )
    parser.add_argument("--spectrum-draws", type=int, default=20)
    parser.add_argument("--spectrum-chunk-size", type=int, default=8)
    run(parser.parse_args())
