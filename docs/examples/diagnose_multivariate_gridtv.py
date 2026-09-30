"""Separate basis capacity from the finite-window TVVAR truth approximation.

This is a deterministic diagnostic of the default two-channel example. No
posterior fit uses the oracle coefficients or expected observation matrices.
Run with .venv/bin/python docs/examples/diagnose_multivariate_gridtv.py.
Optionally add --replicates 16 for an independent proper-complex recovery
experiment under the same local spectral covariance (not time-domain TVVAR).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from log_psplines import (
    ANOVALogPSpline,
    PowerConfig,
    SpectralMatrix,
    SplineBasis,
    WishartGridData,
    fit,
    local_wishart_grid,
)
from log_psplines.diagnostics.sampling import sampling_diagnostics
from log_psplines.example_datasets.tvvar_data import TVVARData


def expected_fourier_covariance(
    simulation: TVVARData, segment_length: int
) -> np.ndarray:
    """Exact (T,F,C,C) covariance of the interior local FFT coefficients.

    Propagate the VAR state covariance and within-segment cross covariances;
    initialize from the stationary burn-in law at u=0. The finite burn-in
    transient is negligible for the stable example's 512 burn-in samples.
    """
    channels = simulation.p
    eye = np.eye(channels)
    zero = np.zeros_like(eye)
    innovation = np.zeros((2 * channels, 2 * channels))
    innovation[:channels, :channels] = simulation.sigma

    def transition(u):
        return np.block([[simulation.a1, simulation.a2_at(u)], [eye, zero]])

    covariance = solve_discrete_lyapunov(transition(0), innovation)
    bins = np.arange(1, (segment_length + 1) // 2)
    fourier = np.exp(
        -2j
        * np.pi
        * bins[:, None]
        * np.arange(segment_length)
        / segment_length
    )
    spectra = []
    for block in range(simulation.n_samples // segment_length):
        gamma = np.empty((segment_length, segment_length, channels, channels))
        cross = []
        for index in range(segment_length):
            u = (block * segment_length + index) / simulation.n_samples
            matrix = transition(u)
            covariance = matrix @ covariance @ matrix.T + innovation
            cross = [matrix @ value for value in cross] + [covariance.copy()]
            for previous in range(index + 1):
                gamma[index, previous] = cross[previous][:channels, :channels]
                gamma[previous, index] = gamma[index, previous].T
        spectra.append(
            2
            / (simulation.fs * segment_length)
            * np.einsum("kn,nmij,km->kij", fourier, gamma, fourier.conj())
        )
    return np.asarray(spectra)


def cholesky_fields(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independently recover diagonal logs and signed lower-triangle fields."""
    channels = truth.shape[-1]
    lower = np.linalg.cholesky(truth)
    diagonal = np.diagonal(lower, axis1=-2, axis2=-1).real
    triangular = np.linalg.inv(lower / diagonal[..., None, :])
    row, col = np.tril_indices(channels, k=-1)
    theta = -triangular[..., row, col]
    return 2 * np.log(diagonal), theta


def project_truth(truth: np.ndarray, spline: ANOVALogPSpline) -> np.ndarray:
    """Least-squares oracle projection of Cholesky fields, not a posterior."""
    channels = truth.shape[-1]
    logs, theta = cholesky_fields(truth)
    pairs = theta.shape[-1]
    fields = np.concatenate([logs, theta.real, theta.imag], axis=-1)
    bt, bf = spline.design()
    time_design = np.column_stack([np.ones(len(bt)), bt])
    weights = np.einsum(
        "at,tfu,qf->auq",
        np.linalg.pinv(time_design),
        fields,
        np.linalg.pinv(bf),
    )
    projected = np.einsum("ta,auq,fq->tfu", time_design, weights, bf)
    return SpectralMatrix(channels)(
        projected[..., :channels],
        projected[..., channels : channels + pairs],
        projected[..., channels + pairs :],
    )


def errors(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Equal-cell errors matching the example's spectral components."""
    diagonal = np.diagonal(truth, axis1=-2, axis2=-1).real
    cross = (estimate[..., 0, 1] - truth[..., 0, 1]) / np.sqrt(
        diagonal[..., 0] * diagonal[..., 1]
    )
    return {
        "log_diagonal_rmse": float(
            np.sqrt(
                np.mean(
                    np.log(
                        np.diagonal(estimate, axis1=-2, axis2=-1).real
                        / diagonal
                    )
                    ** 2
                )
            )
        ),
        "normalized_cross_real_rmse": float(np.sqrt(np.mean(cross.real**2))),
        "normalized_cross_imag_rmse": float(np.sqrt(np.mean(cross.imag**2))),
    }


def run(directory: Path, replicates: int = 0) -> None:
    """Save reproducible evidence for the original 16-by-31 example."""
    defaults = TVVARData(n_samples=64, seed=913)
    simulation = TVVARData(
        n_samples=1024,
        fs=16,
        seed=913,
        a1=defaults.a1[:2, :2],
        a2_base=defaults.a2_base[:2, :2],
        sigma=defaults.sigma[:2, :2],
    )
    grid = local_wishart_grid(simulation.ts, 64)
    truth = simulation.get_true_psd(
        time_grid=grid.time / simulation.duration, freq_grid=grid.frequency
    )
    directory.mkdir(parents=True, exist_ok=True)
    expected = expected_fourier_covariance(simulation, 64)
    np.save(directory / "finite-window-expectation.npy", expected)
    report = {
        "finite_window_expectation": errors(expected, truth),
        "projection": {},
    }
    logs, theta = cholesky_fields(truth)
    stationary = np.broadcast_to(
        SpectralMatrix(2)(
            logs.mean(axis=0), theta.real.mean(axis=0), theta.imag.mean(axis=0)
        ),
        truth.shape,
    )

    def log_likelihood(spectrum, statistic):
        # One native observation per cell. This independent matrix calculation
        # is a fixed-parameter diagnostic, not a marginal evidence estimate.
        return (
            -np.linalg.slogdet(spectrum)[1].sum()
            - np.einsum(
                "...ij,...ji->...", np.linalg.inv(spectrum), statistic
            ).real.sum()
        )

    report["fixed_truth_log_likelihood_gain_over_stationary_field_mean"] = (
        float(
            log_likelihood(truth, grid.Y) - log_likelihood(stationary, grid.Y)
        )
    )
    report["expected_log_likelihood_gain"] = float(
        log_likelihood(truth, expected) - log_likelihood(stationary, expected)
    )
    for time_knots, frequency_knots in [(1, 2), (1, 10), (5, 2), (5, 10)]:
        spline = ANOVALogPSpline(
            SplineBasis.from_grid(grid.frequency, frequency_knots, degree=2),
            SplineBasis.from_grid(grid.time, time_knots, degree=2),
        )
        projected = project_truth(truth, spline)
        label = f"time-{time_knots}-frequency-{frequency_knots}"
        np.save(directory / f"oracle-{label}.npy", projected)
        report["projection"][label] = errors(projected, truth)
    payload = json.dumps(report, indent=2) + "\n"
    (directory / "basis-and-window-diagnosis.json").write_text(payload)
    print(payload)
    if replicates:
        from multivariate_gridtv import metrics, plot_comparison

        rng = np.random.default_rng(45)
        shape = (*truth.shape[:2], 2, replicates)
        noise = (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ) / np.sqrt(2)
        data = WishartGridData.from_coefficients(
            np.linalg.cholesky(truth) @ noise, grid.time, grid.frequency
        )
        config = PowerConfig(
            structure="anova",
            degree_time=2,
            degree_freq=2,
            n_interior_knots_time=5,
            n_interior_knots_freq=10,
            n_warmup=800,
            n_samples=1000,
            num_chains=2,
            seed=913,
            target_accept_prob=0.97,
            max_tree_depth=10,
            spectrum_draws=20,
            spectrum_chunk_size=8,
            progress_bar=False,
        )
        result = fit(data, config, true_psd=truth)
        output = directory / "proper-complex-recovery"
        output.mkdir(exist_ok=True)
        result.to_netcdf(output / "result.nc")
        plot_comparison(
            {"native": result},
            truth,
            output,
            subtitle=f"Independent complex fixture: {replicates} replicates per cell",
        )
        recovery = {
            "data": "Independent proper complex vectors; not time-domain TVVAR",
            "replicates_per_cell": replicates,
            "metrics": metrics(result, truth),
            "diagnostics": sampling_diagnostics(result),
            "metadata": result.metadata,
        }
        (output / "report.json").write_text(
            json.dumps(recovery, indent=2) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tests/test-output/gridtv-diagnosis"),
    )
    parser.add_argument("--replicates", type=int, default=0)
    args = parser.parse_args()
    if args.replicates < 0:
        parser.error("--replicates must be nonnegative")
    run(args.outdir, args.replicates)
