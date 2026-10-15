"""Restricted MP/WDM likelihood diagnosis, separate from spline calibration.

Known covariance shape C0, unknown common variance scale alpha. All methods
use the same proper IG(2,1) prior. Exact observed-time and working power
posteriors are analytic; simulation truth never enters the production fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_bias_audit as audit
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_bias_extension as extension
from experiments.dw_wdm_small import taper_coloured_audit as coloured
from experiments.dw_wdm_small import taper_smoke as taper
from scipy import linalg, optimize, sparse, stats
from wdm_transform import TimeSeries

from log_psplines.example_datasets.ls2_data import LS2Data

ROOT = bench.ROOT / "likelihood_comparison"
VARIANTS = ("rectangular", "boxcar_sparse", "hann_sparse", "hann_all", "wdm")
PRIOR = (2.0, 1.0)


def covariance_shape(generator):
    u = np.arange(bench.N) / bench.N
    if generator == "broadband":
        return bench.variance(u), np.zeros(bench.N)
    b = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200).coefficient(u)
    return 1 + b**2, b


def observed_cholesky(diagonal, lower, observed):
    """Marginal covariance of observed samples, including both endpoints."""
    indices = np.flatnonzero(observed)
    bands = np.zeros((2, len(indices)))
    bands[0] = diagonal[indices]
    adjacent = np.diff(indices) == 1
    bands[1, :-1] = np.where(adjacent, lower[indices[1:]], 0)
    return indices, linalg.cholesky_banded(bands, lower=True)


def exact_quadratic(records, indices, cholesky):
    whitened = linalg.solve_banded((1, 0), cholesky, records[:, indices].T)
    return np.sum(whitened**2, axis=0)


def posterior_quantiles(q, nu):
    a0, b0 = PRIOR
    return stats.invgamma.ppf(
        np.array([0.05, 0.5, 0.95])[:, None],
        a=a0 + nu / 2,
        scale=b0 + np.asarray(q)[None, :] / 2,
    )


def mp_rows(data, variant):
    h = taper.weights("hann" if variant.startswith("hann") else "boxcar")
    centres = np.rint(data.time * bench.N).astype(int)
    j = np.arange(len(h))
    values = np.exp(-2j * np.pi * data.frequency[:, None] * bench.DT * j)
    values *= h / np.linalg.norm(h)
    return sparse.csr_matrix(
        (
            values.ravel(),
            (
                np.repeat(np.arange(len(centres)), len(h)),
                (centres[:, None] - bench.M + j).ravel(),
            ),
        ),
        shape=(len(centres), bench.N),
    )


def mp_moments(rows, diagonal, lower, reference):
    cov = sparse.diags(
        (lower[1:], diagonal, lower[1:]), (-1, 0, 1), format="csr"
    )
    projected = rows @ cov
    k = projected @ rows.conjugate().T
    p = projected @ rows.T
    weights = sparse.diags(1 / reference)
    power_cov = abs(k).power(2) + abs(p).power(2)
    return float(2 * np.sum(k.diagonal().real / reference)), float(
        4 * (weights @ power_cov @ weights).sum()
    )


def prepare(generator, mask, variant, diagonal, lower):
    saved = extension.saved_mask(mask)
    observed, rho = saved["observed"], saved["rho"]
    adapter = "rectangular" if variant == "hann_all" else variant
    data = extension.observations(np.zeros(bench.N), adapter, mask, generator)
    if data.is_grid:
        uu, ff = np.meshgrid(data.time, data.frequency, indexing="ij")
    else:
        uu, ff = data.time, data.frequency
    active = data.counts > 0
    u, f = uu[active], ff[active]
    if generator == "broadband":
        point = 2 * bench.DT * bench.variance(u)
    else:
        b0 = LS2Data(
            n_samples=bench.N, fs=1 / bench.DT, seed=8200
        ).coefficient(u)
        point = (
            2
            * bench.DT
            * (1 + b0**2 + 2 * b0 * np.cos(2 * np.pi * f * bench.DT))
        )
    output = {
        "u": u,
        "f": f,
        "point": point,
        "nu": float(data.counts[active].sum()),
        "observed": observed,
    }
    if variant == "wdm":
        native_frequency = np.arange(1, bench.NF) / (2 * bench.NF * bench.DT)
        allowed = np.isin(native_frequency, data.frequency)
        if generator == "broadband":
            _, _, masked = gap.kernel_expectations(observed)
            means = 2 * bench.DT * masked / np.maximum(rho, 1e-30)
        else:
            means = coloured.expected_wdm(lower, observed, rho)
        output.update(
            mean=means[:, allowed][active], rho=rho[:, allowed][active]
        )
        t, frequency = np.nonzero(active)
        output.update(
            t_index=t + 1, f_index=np.flatnonzero(allowed)[frequency] + 1
        )
        for p in np.linspace(0, len(u) - 1, 5, dtype=int):
            row = (
                bench.wdm_row(
                    int(output["t_index"][p]), int(output["f_index"][p])
                )
                * observed
                / np.sqrt(output["rho"][p])
            )
            projected_mean = (
                2
                * bench.DT
                * (
                    row**2 @ diagonal
                    + 2 * np.sum(row[1:] * row[:-1] * lower[1:])
                )
            )
            np.testing.assert_allclose(
                output["mean"][p], projected_mean, rtol=3e-12, atol=3e-12
            )
    else:
        rows = mp_rows(data, variant)
        # Explicit finite tridiagonal covariance is also used for covariance moments.
        covariance = sparse.diags(
            (lower[1:], diagonal, lower[1:]), (-1, 0, 1), format="csr"
        )
        marginal = np.asarray(
            (rows @ covariance).multiply(rows.conjugate()).sum(axis=1)
        ).real.ravel()
        output.update(rows=rows, mean=2 * bench.DT * marginal)
        output["exact_moments"] = {
            kind: mp_moments(rows, diagonal, lower, reference / (2 * bench.DT))
            for kind, reference in (("point", point), ("mean", output["mean"]))
        }
    output["expected_q_point"] = float(
        np.sum(data.counts[active] * output["mean"] / point)
    )
    return output


def population_projection(generator, prepared):
    """Noiseless expected-working-likelihood optimum; no prior or posterior."""
    model = extension.model(generator)
    u = np.linspace(0.05, 0.95, 81)
    f = np.linspace(0.05, 0.45, 81)
    bt, bf = model.time.design_at(u), model.frequency.design_at(f)
    score = np.einsum("ti,fj->tfij", bt, bf).reshape(len(u) * len(f), -1)
    target = np.log(extension.truth(generator, u, f)).ravel()
    # The interior scoring grid omits some boundary basis supports. Truncate
    # numerical null directions rather than extrapolating huge coefficients.
    initial = linalg.lstsq(score, target, cond=1e-12)[0]
    records = {}
    for variant, item in prepared.items():
        design = np.einsum(
            "pi,pj->pij",
            model.time.design_at(item["u"]),
            model.frequency.design_at(item["f"]),
        ).reshape(len(item["u"]), -1)
        singular = linalg.svdvals(design)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
        means = item["mean"]

        def fun(coef, design=design, means=means):
            eta = design @ coef
            ratio = means * np.exp(-eta)
            return float(np.mean(eta + ratio)), design.T @ (1 - ratio) / len(
                eta
            )

        def hess(coef, design=design, means=means):
            ratio = means * np.exp(-(design @ coef))
            return design.T @ (design * ratio[:, None]) / len(means)

        result = optimize.minimize(
            fun,
            initial,
            jac=True,
            hess=hess,
            method="trust-exact",
            options={"gtol": 2e-9, "maxiter": 100},
        )
        error = score @ result.x - target
        gradient = float(np.max(abs(fun(result.x)[1])))
        assert gradient < 1e-7, (variant, result.message, gradient)
        records[variant] = {
            "rms_log_error": float(np.sqrt(np.mean(error**2))),
            "max_absolute_log_error": float(np.max(abs(error))),
            "gradient_max": gradient,
            "iterations": int(result.nit),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "design_rank": rank,
            "coefficients": design.shape[1],
            "coefficient_vector": result.x.tolist(),
            "design_condition_number": float(singular[0] / singular[-1]),
        }
    return {
        "basis_oracle_rms": float(
            np.sqrt(np.mean((score @ initial - target) ** 2))
        ),
        "methods": records,
    }


def summarize(q, nu, expected, analytic_variance=None):
    quantiles = posterior_quantiles(q, nu)
    covered = (quantiles[0] <= 1) & (quantiles[2] >= 1)
    mean, variance = float(np.mean(q)), float(np.var(q, ddof=1))
    mean_se = float(np.std(q, ddof=1) / np.sqrt(len(q)))
    assert abs(mean - expected) < 6 * mean_se
    # MC uncertainty of the variance estimator, using its fourth central moment.
    fourth = float(np.mean((q - mean) ** 4))
    variance_se = np.sqrt(max(0, fourth - variance**2) / len(q))
    effective = 2 * expected**2 / variance
    result = {
        "nominal_nu": nu,
        "expected_q": expected,
        "expected_scale_mle": expected / nu,
        "record_mean_q": mean,
        "record_mean_q_mc_se": mean_se,
        "record_variance_q": variance,
        "record_variance_q_mc_se": float(variance_se),
        "effective_nu_mc": float(effective),
        "effective_nu_mc_approx_95_interval": [
            float(2 * expected**2 / (variance + 1.96 * variance_se)),
            float(2 * expected**2 / (variance - 1.96 * variance_se)),
        ],
        "coverage90": float(covered.mean()),
        "coverage90_count": int(covered.sum()),
        "coverage90_wilson95": list(audit.wilson(int(covered.sum()), len(q))),
        "median_posterior_width90": float(
            np.median(quantiles[2] - quantiles[0])
        ),
        "mean_posterior_median": float(np.mean(quantiles[1])),
        "scale_mle_rmse": float(np.sqrt(np.mean((q / nu - 1) ** 2))),
    }
    if analytic_variance is not None:
        result["analytic_variance_q"] = analytic_variance
        result["effective_nu_exact"] = 2 * expected**2 / analytic_variance
        assert abs(variance - analytic_variance) < 6 * variance_se
    return result


def run(records):
    ROOT.mkdir(exist_ok=True)
    source_paths = [
        Path(__file__),
        Path(bench.__file__),
        Path(extension.__file__),
        Path(gap.__file__),
        Path(taper.__file__),
    ]
    source_paths += [
        bench.ROOT.parents[1] / "src/log_psplines/inference/power.py"
    ]
    source_paths += [
        bench.ROOT.parents[1] / "src/log_psplines/likelihoods/whittle.py",
        bench.ROOT.parents[1]
        / "src/log_psplines/preprocessing/moving_periodogram.py",
        bench.ROOT.parents[1] / "src/log_psplines/preprocessing/wdm.py",
    ]
    protocol = {
        "date": "2026-10-14",
        "records_per_generator": records,
        "experiment": "Exploratory likelihood diagnosis; not a preregistered confirmation run",
        "geometry": {
            "N": bench.N,
            "dt": bench.DT,
            "m": bench.M,
            "thin": bench.THIN,
            "nt": bench.NT,
            "nf": bench.NF,
        },
        "rng_seed_by_generator": {"broadband": 2026101401, "LS2": 2026101402},
        "masks": ["complete", "single"],
        "variants": list(VARIANTS),
        "hann_all_control": "Hann at the original retained rectangular sites; restricted scale/population diagnosis only, no surface posterior fits.",
        "alpha_truth": 1.0,
        "prior": "InverseGamma(shape=2,scale=1)",
        "scope": "Known covariance shape, unknown single scale. Separate restricted reference, not free-surface calibration.",
        "fit_bands": {
            "broadband": "original complete interior schedules",
            "LS2": [0.04, 0.47],
        },
        "exact_reference": "Observed-time marginal Gaussian with finite diagonal/MA(1) covariance; no periodic innovation wrap.",
        "population_projection": "Complete data, noiseless expected working objective, common generator-specific spline basis, no prior.",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "input_masks": {
            mask: {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                for name, array in extension.saved_mask(mask).items()
            }
            for mask in ("complete", "single")
        },
        "versions": {
            package: version(package)
            for package in ("numpy", "scipy", "wdm_transform")
        },
        "python": platform.python_version(),
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    snapshots = ROOT / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for path in source_paths:
        (
            snapshots / (protocol["source_sha256"][str(path)] + ".py")
        ).write_bytes(path.read_bytes())
    results, arrays = {}, {}
    started = perf_counter()
    for generator in ("broadband", "LS2"):
        diagonal, lower = covariance_shape(generator)
        prepared, reference, qs = {}, {}, {}
        for mask in ("complete", "single"):
            observed = extension.saved_mask(mask)["observed"]
            reference[mask] = observed_cholesky(diagonal, lower, observed)
            qs[mask] = {"exact": []}
            prepared[mask] = {}
            for variant in VARIANTS:
                prepared[mask][variant] = prepare(
                    generator, mask, variant, diagonal, lower
                )
                qs[mask][variant + "_point"] = []
                qs[mask][variant + "_mean"] = []
        population = population_projection(generator, prepared["complete"])
        rng = np.random.default_rng(
            protocol["rng_seed_by_generator"][generator]
        )
        for start in range(0, records, 64):
            size = min(64, records - start)
            innovations = rng.normal(size=(size, bench.N + 1))
            if generator == "broadband":
                x = innovations[:, 1:] * np.sqrt(diagonal)
            else:
                x = innovations[:, 1:] + innovations[:, :-1] * lower
            for mask in ("complete", "single"):
                qs[mask]["exact"].extend(exact_quadratic(x, *reference[mask]))
                wdm = prepared[mask]["wdm"]
                for variant in VARIANTS[:-1]:
                    item = prepared[mask][variant]
                    powers = 4 * bench.DT * abs(item["rows"] @ x.T) ** 2
                    for kind in ("point", "mean"):
                        qs[mask][variant + "_" + kind].extend(
                            np.sum(powers / item[kind][:, None], axis=0)
                        )
                powers = []
                for record in x:
                    coeff = np.asarray(
                        TimeSeries(
                            np.where(wdm["observed"], record, 0.0), dt=bench.DT
                        )
                        .to_wdm(nt=bench.NT)
                        .coeffs
                    )[0]
                    power = (
                        2
                        * bench.DT
                        * coeff[wdm["t_index"], wdm["f_index"]] ** 2
                        / bench.N
                        / wdm["rho"]
                    )
                    powers.append(power)
                powers = np.asarray(powers)
                for kind in ("point", "mean"):
                    qs[mask]["wdm_" + kind].extend(
                        np.sum(powers / wdm[kind], axis=1)
                    )
                if start == 0:
                    # Reconcile independent row/forward calculations against public adapters.
                    for variant in VARIANTS:
                        if variant == "hann_all":
                            item = prepared[mask][variant]
                            # Direct local DFT, independent of sparse row multiplication.
                            h = taper.weights("hann")
                            for p in np.linspace(
                                0, len(item["u"]) - 1, 9, dtype=int
                            ):
                                centre = round(item["u"][p] * bench.N)
                                window = x[
                                    0, centre - bench.M : centre + bench.M + 1
                                ]
                                coeff = np.sum(
                                    window
                                    * h
                                    * np.exp(
                                        -2j
                                        * np.pi
                                        * item["f"][p]
                                        * bench.DT
                                        * np.arange(len(h))
                                    )
                                ) / np.linalg.norm(h)
                                np.testing.assert_allclose(
                                    coeff,
                                    (item["rows"][p] @ x[0])[0],
                                    atol=3e-13,
                                )
                            continue
                        actual = extension.observations(
                            x[0], variant, mask, generator
                        )
                        item = prepared[mask][variant]
                        a = actual.counts > 0
                        for kind in ("point", "mean"):
                            np.testing.assert_allclose(
                                np.sum(actual.power[a] / item[kind]),
                                qs[mask][variant + "_" + kind][0],
                                rtol=3e-13,
                            )
            if start % 512 == 0:
                print(
                    json.dumps(
                        {
                            "generator": generator,
                            "records_done": start + size,
                            "total": records,
                        }
                    ),
                    flush=True,
                )
        results[generator] = {"population_projection": population, "masks": {}}
        for mask in ("complete", "single"):
            n_observed = len(reference[mask][0])
            case = {"observed_samples": n_observed, "methods": {}}
            for variant, q in qs[mask].items():
                q = np.asarray(q)
                arrays[f"{generator}_{mask}_{variant}"] = q
                if variant == "exact":
                    case["methods"][variant] = summarize(
                        q, n_observed, n_observed, 2 * n_observed
                    )
                    continue
                name, kind = variant.rsplit("_", 1)
                item = prepared[mask][name]
                expected = (
                    item["nu"] if kind == "mean" else item["expected_q_point"]
                )
                moments = item.get("exact_moments", {}).get(kind)
                case["methods"][variant] = summarize(
                    q,
                    item["nu"],
                    expected,
                    None if moments is None else moments[1],
                )
            results[generator]["masks"][mask] = case
    results["elapsed_seconds"] = perf_counter() - started
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(ROOT / "record_quadratics.npz", **arrays)
    figures(results)
    return results


def figures(results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    variants = (
        "exact",
        "rectangular_mean",
        "boxcar_sparse_mean",
        "hann_sparse_mean",
        "hann_all_mean",
        "wdm_mean",
    )
    labels = (
        "Exact time",
        "Original rect.",
        "Matched rect.",
        "Sparse Hann",
        "All-rung Hann",
        "WDM",
    )
    for row, generator in enumerate(("broadband", "LS2")):
        for col, mask in enumerate(("complete", "single")):
            ax = axes[row, col]
            methods = results[generator]["masks"][mask]["methods"]
            for i, name in enumerate(variants):
                result = methods[name]
                lo, hi = result["coverage90_wilson95"]
                ax.errorbar(
                    i,
                    result["coverage90"],
                    yerr=[
                        [result["coverage90"] - lo],
                        [hi - result["coverage90"]],
                    ],
                    fmt="o",
                    color="#2866a2",
                    capsize=3,
                )
            ax.axhline(0.9, color="black", ls="--", lw=1)
            ax.set(
                xticks=np.arange(6),
                xticklabels=labels,
                ylim=(0.6, 0.96),
                title=f"{generator}, {mask}",
                ylabel="Repeated-record coverage of 90% interval",
            )
            ax.tick_params(axis="x", rotation=18)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Known covariance shape, unknown scale: exact-mean working posteriors\n95% Wilson Monte Carlo intervals; separate from spline-surface calibration"
    )
    fig.savefig(ROOT / "scale_coverage.pdf")
    fig.savefig(ROOT / "scale_coverage.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=4096)
    args = parser.parse_args()
    if args.records < 512:
        parser.error("Use at least 512 independent records per generator")
    run(args.records)
