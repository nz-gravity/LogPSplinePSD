"""Summarize spectral-vs-cholesky knot scoring outputs.

Reads outputs from ``plot_knot_scoring_diagnostics.py`` and writes:
- analysis_summary.json
- analysis_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import arviz as az
import numpy as np


def _read_score_summary(path: str) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(open(path, newline="")))
    out: dict[str, dict[str, float]] = {"spectral": {}, "cholesky": {}}
    for method in ("spectral", "cholesky"):
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        diag_rows = [
            r for r in method_rows if r["component"].startswith("diag_")
        ]
        offdiag_rows = [
            r for r in method_rows if r["component"].startswith("theta_")
        ]

        diag_top10 = (
            np.mean([float(r["top10_mass"]) for r in diag_rows])
            if diag_rows
            else float("nan")
        )
        diag_eff = (
            np.mean([float(r["effective_fraction"]) for r in diag_rows])
            if diag_rows
            else float("nan")
        )
        off_top10 = (
            float(np.mean([float(r["top10_mass"]) for r in offdiag_rows]))
            if offdiag_rows
            else float("nan")
        )
        off_eff = (
            float(
                np.mean([float(r["effective_fraction"]) for r in offdiag_rows])
            )
            if offdiag_rows
            else float("nan")
        )

        out[method] = {
            "diag_top10_mass_mean": float(diag_top10),
            "diag_effective_fraction_mean": float(diag_eff),
            "offdiag_top10_mass_mean": float(off_top10),
            "offdiag_effective_fraction_mean": float(off_eff),
        }
    return out


def _read_posterior_metrics(nc_path: str) -> dict[str, float | None]:
    if not os.path.exists(nc_path):
        return {
            "coverage": None,
            "riae_matrix": None,
            "l2_matrix": None,
            "runtime": None,
        }
    idata = az.from_netcdf(nc_path)
    return {
        "coverage": (
            float(idata.attrs["coverage"])
            if "coverage" in idata.attrs
            else None
        ),
        "riae_matrix": (
            float(idata.attrs["riae_matrix"])
            if "riae_matrix" in idata.attrs
            else None
        ),
        "l2_matrix": (
            float(idata.attrs["l2_matrix"])
            if "l2_matrix" in idata.attrs
            else None
        ),
        "runtime": (
            float(idata.attrs["runtime"]) if "runtime" in idata.attrs else None
        ),
    }


def _fmt(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{ndigits}f}"


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def build_analysis(outdir: str) -> dict[str, Any]:
    score_csv = os.path.join(outdir, "score_concentration_summary.csv")
    if not os.path.exists(score_csv):
        raise FileNotFoundError(f"Missing score summary CSV: {score_csv}")

    score = _read_score_summary(score_csv)
    spectral_nc = os.path.join(
        outdir, "posterior_spectral", "inference_data.nc"
    )
    cholesky_nc = os.path.join(
        outdir, "posterior_cholesky", "inference_data.nc"
    )
    post = {
        "spectral": _read_posterior_metrics(spectral_nc),
        "cholesky": _read_posterior_metrics(cholesky_nc),
    }

    analysis: dict[str, Any] = {
        "score_concentration": score,
        "posterior_metrics": post,
        "deltas_spectral_minus_cholesky": {
            "diag_top10_mass_mean": _delta(
                score["spectral"].get("diag_top10_mass_mean"),
                score["cholesky"].get("diag_top10_mass_mean"),
            ),
            "diag_effective_fraction_mean": _delta(
                score["spectral"].get("diag_effective_fraction_mean"),
                score["cholesky"].get("diag_effective_fraction_mean"),
            ),
            "coverage": _delta(
                post["spectral"]["coverage"], post["cholesky"]["coverage"]
            ),
            "riae_matrix": _delta(
                post["spectral"]["riae_matrix"],
                post["cholesky"]["riae_matrix"],
            ),
            "l2_matrix": _delta(
                post["spectral"]["l2_matrix"], post["cholesky"]["l2_matrix"]
            ),
            "runtime": _delta(
                post["spectral"]["runtime"], post["cholesky"]["runtime"]
            ),
        },
    }
    return analysis


def _analysis_markdown(analysis: dict[str, Any]) -> str:
    score = analysis["score_concentration"]
    post = analysis["posterior_metrics"]
    delta = analysis["deltas_spectral_minus_cholesky"]

    lines = [
        "# Knot Scoring Analysis",
        "",
        "## Score Concentration",
        "",
        "| Method | Diag top10 mass (mean) | Diag effective fraction (mean) | Offdiag top10 mass (mean) | Offdiag effective fraction (mean) |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| spectral | {_fmt(score['spectral'].get('diag_top10_mass_mean'))} | "
            f"{_fmt(score['spectral'].get('diag_effective_fraction_mean'))} | "
            f"{_fmt(score['spectral'].get('offdiag_top10_mass_mean'))} | "
            f"{_fmt(score['spectral'].get('offdiag_effective_fraction_mean'))} |"
        ),
        (
            f"| cholesky | {_fmt(score['cholesky'].get('diag_top10_mass_mean'))} | "
            f"{_fmt(score['cholesky'].get('diag_effective_fraction_mean'))} | "
            f"{_fmt(score['cholesky'].get('offdiag_top10_mass_mean'))} | "
            f"{_fmt(score['cholesky'].get('offdiag_effective_fraction_mean'))} |"
        ),
        "",
        "## Posterior Metrics",
        "",
        "| Method | Coverage | RIAE matrix | L2 matrix | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| spectral | {_fmt(post['spectral']['coverage'])} | "
            f"{_fmt(post['spectral']['riae_matrix'])} | "
            f"{_fmt(post['spectral']['l2_matrix'])} | "
            f"{_fmt(post['spectral']['runtime'], ndigits=2)} |"
        ),
        (
            f"| cholesky | {_fmt(post['cholesky']['coverage'])} | "
            f"{_fmt(post['cholesky']['riae_matrix'])} | "
            f"{_fmt(post['cholesky']['l2_matrix'])} | "
            f"{_fmt(post['cholesky']['runtime'], ndigits=2)} |"
        ),
        "",
        "## Deltas (spectral - cholesky)",
        "",
        f"- diag_top10_mass_mean: {_fmt(delta['diag_top10_mass_mean'])}",
        f"- diag_effective_fraction_mean: {_fmt(delta['diag_effective_fraction_mean'])}",
        f"- coverage: {_fmt(delta['coverage'])}",
        f"- riae_matrix: {_fmt(delta['riae_matrix'])}",
        f"- l2_matrix: {_fmt(delta['l2_matrix'])}",
        f"- runtime (s): {_fmt(delta['runtime'], ndigits=2)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize knot-scoring diagnostics and posterior comparisons."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory produced by plot_knot_scoring_diagnostics.py",
    )
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    analysis = build_analysis(outdir)

    json_path = os.path.join(outdir, "analysis_summary.json")
    md_path = os.path.join(outdir, "analysis_summary.md")
    with open(json_path, "w") as handle:
        json.dump(analysis, handle, indent=2)
    with open(md_path, "w") as handle:
        handle.write(_analysis_markdown(analysis))

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
