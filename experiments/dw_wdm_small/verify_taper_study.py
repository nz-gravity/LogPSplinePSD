"""Reload all primary posteriors and independently verify saved study summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from experiments.dw_wdm_small import taper_study as study

from log_psplines import PSDResult


def main() -> None:
    rows = json.loads((study.ROOT / "rows.json").read_text())
    observed = np.load(study.ROOT / "mask.npz")["observed"]
    master = study.bench.master_model()
    checks = []
    for row in rows:
        method = "wdm" if row["variant"] == "wdm" else "dw"
        path = (
            study.ROOT
            / "fits"
            / row["variant"]
            / f"seed{row['seed']}_{method}"
        )
        result = PSDResult.from_netcdf(path / "posterior.nc")
        d = np.load(path / "surface.npz")
        u, f = d["time"], d["freq"]
        assert np.isfinite(result.posterior.weights.values).all()
        assert np.isfinite(d["high"]).all() and np.all(np.exp(d["low"]) > 0)
        assert result.posterior.sizes["chain"] == 2
        assert result.posterior.sizes["draw"] == 1000
        max_error = 0.0
        # Explicit per-draw matrix products, independent of the bulk einsum.
        for ti in (0, 20, 30, 40, 60):
            for fi in (4, 28, 52):
                bt, bf = (
                    master.time.design_at(u[[ti]])[0],
                    master.frequency.design_at(f[[fi]])[0],
                )
                draws = [
                    bt @ w @ bf
                    for w in result.posterior.weights.values.reshape(
                        -1, *master.weights.shape
                    )
                ]
                expected = np.quantile(draws, [0.05, 0.5, 0.95])
                saved = np.array(
                    [d[k][ti, fi] for k in ("low", "median", "high")]
                )
                np.testing.assert_allclose(
                    saved, expected, rtol=2e-12, atol=2e-12
                )
                max_error = max(
                    max_error, float(np.max(abs(saved - expected)))
                )
        band = (f >= 0.05) & (f <= 0.45)
        f = f[band]
        delta = d["median"][:, band] - d["truth"][:, band]
        inclusion = (
            (d["low"][:, band] <= d["truth"][:, band])
            & (d["truth"][:, band] <= d["high"][:, band])
        ).astype(float)
        edges = (
            np.flatnonzero(observed[1:] != observed[:-1]) + 1
        ) / study.bench.N
        inside = ~observed[np.rint(u * study.bench.N).astype(int)]
        near = np.min(abs(u[:, None] - edges), axis=1) <= 0.03
        regions = {
            "all": np.ones(len(u), dtype=bool),
            "gap": inside,
            "edge_observed": near & ~inside,
            "observed_far": ~near & ~inside,
        }
        for region, mask in regions.items():
            support = np.broadcast_to(mask[:, None], delta.shape).astype(float)
            denominator = np.trapezoid(np.trapezoid(support, f, axis=1), u)
            for field, values in (
                ("log_psd_mse", delta**2),
                ("coverage90", inclusion),
                ("signed_log_error", delta),
            ):
                calculated = float(
                    np.trapezoid(np.trapezoid(values * support, f, axis=1), u)
                    / denominator
                )
                np.testing.assert_allclose(
                    calculated, row["regions"][region][field], atol=1e-13
                )
        checks.append(
            {
                "seed": row["seed"],
                "variant": row["variant"],
                "posterior_quantiles_max_error": max_error,
                "regional_quadrature_verified": True,
            }
        )
    protocol = json.loads((study.ROOT / "protocol.json").read_text())
    for path in (
        "src/log_psplines/inference/power.py",
        "src/log_psplines/preprocessing/moving_periodogram.py",
        "src/log_psplines/preprocessing/wdm.py",
    ):
        assert (
            study.bench.digest(
                np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
            )
            == protocol["source_sha256"][path]
        )
    study.bench.write_json(
        study.ROOT / "posterior_validation.json",
        {
            "primary_fits": len(checks),
            "checks": checks,
            "production_source_unchanged": True,
        },
    )
    print(
        f"Verified {len(checks)} posteriors, saved quantiles, regional metrics, and unchanged production sources."
    )


if __name__ == "__main__":
    main()
