"""Independently check exported paired metrics and physical draw-wise averages."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import xarray as xr
from experiments.dw_wdm_small import hann_paper as s
from experiments.dw_wdm_small import hann_paper_lisa as lisa
from scipy.integrate import trapezoid


def main() -> None:
    rows = list(csv.DictReader((s.ROOT / "metrics.csv").open()))
    lookup = {(r["target"], r["region"]): r for r in rows}
    paired = 0
    for r in rows:
        if r["dataset"] == "smooth" and r["mask"] == "single":
            c = lookup[
                (r["target"].replace("_single_", "_complete_"), r["region"])
            ]
            np.testing.assert_allclose(
                float(r["Delta_B"]), float(r["B"]) - float(c["B"]), atol=2e-15
            )
            paired += 1
    ff = np.linspace(0.25, 4.75, 48)
    fraction = np.mean((ff < 3 / 6.5) | (ff > 30 / 6.5))
    for r in rows:
        if r["dataset"] == "LS2" and r["region"] == "all":
            edge = lookup[(r["target"], "edge_prediction")]
            interior = lookup[(r["target"], "within_center_frequency_range")]
            np.testing.assert_allclose(
                float(r["RMSE_logS"]) ** 2,
                fraction * float(edge["RMSE_logS"]) ** 2
                + (1 - fraction) * float(interior["RMSE_logS"]) ** 2,
                atol=2e-14,
            )
    summaries = np.load(s.ROOT / "lisa_summary/summaries.npz")
    basis = np.load(s.LISA_MP / "model_basis.npz")
    spline = lisa.wdm.model_basis(lisa.arguments(s.ROOT / "lisa_summary"))
    days, f = summaries["days"], summaries["frequency_hz"]
    nominal = np.load(s.LISA_WDM / "plot_data.npz")["nominal"]
    inv = s.read(s.ROOT / "inventory.json")
    checked = {}
    for cell in inv:
        if cell["dataset"] != "LISA" or not cell.get("path"):
            continue
        key = f"{cell['mask']}_{cell['method']}"
        post = xr.load_dataset(Path(cell["path"]) / "posterior.nc")
        bt, bf = spline.design(days * 86400 / (lisa.mp.N * lisa.mp.DT), f)
        bt, bf = bt @ basis["U_time"], bf @ basis["U_freq"]
        g = post.g.values.reshape(-1, 16)
        eta = post.eta.values.reshape(-1, 6, 16)
        errors = []
        for j in (0, 60, 180):
            # Matrix products plus direct integration, independent of the
            # summary's full-field einsum and discrete weighted average.
            field = (g @ bf[j])[:, None] + (eta @ bf[j]) @ bt.T
            linear = np.exp(field) * nominal[None, :, j]
            average = trapezoid(linear, x=days, axis=1) / (days[-1] - days[0])
            q = np.quantile(average, [0.05, 0.5, 0.95])
            saved = summaries[key + "_time_average_quantiles"][:, j]
            np.testing.assert_allclose(q, saved, rtol=3e-13, atol=0)
            errors.append(float(np.max(abs(q / saved - 1))))
        # Independently integrate the low-frequency spectrum at the first epoch.
        eta0 = np.tensordot(eta, bt[0], axes=(1, 0))
        linear = np.exp((g + eta0) @ bf.T) * nominal[None, 0, :]
        band = f < 0.005
        averaged = trapezoid(linear[:, band], x=f[band], axis=1) / (
            f[band][-1] - f[band][0]
        )
        np.testing.assert_allclose(
            np.quantile(averaged, [0.05, 0.5, 0.95]),
            summaries[key + "_low_band_temporal_quantiles"][:, 0],
            rtol=3e-13,
            atol=0,
        )
        checked[key] = dict(
            time_average_max_relative_error=max(errors),
            low_band_average_verified=True,
            posterior_sha256=s.audit.file_hash(
                Path(cell["path"]) / "posterior.nc"
            ),
        )
    truth_time = trapezoid(summaries["truth"], x=days, axis=0) / (
        days[-1] - days[0]
    )
    np.testing.assert_allclose(
        truth_time, summaries["truth_time_average"], rtol=2e-14
    )
    ledger = s.read(s.ROOT / "source_ledger.json")
    assert (
        s.audit.file_hash(s.PAPER / "main.tex")
        == ledger["baseline_main_tex_sha256"]
    )
    # Preserve the originally dirty shared implementation byte for byte.
    protocol = s.read(s.ROOT / "protocol.json")
    core = s.b.ROOT.parents[1] / "src/log_psplines/inference/power.py"
    assert s.audit.file_hash(core) == protocol["source_sha256"][str(core)]
    s.write(
        "numerical_validation.json",
        dict(
            paired_metric_rows_checked=paired,
            LS2_edge_contributions_verified=True,
            lisa_draw_wise_averages=checked,
            manuscript_baseline_unchanged=True,
            preexisting_shared_source_edit_unchanged=True,
        ),
    )
    print(
        f"Verified {paired} paired metric rows and draw-wise physical averages for {len(checked)} LISA cells."
    )


if __name__ == "__main__":
    main()
