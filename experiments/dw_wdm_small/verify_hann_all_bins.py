"""Independently check all-bin counts, exported scores and record SDs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import xarray as xr
from experiments.dw_wdm_small import hann_paper as s
from experiments.dw_wdm_small import hann_paper_all_bins as study

from log_psplines import PSDResult


def main() -> None:
    inventory = s.read(study.ROOT / "inventory.json")
    assert len(inventory) == 60 and all(
        c["status"] in ("completed", "reused") and c["diagnostics"]["pass"]
        for c in inventory
    )
    rows = [
        r
        for r in csv.DictReader((study.ROOT / "metrics.csv").open())
        if r["region"] == "all"
    ]
    summary = s.read(study.ROOT / "summary.json")
    previous = s.read(s.ROOT / "ls2_intermediate_20261015/summary.json")
    metrics = ("RMSE_logS", "W", "Inc90")
    for key, aggregate in summary["aggregate"].items():
        n, method = key.split("_")
        group = [
            r for r in rows if int(r["n"]) == int(n) and r["method"] == method
        ]
        assert len(group) == 5 and {int(r["record"]) for r in group} == set(
            range(5)
        )
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            np.testing.assert_allclose(
                aggregate[metric]["mean"], mean(values), atol=2e-15
            )
            np.testing.assert_allclose(
                aggregate[metric]["sample_sd"], stdev(values), atol=2e-15
            )
            if method == "WDM":
                for statistic in ("mean", "sample_sd"):
                    np.testing.assert_allclose(
                        aggregate[metric][statistic],
                        previous["aggregate"][key][metric][statistic],
                        rtol=2e-13,
                        atol=2e-15,
                    )
    for cell in inventory:
        if cell["method"] != "MP":
            continue
        data = study.observations(cell)
        np.testing.assert_array_equal(
            np.unique(np.rint(data.frequency * 32.5).astype(int)),
            np.arange(1, 33),
        )
        assert data.counts.sum() == cell["n"] - 64
        metadata = s.read(Path(cell["path"]) / "metrics.json")
        for key in ("power", "counts", "time", "frequency"):
            assert (
                s.b.digest(getattr(data, key))
                == metadata["observations_sha256"][key]
            )
    u, f = np.linspace(0.05, 0.95, 96), np.linspace(0.25, 4.75, 48)
    checked = {}
    for cell in inventory:
        if cell["record"] != 0:
            continue
        path = Path(cell["path"])
        post = (
            xr.load_dataset(path.with_suffix(".posterior.nc"))
            if cell["method"] == "WDM"
            else PSDResult.from_netcdf(path / "posterior.nc").posterior
        )
        model = s.endpoint_model(cell["n"])
        weights = post.weights.values.reshape(-1, *model.weights.shape)
        bt, bf = model.time.design_at(u), model.frequency.design_at(f / 5)
        projected = (weights.reshape(-1, weights.shape[-1]) @ bf.T).reshape(
            len(weights), weights.shape[1], len(f)
        )
        q = np.empty((3, len(u), len(f)))
        for i in range(0, len(u), 8):
            logs = np.tensordot(
                projected, bt[i : i + 8].T, axes=(1, 0)
            ).transpose(0, 2, 1)
            q[:, i : i + 8] = np.quantile(
                np.exp(logs), [0.05, 0.5, 0.95], axis=0
            )
        truth = 0.2 * s.LS2Data(
            n_samples=cell["n"], fs=10, seed=cell["seed"]
        ).get_true_psd(time_grid=u, freq_grid=f)
        direct = dict(
            RMSE_logS=float(np.sqrt(np.mean(np.log(q[1] / truth) ** 2))),
            W=float(np.log(q[2] / q[0]).mean()),
            Inc90=float(np.mean((q[0] <= truth) & (truth <= q[2]))),
        )
        exported = next(r for r in rows if r["target"] == cell["id"])
        for key, value in direct.items():
            np.testing.assert_allclose(
                value, float(exported[key]), rtol=2e-12, atol=2e-14
            )
        checked[cell["id"]] = direct
    record = dict(
        pass_all=True,
        records_checked=60,
        aggregate_groups=12,
        sample_sd_ddof=1,
        all_30_MP_observation_hashes_verified=True,
        WDM_results_unchanged=True,
        independent_spectral_reconstructions=checked,
    )
    (study.ROOT / "numerical_validation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(
        "Verified all 30 MP observation arrays, 12 mean/SD groups and 12 independent spectral reconstructions; WDM results unchanged."
    )


if __name__ == "__main__":
    main()
