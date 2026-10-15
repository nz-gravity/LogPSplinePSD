"""Frozen headline Hann replacements; reuse the existing data/model/samplers.

Run from the package root with .venv/bin/python -m experiments.dw_wdm_small.hann_paper.
All arrays and posteriors stay in this study directory; old outputs are read-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import jax
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import benchmark as b
from experiments.dw_wdm_small import focused_study as focused
from experiments.dw_wdm_small import gap_bias_audit as audit
from experiments.dw_wdm_small import taper_bias_extension as extension
from experiments.dw_wdm_small import taper_smoke as taper
from scipy import sparse

from log_psplines import (
    LogPSpline,
    PowerConfig,
    PowerData,
    PSDResult,
    SplineBasis,
    fit,
)
from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
)

ROOT = b.ROOT / "hann_paper"
PAPER = Path("/Users/avi/Documents/papers/moving_periodogram_pspline")
LS2 = PAPER / "studies/ls2_local"
LISA_MP = b.ROOT / "lisa_hann_one_case/noncentered_eta"
LISA_WDM = PAPER / "studies/lisa_gap_comparison"
MASKS = ("complete", "single")
METHODS = ("MP", "WDM")
LS2_TIME_KNOTS = dict(
    zip((2**k for k in range(10, 16)), (8, 11, 14, 18, 24, 32), strict=True)
)
SEEDS = list(range(6101, 6121))
jax.config.update("jax_enable_x64", True)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(name: str, value: object) -> None:
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    b.write_json(temporary, value)
    temporary.replace(path)


def csv_write(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def hann_observations(
    x: np.ndarray,
    observed: np.ndarray,
    dt: float,
    *,
    rungs: tuple[int, ...] | None = None,
) -> PowerData:
    """Gate original 32-window cycles, then select one-based frequency rungs."""
    n, m = len(x), 32
    raw = b.tang_moving_periodogram(np.zeros(n), m=m, thin=2)
    centers = np.rint(raw["u"] * n).astype(int) - 1
    prefix = np.r_[0, np.cumsum(~observed)]
    valid = prefix[centers + m + 1] == prefix[centers - m]
    keep = np.repeat(valid.reshape(-1, m).all(axis=1), m)
    selected = taper.RUNGS if rungs is None else np.asarray(rungs) - 1
    assert np.all((selected >= 0) & (selected < m))
    keep &= np.isin(np.arange(len(keep)) % m, selected)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.where(observed, x, 0.0), 65
    )
    h = np.hanning(65)
    coefficients = np.sum(
        windows[centers[keep] - m]
        * h
        * np.exp(-1j * raw["omega"][keep, None] * np.arange(65)),
        axis=1,
    )
    return PowerData(
        4 * dt * abs(coefficients) ** 2 / (h @ h),
        np.full(keep.sum(), 2.0),
        raw["omega"][keep] / (2 * np.pi * dt),
        centers[keep] / n,
        "one-sided PSD (data-unit squared/Hz)",
    )


def endpoint_model(n: int) -> LogPSpline:
    """Recover the supplied runner's actual uniform knots, not config defaults."""

    def basis(interior: int) -> SplineBasis:
        return SplineBasis.from_grid(np.array([0.0, 1.0]), interior)

    return LogPSpline(basis(6), time=basis(LS2_TIME_KNOTS[n]))


def save_basis(path: Path, model: LogPSpline) -> dict:
    arrays = {
        "time_knots": model.time.knots,
        "frequency_knots": model.frequency.knots,
        "time_penalty": np.asarray(model.time.penalty),
        "frequency_penalty": np.asarray(model.frequency.penalty),
    }
    np.savez_compressed(path, **arrays)
    return {k: b.digest(v) for k, v in arrays.items()}


def diagnostics(
    posterior: xr.Dataset, stats: xr.Dataset, model: LogPSpline, dataset: str
) -> dict:
    freq = (
        np.linspace(0.05, 0.95, 5)
        if dataset == "LS2"
        else np.linspace(0.05, 0.45, 5)
    )
    logs = np.einsum(
        "ti,cdij,fj->cdtf",
        model.time.design_at(np.linspace(0.05, 0.95, 9)),
        posterior.weights.values,
        model.frequency.design_at(freq),
        optimize=True,
    )
    tables = [
        azs.summary(
            posterior[
                [
                    k
                    for k in ("s", "sigma_time", "sigma_freq")
                    if k in posterior
                ]
            ],
            round_to="none",
        ),
        azs.summary(
            xr.Dataset({"log_psd": (("chain", "draw", "t", "f"), logs)}),
            round_to="none",
        ),
    ]
    energy = stats.energy.values
    d = {
        "rhat_max": float(max(t.r_hat.max() for t in tables)),
        "ess_bulk_min": float(min(t.ess_bulk.min() for t in tables)),
        "ess_tail_min": float(min(t.ess_tail.min() for t in tables)),
        "divergences": int(stats.diverging.sum()),
        "depth_hits": int((stats.n_steps >= 1023).sum()),
        "depth_fraction": float((stats.n_steps >= 1023).mean()),
        "ebfmi_min": float(
            (
                np.mean(np.diff(energy, axis=1) ** 2, axis=1)
                / np.var(energy, axis=1)
            ).min()
        ),
        "chains": posterior.sizes["chain"],
        "draws_per_chain": posterior.sizes["draw"],
    }
    minimum = 400 if dataset == "LS2" else 200
    d["pass"] = bool(
        d["rhat_max"] <= 1.01
        and min(d["ess_bulk_min"], d["ess_tail_min"]) >= minimum
        and d["divergences"] == 0
        and d["ebfmi_min"] > 0.3
        and (
            d["depth_fraction"] < 0.05
            if dataset == "LS2"
            else d["depth_hits"] == 0
        )
    )
    return d


def targets() -> list[dict]:
    cells = [
        dict(
            id=f"LS2_n{n}_r{r}_{method}",
            dataset="LS2",
            n=n,
            dt=0.1,
            record=r,
            seed=20261001 + r,
            mask="complete",
            method=method,
        )
        for n in (1024, 32768)
        for r in range(5)
        for method in METHODS
    ]
    cells += [
        dict(
            id=f"smooth_s{seed}_{mask}_{method}",
            dataset="smooth",
            n=8192,
            dt=1.0,
            record=seed,
            seed=seed,
            mask=mask,
            method=method,
        )
        for seed in SEEDS
        for mask in MASKS
        for method in METHODS
    ]
    cells += [
        dict(
            id=f"LISA_{mask}_{method}",
            dataset="LISA",
            n=15728640,
            dt=2.0,
            record="A_days7_35",
            seed=None,
            mask=mask,
            method=method,
        )
        for mask in ("single", "complete")
        for method in METHODS
    ]
    return cells


def candidates(cell: dict) -> list[Path]:
    if cell["dataset"] == "LS2" and cell["method"] == "WDM":
        return [LS2 / f"wdm_n{cell['n']}_r{cell['record']}"]
    if cell["dataset"] == "smooth":
        if cell["method"] == "WDM":
            selected = focused.selected(cell["seed"], "wdm", cell["mask"])
            return [selected]
        if cell["seed"] <= 6110:
            if cell["mask"] == "single":
                primary = (
                    b.ROOT
                    / "taper_study/fits/hann_sparse"
                    / f"seed{cell['seed']}_dw"
                )
                repair = (
                    b.ROOT
                    / "taper_study/repairs/hann_sparse"
                    / f"seed{cell['seed']}_dw"
                )
                return [primary, repair]
            return [
                extension.fit_directory(
                    "broadband", cell["mask"], "hann_sparse", cell["seed"], i
                )
                for i in range(3)
            ]
    if cell["dataset"] == "LISA" and cell["mask"] == "single":
        return [LISA_MP if cell["method"] == "MP" else LISA_WDM]
    return [ROOT / "fits" / cell["id"] / f"attempt{i}" for i in range(2)]


def config(cell: dict, attempt: int) -> PowerConfig:
    endpoint = cell["dataset"] == "LS2"
    # Historical endpoints explicitly used this effort/acceptance. One extension only.
    return PowerConfig(
        n_warmup=(500 if cell["n"] == 1024 else 1000)
        if endpoint and attempt == 0
        else (1000 if attempt else 600),
        n_samples=(1000 if cell["n"] == 1024 else 2000)
        if endpoint and attempt == 0
        else (4000 if endpoint else 1500)
        if attempt
        else 1000,
        num_chains=2,
        target_accept_prob=(0.99 if attempt else 0.9 if endpoint else 0.95),
        max_tree_depth=10,
        dense_mass=not endpoint,
        centered=False,
        seed=17000 + cell["record"] if endpoint else cell["seed"] + 10000,
        n_interior_knots_time=LS2_TIME_KNOTS[cell["n"]] if endpoint else 8,
        n_interior_knots_freq=6 if endpoint else 2,
        spectrum_draws=8,
        spectrum_chunk_size=8,
        progress_bar=False,
    )


def freeze() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "protocol.json").exists():
        return
    (ROOT / "inputs").mkdir(exist_ok=True)
    recovered, contradictions = {}, []
    for n in (1024, 32768):
        for r in range(5):
            stem = LS2 / f"moving_n{n}_r{r}"
            old = read(stem.with_suffix(".json"))
            assert old["data_seed"] == 20261001 + r
            x = LS2Data(n_samples=n, fs=10, seed=old["data_seed"]).data[:, 0]
            saved = np.load(stem.with_suffix(".npz"))
            native = scattered_moving_periodogram(x, dt=0.1, m=32, thin=2)
            for key in ("power", "counts", "time", "frequency"):
                np.testing.assert_array_equal(
                    getattr(native, key), saved[f"observed_{key}"]
                )
            wstem = LS2 / f"wdm_n{n}_r{r}"
            from log_psplines import TimeSeries
            from log_psplines.preprocessing.wdm import wdm_periodogram

            wd = wdm_periodogram(
                TimeSeries(data=x, t=np.arange(n) * 0.1), nt=n // 32
            )
            wz = np.load(wstem.with_suffix(".npz"))
            for key in ("power", "counts", "time", "frequency"):
                np.testing.assert_array_equal(
                    getattr(wd, key), wz[f"observed_{key}"]
                )
            np.savez_compressed(ROOT / "inputs" / f"LS2_n{n}_r{r}.npz", x=x)
            model = endpoint_model(n)
            hashes = save_basis(ROOT / "inputs" / f"LS2_n{n}_basis.npz", model)
            assert (
                saved["weights"].shape[-2:]
                == model.weights.shape
                == wz["weights"].shape[-2:]
            )
            recovered[f"LS2_n{n}_r{r}"] = dict(
                raw_sha256=b.digest(x),
                recovery="deterministic archive generator replay; bitwise equality of every saved MP and WDM power/count/time/frequency array",
                archive_mp_sha256=audit.file_hash(stem.with_suffix(".npz")),
                archive_wdm_sha256=audit.file_hash(wstem.with_suffix(".npz")),
                model_hashes=hashes,
            )
    for seed in SEEDS:
        x = b.simulate(seed)
        paths = [focused.selected(seed, "wdm", mask) for mask in MASKS]
        for p in paths:
            assert read(p / "metrics.json")["raw_sha256"] == b.digest(x)
        np.savez_compressed(ROOT / "inputs" / f"smooth_s{seed}.npz", x=x)
        recovered[f"smooth_s{seed}"] = dict(
            raw_sha256=b.digest(x),
            recovery="exact archived raw SHA256 match after generator replay",
            model_hashes=save_basis(
                ROOT / "inputs" / "smooth_basis.npz", b.master_model()
            ),
        )
    # Config knot counts are stale defaults in the historical LS2 custom-model fits.
    contradictions.append(
        "LS2 saved config says 8/10 interior knots even for custom models. Supplied runner plus archived weight dimensions recover actual time 8/32 and frequency 6; saved config is not the fitted basis."
    )
    source_paths = [
        Path(b.__file__),
        Path(taper.__file__),
        Path(focused.__file__),
        Path(extension.__file__),
        PAPER / "scripts/run_ls2_local.py",
        PAPER / "scripts/lisa_gap_comparison.py",
        b.ROOT.parents[1] / "src/log_psplines/inference/power.py",
    ]
    protocol = dict(
        date="2026-10-15",
        scope="endpoint-only Hann replacement; 50 toy MP plus 2 complete LISA before reuse",
        targets=targets(),
        recovered_inputs=recovered,
        contradictions_resolved=contradictions,
        observation=dict(
            m=32,
            L=65,
            thin=2,
            rungs=list(range(3, 31, 3)),
            energy=24,
            gap_support="all original 32 windows checked before sparse selection; full cycle rejected; zero-weight endpoints still count",
            coordinates="paired center/N and j/(65*dt); original cycle phase and incomplete-tail convention",
            adapter="power=4*dt*abs(weighted Fourier sum)^2/sum(h^2); count=2",
        ),
        smooth=dict(
            n=8192,
            dt=1,
            gap=[3686, 4505],
            basis=[12, 6],
            grid=read(focused.ROOT / "protocol.json"),
            wdm="nt=256 nf=32 whole-column 99.99% retained-energy guard and rho correction",
        ),
        ls2=dict(
            lengths=[1024, 32768],
            repeats=list(range(5)),
            dt=0.1,
            grid_time=np.linspace(0.05, 0.95, 96).tolist(),
            grid_frequency=np.linspace(0.25, 4.75, 48).tolist(),
            historical_mp_reconstruction="evaluate archived basis at physical t+1/N exactly once; new MP uses physical t directly",
            frequency_mapping="f/5 Hz",
            prior="unchanged tensor HalfNormal(10) smoothing scales, trace-normalized penalties, null_precision=1e-4 ridge=1e-6",
        ),
        primary_and_extension={
            c["id"]: [asdict(config(c, i)) for i in range(2)]
            for c in targets()
            if c["dataset"] != "LISA" and c["method"] == "MP"
        },
        lisa=dict(
            interval_days=[7, 35],
            sigma_g_prior=10,
            sigma_eta_prior=0.5,
            parameters=114,
            warmup=1200,
            draws=1600,
            chains=2,
            depth=12,
            target_accept={"MP": 0.99, "WDM": 0.995},
            extension=dict(warmup=1800, draws=2400),
            basis="16 frequency, 7 raw time reduced to 6; center over 2046 full-interval sites before masking",
            MP="L=15361; rungs6,9,...3072; global origin preserved",
            WDM="nf=7680,nt=2048; archived one-hour ramps and 1e-4 conservative energy bound",
        ),
        diagnostics="Rhat<=1.01; bulk/tail ESS>=200 smooth, >=400 LS2/LISA; no divergences; E-BFMI>0.3; zero depth hits smooth/LISA, <5% LS2; fixed spectral functionals",
        summary="linear PSD posterior median, central 90%; natural-log errors; whole-record paired bootstrap 5000 resamples seed20261015; no outcome-based selection",
        tolerances=dict(
            direct_power_atol=2e-13,
            direct_power_rtol=1e-12,
            basis_atol=1e-12,
            rationale="float64 direct Fourier and existing taper regression tolerances; covariance diagnostics are descriptive, without a new universal pass cutoff",
        ),
        source_sha256={str(p): audit.file_hash(p) for p in source_paths},
        package_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        versions={
            p: version(p)
            for p in ("numpy", "jax", "numpyro", "scipy", "wdm_transform")
        },
        python=platform.python_version(),
    )
    write("protocol.json", protocol)
    print(
        "Frozen 30 recovered raw records; archived powers/checksums verified.",
        flush=True,
    )


def load_input(cell: dict) -> np.ndarray:
    stem = (
        f"LS2_n{cell['n']}_r{cell['record']}"
        if cell["dataset"] == "LS2"
        else f"smooth_s{cell['seed']}"
    )
    return np.load(ROOT / "inputs" / f"{stem}.npz")["x"]


def data_model(cell: dict) -> tuple[PowerData, LogPSpline]:
    x = load_input(cell)
    observed = (
        np.ones(len(x), bool)
        if cell["mask"] == "complete"
        else focused.masks()["single"]
    )
    data = hann_observations(x, observed, cell["dt"])
    if cell["dataset"] == "LS2":
        data = PowerData(
            data.power, data.counts, data.frequency / 5, data.time, data.units
        )
        return data, endpoint_model(cell["n"])
    return data, b.master_model()


def preflight() -> None:
    from scipy.signal import periodogram

    checks = {}
    h = np.hanning(65)
    np.testing.assert_array_equal(h, h[::-1])
    assert h[0] == h[-1] == 0 and h @ h == 24
    for dataset, n, dt in (
        ("LS2", 1024, 0.1),
        ("LS2", 32768, 0.1),
        ("smooth", 8192, 1.0),
    ):
        c = next(
            c
            for c in targets()
            if c["dataset"] == dataset and c["n"] == n and c["method"] == "MP"
        )
        x = load_input(c)
        observed = np.ones(n, bool)
        data = hann_observations(x, observed, dt)
        centers = np.rint(data.time * n).astype(int)
        windows = np.lib.stride_tricks.sliding_window_view(x, 65)[centers - 32]
        f, p = periodogram(
            windows,
            fs=1 / dt,
            window=h,
            detrend=False,
            scaling="density",
            axis=-1,
        )
        rungs = np.rint(data.frequency * 65 * dt).astype(int)
        np.testing.assert_allclose(
            data.power / 2, p[np.arange(len(p)), rungs], atol=2e-13, rtol=1e-12
        )
        assert (
            set(rungs) == set(range(3, 31, 3))
            and centers.min() >= 32
            and centers.max() + 32 < n
        )
        if dataset == "smooth":
            for mask in MASKS:
                obs = focused.masks()[mask]
                g = hann_observations(x, obs, dt)
                reference = taper.prepare(x, "hann", obs)
                for k in ("time", "frequency", "counts", "power"):
                    np.testing.assert_allclose(
                        getattr(g, k),
                        getattr(reference, k),
                        atol=2e-13,
                        rtol=1e-12,
                    )
                lookup = {t: i for i, t in enumerate(data.time)}
                np.testing.assert_array_equal(
                    g.power, data.power[[lookup[t] for t in g.time]]
                )
                for hidden in (np.nan, 0.0, 1e50):
                    z = np.where(obs, x, hidden)
                    np.testing.assert_array_equal(
                        g.power, hann_observations(z, obs, dt).power
                    )
        # Exact covariance and nonproper complex audit, including cross-cycle pairs.
        row_values = (
            h[None, :]
            * np.exp(-2j * np.pi * rungs[:, None] * np.arange(65) / 65)
            / np.sqrt(24)
        )
        cols = (centers[:, None] + np.arange(-32, 33)).ravel()
        rows = sparse.csr_matrix(
            (
                row_values.ravel(),
                (np.repeat(np.arange(len(centers)), 65), cols),
            ),
            shape=(len(centers), n),
        )
        if dataset == "LS2":
            coef = LS2Data(n_samples=n, fs=1 / dt, seed=c["seed"]).coefficient(
                np.arange(n) / n
            )
            diagonal, lower = 1 + coef**2, coef
            wb = np.lib.stride_tricks.sliding_window_view(coef, 65)[
                centers - 32
            ]
            expected = (
                2
                * dt
                * (
                    (1 + wb**2) @ (h**2)
                    + 2
                    * np.cos(2 * np.pi * data.frequency * dt)
                    * (wb[:, 1:] @ (h[1:] * h[:-1]))
                )
                / 24
            )
            truth = (
                2
                * dt
                * (
                    1
                    + coef[centers] ** 2
                    + 2
                    * coef[centers]
                    * np.cos(2 * np.pi * data.frequency * dt)
                )
            )
        else:
            diagonal, lower = b.variance(np.arange(n) / n), np.zeros(n)
            expected = (
                2
                * dt
                * np.lib.stride_tricks.sliding_window_view(diagonal, 65)[
                    centers - 32
                ]
                @ (h**2)
                / 24
            )
            truth = 2 * dt * diagonal[centers]
        dependence = audit.gaussian_dependence(rows, diagonal, lower)
        logerr = np.log(expected / truth)
        eta = np.log(truth)
        np.testing.assert_allclose(
            float(power_whittle_log_likelihood(data.power, data.counts, eta)),
            -np.sum(eta + data.power / (2 * truth)),
            rtol=1e-12,
        )
        checks[f"{dataset}_{n}"] = dict(
            sites=len(centers),
            direct_max_error=float(
                np.max(abs(data.power / 2 - p[np.arange(len(p)), rungs]))
            ),
            rms_log_expected_discrepancy=float(np.sqrt(np.mean(logerr**2))),
            maximum_fractional_expected_discrepancy=float(
                np.max(abs(np.expm1(logerr)))
            ),
            dependence=dependence,
            white_expectation=2 * dt,
            selected_range_hz=[3 / (65 * dt), 30 / (65 * dt)],
            edge_prediction="preserve full archived scoring band; expose edge contribution",
        )
    checks["smooth_masks_dependence"] = {
        mask: __import__(
            "experiments.dw_wdm_small.taper_study",
            fromlist=["all_local_dependence"],
        ).all_local_dependence(h, focused.masks()[mask])
        for mask in MASKS
    }
    checks["status"] = "pass"
    checks["interpretation"] = (
        "Arithmetic, units and mask invariance pass. Expected means and small residual correlations do not certify calibration; final frozen model assessed on preselected records."
    )
    write("preflight.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def inventory() -> list[dict]:
    protocol = read(ROOT / "protocol.json")
    previous = (
        {r["id"]: r for r in read(ROOT / "inventory.json")}
        if (ROOT / "inventory.json").exists()
        else {}
    )
    validator = audit.file_hash(__file__)
    rows = []
    for cell in targets():
        fingerprint = [validator, 2]
        for path in candidates(cell):
            if path is None:
                fingerprint.append(None)
                continue
            legacy = cell["dataset"] == "LS2" and cell["method"] == "WDM"
            files = (
                [
                    path.with_suffix(s)
                    for s in (".json", ".posterior.nc", ".sample_stats.nc")
                ]
                if legacy
                else [
                    path / name
                    for name in (
                        "metrics.json",
                        "fit.json",
                        "posterior.nc",
                        "sample_stats.nc",
                        "failure.json",
                    )
                ]
            )
            fingerprint.append(
                [
                    [str(p), p.stat().st_size, p.stat().st_mtime_ns]
                    for p in files
                    if p.exists()
                ]
            )
        if (
            previous.get(cell["id"], {}).get("validation_fingerprint")
            == fingerprint
        ):
            rows.append(previous[cell["id"]])
            continue
        row = {
            **cell,
            "status": "not started",
            "reuse_reason": "new frozen Hann target",
            "attempts": [],
        }
        if cell["dataset"] != "LISA":
            key = (
                f"LS2_n{cell['n']}_r{cell['record']}"
                if cell["dataset"] == "LS2"
                else f"smooth_s{cell['seed']}"
            )
            row.update(protocol["recovered_inputs"][key])
            data, master = data_model(cell)
            if cell["method"] == "WDM":
                if cell["dataset"] == "LS2":
                    z = np.load(
                        (
                            LS2 / f"wdm_n{cell['n']}_r{cell['record']}"
                        ).with_suffix(".npz")
                    )
                    data = PowerData(
                        z["observed_power"] * (2 * cell["dt"] / cell["n"]),
                        z["observed_counts"],
                        z["observed_frequency"] / 5,
                        z["observed_time"],
                    )
                else:
                    data = extension.observations(
                        load_input(cell), "wdm", cell["mask"], "broadband"
                    )
            observations_path = ROOT / "observations" / f"{cell['id']}.npz"
            observations_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                observations_path,
                **{
                    k: np.asarray(getattr(data, k))
                    for k in ("time", "frequency", "power", "counts")
                },
            )
            row.update(
                observations_path=str(observations_path),
                observations_sha256={
                    k: b.digest(getattr(data, k))
                    for k in ("time", "frequency", "power", "counts")
                },
                likelihood_observations=int(np.count_nonzero(data.counts)),
                real_components=float(data.counts.sum()),
                absolute_epoch_range_seconds=[0, (cell["n"] - 1) * cell["dt"]],
                basis_time_knots=master.time.knots.tolist(),
                basis_frequency_knots=master.frequency.knots.tolist(),
                coordinate_map="t=center/N; f/5"
                if cell["dataset"] == "LS2"
                else "t=center/N; frequency Hz",
                mask_intervals=[]
                if cell["mask"] == "complete"
                else [[3686, 4505]],
            )
        for path in candidates(cell):
            if path is None:
                row.update(
                    status="blocked",
                    reuse_reason="missing WDM comparator; extra inference outside frozen scope",
                )
                continue
            legacy = cell["dataset"] == "LS2" and cell["method"] == "WDM"
            meta = (
                path.with_suffix(".json")
                if legacy
                else path
                / ("fit.json" if cell["dataset"] == "LISA" else "metrics.json")
            )
            if not meta.exists():
                if (path / "failure.json").exists():
                    row["attempts"].append(
                        dict(path=str(path), error=read(path / "failure.json"))
                    )
                    if not row.get("path"):
                        row.update(
                            status="failed",
                            reuse_reason="execution failure; see attempt receipt",
                        )
                continue
            try:
                m = read(meta)
                if legacy:
                    posterior = xr.load_dataset(
                        path.with_suffix(".posterior.nc")
                    )
                    stats = xr.load_dataset(
                        path.with_suffix(".sample_stats.nc")
                    )
                    d = diagnostics(posterior, stats, master, "LS2")
                    posterior_path = path.with_suffix(".posterior.nc")
                    for key in (
                        "roughness_scale",
                        "null_precision",
                        "ridge_eps",
                        "centered",
                        "degree_time",
                        "degree_freq",
                        "penalty_order_time",
                        "penalty_order_freq",
                    ):
                        assert (
                            m["config"][key] == asdict(config(cell, 0))[key]
                        ), key
                elif cell["dataset"] == "LISA":
                    posterior_path = path / "posterior.nc"
                    posterior = xr.load_dataset(posterior_path)
                    d = m["diagnostics"]
                    assert (
                        posterior.sizes["chain"] == 2
                        and posterior.sizes["draw"] >= 1600
                    )
                    prep_path = path / "preparation.json"
                    if not prep_path.exists():
                        prep_path = path.parent / "preparation.json"
                    prep = read(prep_path)
                    sites = np.load(path / "prepared.npz")
                    anchor_epochs = np.load(LISA_MP / "prepared.npz")[
                        "anchor_epochs"
                    ]
                    row.update(
                        raw_sha256=prep["tdi_total_sha256"],
                        preparation=prep,
                        physical_basis_sha256=audit.file_hash(
                            LISA_MP / "model_basis.npz"
                        ),
                        absolute_epoch_range_tcb=[
                            float(anchor_epochs[0]),
                            float(anchor_epochs[-1]),
                        ],
                        observations_path=str(path / "prepared.npz"),
                        likelihood_observations=int(sites["power"].size),
                        real_components=int(sites["power"].size)
                        * (2 if cell["method"] == "MP" else 1),
                        coordinates="saved exact paired global sample time (time*N*dt seconds) and physical frequency in prepared.npz",
                    )
                else:
                    posterior_path = path / "posterior.nc"
                    result = PSDResult.from_netcdf(posterior_path)
                    np.testing.assert_array_equal(
                        result.model_data.knots_time, master.time.knots
                    )
                    np.testing.assert_array_equal(
                        result.model_data.knots_frequency,
                        master.frequency.knots,
                    )
                    assert (
                        m["raw_sha256"] == row["raw_sha256"]
                        and m["model_hashes"] == row["model_hashes"]
                    )
                    for k in (
                        "roughness_scale",
                        "null_precision",
                        "ridge_eps",
                        "centered",
                        "degree_time",
                        "degree_freq",
                        "penalty_order_time",
                        "penalty_order_freq",
                    ):
                        assert m["config"][k] == asdict(config(cell, 0))[k], k
                    if cell["mask"] == "single":
                        assert m["mask_sha256"] == b.digest(
                            focused.masks()["single"]
                        )
                    if "observations_sha256" in m and cell["method"] == "MP":
                        for k in ("power", "counts", "time", "frequency"):
                            # Last-bit multiplication-order difference is harmless; exact saved
                            # coordinates and power parity are independently checked below.
                            if k != "power":
                                assert m["observations_sha256"][k] == b.digest(
                                    getattr(data, k)
                                ), k
                    for key in ("power", "counts", "frequency", "time"):
                        np.testing.assert_allclose(
                            result.observed_data[key].values,
                            getattr(data, key),
                            rtol=1e-12,
                            atol=2e-13,
                        )
                    d = diagnostics(
                        result.posterior,
                        result.sample_stats,
                        master,
                        cell["dataset"],
                    )
                row["attempts"].append(
                    dict(
                        path=str(path),
                        diagnostics=d,
                        inference_seconds=m.get(
                            "inference_seconds", m.get("runtime_s")
                        ),
                        config=m.get("config", m),
                    )
                )
                if d["pass"] and not row.get("path"):
                    row.update(
                        status="completed"
                        if ROOT in path.parents
                        else "reused",
                        path=str(path),
                        posterior_sha256=audit.file_hash(posterior_path),
                        diagnostics=d,
                        inference_config=m.get("config", m),
                        reuse_reason="physical basis/prior, raw identity and retained-draw diagnostics verified",
                        historical_inference_seconds=m.get(
                            "inference_seconds", m.get("runtime_s")
                        ),
                    )
                elif not d["pass"] and not row.get("path"):
                    row.update(
                        status="failed",
                        reuse_reason="saved attempt fails frozen diagnostic screen",
                    )
            except Exception as exc:
                row.update(
                    status="blocked",
                    reuse_reason=f"artifact mismatch: {exc!r}",
                )
        row["validation_fingerprint"] = fingerprint
        rows.append(row)
    write("inventory.json", rows)
    csv_write("inventory.csv", rows)
    print(
        json.dumps(
            {
                s: sum(r["status"] == s for r in rows)
                for s in set(r["status"] for r in rows)
            }
        ),
        flush=True,
    )
    return rows


def run(smoke: bool = False) -> None:
    assert read(ROOT / "preflight.json")["status"] == "pass"
    inv = inventory()
    ordered = sorted(
        [r for r in inv if r["method"] == "MP" and r["dataset"] != "LISA"],
        key=lambda c: (
            0
            if c["dataset"] == "LS2" and c["record"] == 0
            else 1
            if c["dataset"] == "smooth" and c["seed"] == 6101
            else 2,
            c["id"],
        ),
    )
    for cell in ordered:
        if smoke and not (
            (cell["dataset"] == "LS2" and cell["record"] == 0)
            or (cell["dataset"] == "smooth" and cell["seed"] == 6101)
        ):
            continue
        if cell["status"] in ("reused", "completed", "blocked"):
            continue
        data, model = data_model(cell)
        for attempt in range(2):
            path = ROOT / "fits" / cell["id"] / f"attempt{attempt}"
            if (path / "metrics.json").exists():
                if read(path / "metrics.json")["diagnostics"]["pass"]:
                    break
                continue
            path.mkdir(parents=True, exist_ok=True)
            cfg = config(cell, attempt)
            write(
                "execution.json",
                dict(
                    status="running",
                    pid=os.getpid(),
                    target=cell["id"],
                    attempt=attempt,
                ),
            )
            print(
                json.dumps(
                    dict(
                        starting=cell["id"],
                        attempt=attempt,
                        config=asdict(cfg),
                    )
                ),
                flush=True,
            )
            started = perf_counter()
            try:
                result = fit(data, cfg, model=model)
                d = diagnostics(
                    result.posterior,
                    result.sample_stats,
                    model,
                    cell["dataset"],
                )
                result.to_netcdf(path / "posterior.nc")
                # Require a real reload, rather than trusting the writer's return.
                loaded = PSDResult.from_netcdf(path / "posterior.nc")
                np.testing.assert_array_equal(
                    result.posterior.weights, loaded.posterior.weights
                )
                row = dict(
                    raw_sha256=cell["raw_sha256"],
                    model_hashes=cell["model_hashes"],
                    mask_sha256=b.digest(
                        np.ones(cell["n"], bool)
                        if cell["mask"] == "complete"
                        else focused.masks()["single"]
                    ),
                    config=asdict(cfg),
                    diagnostics=d,
                    attempt=attempt,
                    observations_sha256={
                        k: b.digest(getattr(data, k))
                        for k in ("power", "counts", "time", "frequency")
                    },
                    likelihood_observations=data.power.size,
                    inference_seconds=float(
                        result.metadata["inference_seconds"]
                    ),
                    total_seconds=perf_counter() - started,
                    runner_sha256=audit.file_hash(__file__),
                )
                b.write_json(path / "metrics.json", row)
                print(
                    json.dumps(
                        dict(
                            completed=cell["id"],
                            attempt=attempt,
                            seconds=row["total_seconds"],
                            diagnostics=d,
                        )
                    ),
                    flush=True,
                )
                del result, loaded
                jax.clear_caches()
                gc.collect()
                if d["pass"]:
                    break
            except Exception as exc:
                b.write_json(
                    path / "failure.json",
                    dict(error=repr(exc), seconds=perf_counter() - started),
                )
                print(
                    json.dumps(dict(failed=cell["id"], error=repr(exc))),
                    flush=True,
                )
                break  # implementation failures do not trigger statistical retuning
        inventory()
    write(
        "execution.json",
        dict(
            status="smoke complete" if smoke else "bounded toy batch finished",
            pid=os.getpid(),
        ),
    )


def quantiles(
    posterior: xr.Dataset, model: LogPSpline, u: np.ndarray, f: np.ndarray
) -> np.ndarray:
    weights = posterior.weights.values.reshape(-1, *model.weights.shape)
    bt, bf = model.time.design_at(u), model.frequency.design_at(f)
    out = np.empty((3, len(u), len(f)))
    for i in range(0, len(u), 8):
        linear = np.exp(
            np.einsum(
                "ti,dij,fj->dtf", bt[i : i + 8], weights, bf, optimize=True
            )
        )
        out[:, i : i + 8] = np.quantile(linear, [0.05, 0.5, 0.95], axis=0)
    assert np.isfinite(out).all() and (out > 0).all()
    return out


def scores(q: np.ndarray, truth: np.ndarray, w: np.ndarray) -> dict:
    w = w / w.sum()
    err = np.log(q[1] / truth)
    return dict(
        RMSE_logS=float(np.sqrt(np.sum(w * err**2))),
        B=float(np.sum(w * err)),
        fractional_error=float(np.sum(w * (q[1] / truth - 1))),
        W=float(np.sum(w * np.log(q[2] / q[0]))),
        Inc90=float(np.sum(w * ((q[0] <= truth) & (truth <= q[2])))),
    )


def summarize() -> None:

    inv = inventory()
    rows = []
    surfaces = {}
    frequency_rows = []
    for cell in inv:
        if cell["dataset"] == "LISA" or not cell.get("path"):
            continue
        path = Path(cell["path"])
        if cell["dataset"] == "LS2":
            u = np.linspace(0.05, 0.95, 96)
            f = np.linspace(0.25, 4.75, 48)
            model = endpoint_model(cell["n"])
            post = (
                xr.load_dataset(path.with_suffix(".posterior.nc"))
                if cell["method"] == "WDM"
                else PSDResult.from_netcdf(path / "posterior.nc").posterior
            )
            q = quantiles(post, model, u, f / 5)
            true = (
                2
                * 0.1
                * LS2Data(
                    n_samples=cell["n"], fs=10, seed=cell["seed"]
                ).get_true_psd(time_grid=u, freq_grid=f)
            )
            weights = np.ones(true.shape)
            regions = {
                "all": weights,
                "edge_prediction": weights * ((f < 3 / 6.5) | (f > 30 / 6.5)),
                "within_center_frequency_range": weights
                * ((f >= 3 / 6.5) & (f <= 30 / 6.5)),
            }
            for j, fj in enumerate(f):
                frequency_rows.append(
                    dict(
                        target=cell["id"],
                        frequency_hz=fj,
                        signed_log_error=float(
                            np.log(q[1, :, j] / true[:, j]).mean()
                        ),
                    )
                )
        else:
            u, f, _ = extension.grid()
            model = b.master_model()
            post = PSDResult.from_netcdf(path / "posterior.nc").posterior
            q = quantiles(post, model, u, f)
            true = np.broadcast_to(
                2 * b.variance(u)[:, None], (len(u), len(f))
            )
            weights = np.outer(audit.axis_weights(u), audit.axis_weights(f))
            regions = {
                k: weights * v[:, None]
                for k, v in audit.regions(
                    u, focused.masks()["single"], 327 / 8192
                ).items()
            }
            regions["all"] = weights
            edge = (f < 3 / 65) | (f > 30 / 65)
            if edge.any():
                regions["edge_prediction"] = weights * edge
            for region, rm in audit.regions(
                u, focused.masks()["single"], 327 / 8192
            ).items():
                for probe in (0.1, 0.25, 0.4):
                    fw = np.zeros(len(f))
                    fw[np.argmin(abs(f - probe))] = 1
                    regions[f"{region}_f{probe:.2f}"] = np.outer(
                        audit.axis_weights(u) * rm, fw
                    )
        surfaces[cell["id"]] = (q, true, u, f)
        for region, w in regions.items():
            rows.append(
                dict(
                    target=cell["id"],
                    dataset=cell["dataset"],
                    n=cell["n"],
                    record=cell["record"],
                    method=cell["method"],
                    mask=cell["mask"],
                    region=region,
                    posterior_sha256=cell["posterior_sha256"],
                    **scores(q, true, w),
                )
            )
    for r in rows:
        if r["dataset"] != "smooth" or r["mask"] != "single":
            continue
        complete = next(
            (
                c
                for c in rows
                if c["dataset"] == "smooth"
                and c["record"] == r["record"]
                and c["method"] == r["method"]
                and c["mask"] == "complete"
                and c["region"] == r["region"]
            ),
            None,
        )
        if complete:
            r["Delta_B"] = r["B"] - complete["B"]
            r["complete_B"] = complete["B"]
    aggregates = {}
    for dataset, n, method, mask, region in sorted(
        set(
            (r["dataset"], r["n"], r["method"], r["mask"], r["region"])
            for r in rows
        )
    ):
        selected = [
            r
            for r in rows
            if (r["dataset"], r["n"], r["method"], r["mask"], r["region"])
            == (dataset, n, method, mask, region)
        ]
        aggregates[f"{dataset}_{n}_{method}_{mask}_{region}"] = {
            k: focused.ci([r[k] for r in selected])
            for k in (
                "RMSE_logS",
                "B",
                "fractional_error",
                "W",
                "Inc90",
                "Delta_B",
            )
            if all(k in r for r in selected)
        }
    paired = {}
    for n in (1024, 32768):
        for metric in ("RMSE_logS", "B", "W", "Inc90"):
            vectors = []
            for record in range(5):
                pair = [
                    next(
                        (
                            r
                            for r in rows
                            if r["dataset"] == "LS2"
                            and r["n"] == n
                            and r["record"] == record
                            and r["method"] == m
                            and r["region"] == "all"
                        ),
                        None,
                    )
                    for m in METHODS
                ]
                if all(pair):
                    vectors.append(pair[0][metric] - pair[1][metric])
            if vectors:
                paired[f"LS2_{n}_{metric}_MP_minus_WDM"] = focused.ci(vectors)
    for region in sorted(
        set(r["region"] for r in rows if r["dataset"] == "smooth")
    ):
        for metric in ("B", "Delta_B", "RMSE_logS", "W", "Inc90"):
            vectors = []
            for record in SEEDS:
                pair = [
                    next(
                        (
                            r
                            for r in rows
                            if r["dataset"] == "smooth"
                            and r["record"] == record
                            and r["method"] == m
                            and r["mask"] == "single"
                            and r["region"] == region
                            and metric in r
                        ),
                        None,
                    )
                    for m in METHODS
                ]
                if all(pair):
                    vectors.append(pair[0][metric] - pair[1][metric])
            if vectors:
                paired[f"smooth_{region}_{metric}_MP_minus_WDM"] = focused.ci(
                    vectors
                )
    csv_write("metrics.csv", rows)
    csv_write("frequency_signed_errors.csv", frequency_rows)
    summary = dict(
        status_counts={
            s: sum(c["status"] == s for c in inv)
            for s in set(c["status"] for c in inv)
        },
        aggregates=aggregates,
        paired_method_contrasts=paired,
        weighting="LS2 archived uniform grid; smooth physical trapezoidal quadrature; bootstrap whole records with common seed and pairing",
        caveats=[
            "Combined 20 smooth records contain the original 10 development records.",
            "Five LS2 records do not certify calibration; longer N stretches modulation and increases time-basis size.",
            "Missing-interval scores describe interpolation; interval containing zero does not establish negligible bias.",
        ],
    )
    write("summary.json", summary)
    (ROOT / "figures").mkdir(exist_ok=True)
    from experiments.dw_wdm_small.hann_paper_figures import plot_smooth

    # Endpoint results are tabulated; retain only the fixed illustrative gap slice.
    plot_smooth(surfaces)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"$N$ & Method & RMSE$_{\ln S}$ & $W$ & Inc90\\",
        r"\hline",
    ]
    for n in (1024, 32768):
        for method in METHODS:
            a = aggregates.get(f"LS2_{n}_{method}_complete_all")
            if a:
                lines.append(
                    f"{n:,} & {'Hann MP' if method == 'MP' else method} & {a['RMSE_logS']['mean']:.4f} & {a['W']['mean']:.4f} & {a['Inc90']['mean']:.3f} \\\\"
                )
    lines.append(r"\end{tabular}")
    (ROOT / "ls2_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("inventory", "preflight", "smoke", "run", "summarize"),
    )
    args = parser.parse_args()
    if args.action == "inventory":
        freeze()
        inventory()
    elif args.action == "preflight":
        preflight()
    elif args.action in ("smoke", "run"):
        run(args.action == "smoke")
    else:
        summarize()


if __name__ == "__main__":
    main()
