"""Reconstruct and plot the bounded LISA fit; raw results stay local."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import lisa_hann_one_case as case
from scipy.interpolate import interp1d

from log_psplines import ANOVALogPSpline, SplineBasis


def main() -> None:
    out = case.OUT
    fit = json.loads((out / "fit.json").read_text())
    assert fit["diagnostics"]["pass"], (
        "Do not present a flagged posterior as validated."
    )
    posterior = xr.load_dataset(out / "posterior.nc")
    saved, basis = (
        np.load(out / "prepared.npz"),
        np.load(out / "model_basis.npz"),
    )
    interior_f = basis["frequency_knots"][4:-4]
    bf = SplineBasis.from_grid(
        np.linspace(case.FMIN, case.FMAX, 100), interior_knots=interior_f
    )
    reference_time = np.linspace(
        case.START_DAY * 86400 / (case.N * case.DT),
        case.STOP_DAY * 86400 / (case.N * case.DT),
        2046,
    )
    bt = SplineBasis.from_grid(reference_time, 3)
    spline = ANOVALogPSpline(bf, bt, sigma_eta_prior=0.5)
    np.testing.assert_allclose(
        spline.time_transform, basis["time_transform"], atol=1e-13
    )
    days = np.linspace(case.START_DAY, case.STOP_DAY, 61)
    u = days * 86400 / (case.N * case.DT)
    f = np.geomspace(saved["frequency"][0], saved["frequency"][-1], 181)
    time, freq = spline.design(u, f)
    time, freq = time @ basis["U_time"], freq @ basis["U_freq"]
    g = posterior.g.values.reshape(-1, freq.shape[1])
    eta = posterior.eta.values.reshape(-1, time.shape[1], freq.shape[1])
    correction = (
        np.einsum("ti,dij,fj->dtf", time, eta, freq, optimize=True)
        + (g @ freq.T)[:, None, :]
    )
    response, t0, orbit, galaxy = case.inputs()
    epochs = t0 + days * 86400
    nominal = response.analytic_channel_noise_psd(
        "A", orbit, epochs, f, frequency_chunk=384
    )
    tt, ff = np.meshgrid(epochs, f, indexing="ij")
    foreground = galaxy(np.stack((tt, np.log(ff)), axis=-1))
    truth = nominal + foreground
    logs = correction + np.log(nominal)[None, :, :]
    low, median, high = np.exp(np.quantile(logs, [0.05, 0.5, 0.95], axis=0))
    # Check one saved posterior prediction by independent explicit dot products.
    ti, fi = 25, 60
    direct = np.array(
        [
            time[ti] @ e @ freq[fi] + gg @ freq[fi] + np.log(nominal[ti, fi])
            for gg, e in zip(g, eta, strict=True)
        ]
    )
    np.testing.assert_allclose(logs[:, ti, fi], direct, atol=2e-12)
    # Functional chain checks use a small grid; its 61*181 pixels are not
    # independent repeated realizations for a coverage claim.
    import arviz_stats as azs

    functional = xr.Dataset(
        {
            "log_psd": (
                ("chain", "draw", "t", "f"),
                logs.reshape(2, -1, len(days), len(f))[:, :, ::10, ::30],
            )
        }
    )
    diag = azs.summary(functional, round_to="none")
    functional_diagnostics = {
        "rhat_max": float(diag.r_hat.max()),
        "ess_bulk_min": float(diag.ess_bulk.min()),
        "ess_tail_min": float(diag.ess_tail.min()),
    }
    assert (
        functional_diagnostics["rhat_max"] <= 1.01
        and min(
            functional_diagnostics["ess_bulk_min"],
            functional_diagnostics["ess_tail_min"],
        )
        >= 400
    )
    # Retain the annual WDM posterior as explicitly unmatched historical context.
    historical = json.loads(
        (case.HERE / "focused_study/lisa_preflight.json").read_text()
    )["historical_wdm"][1]
    old = np.load(historical["posterior_path"])
    old_t = interp1d(old["time_grid"], old["tv_basis_time"], axis=0)(u)
    old_f = interp1d(old["frequency_hz"], old["tv_basis_frequency"], axis=0)(f)
    old_g = old["tv_sample_g"].reshape(-1, old_f.shape[1])
    old_eta = old["tv_sample_h"].reshape(-1, old_t.shape[1], old_f.shape[1])
    selected_days = [int(np.argmin(abs(days - d))) for d in (12, 22, 32)]
    old_logs = (
        np.einsum(
            "ti,dij,fj->dtf",
            old_t[selected_days],
            old_eta,
            old_f,
            optimize=True,
        )
        + (old_g @ old_f.T)[:, None, :]
    )
    old_median = np.exp(np.median(old_logs, axis=0)) * nominal[selected_days]
    # Model-check powers use a geometric posterior mean prediction, at exact pairs.
    t_sites, f_sites = spline.design(
        saved["time"].ravel(),
        np.broadcast_to(saved["frequency"], saved["time"].shape).ravel(),
    )
    t_sites, f_sites = t_sites @ basis["U_time"], f_sites @ basis["U_freq"]
    mean_correction = np.einsum(
        "qi,ij,qj->q", t_sites, eta.mean(axis=0), f_sites, optimize=True
    ) + f_sites @ g.mean(axis=0)
    whitened = (saved["power"] / (2 * saved["reference"])).ravel() / np.exp(
        mean_correction
    )
    retained, null_ratio, _ = response.response_null_mask(
        nominal, f, continuum_width=25, dilation_bins=2
    )
    delta = np.log(median) - np.log(truth)
    summary = {
        "scope": "single gapped A-channel interval, days 7–35; no annual-modulation or repeated-coverage claim",
        "fit": fit,
        "functional_diagnostics": functional_diagnostics,
        "log_psd_rmse_all_plot_cells": float(np.sqrt(np.mean(delta**2))),
        "log_psd_rmse_outside_response_nulls": float(
            np.sqrt(np.mean(delta[retained] ** 2))
        ),
        "truth_in_90pct_interval_fraction_all_plot_cells": float(
            np.mean((low <= truth) & (truth <= high))
        ),
        "median_relative_interval_width": float(
            np.median((high - low) / truth)
        ),
        "whitening_mean_normalized_power": float(whitened.mean()),
        "whitening_definition": "observed power divided by the fitted expected power; in-sample descriptive check",
        "historical_wdm": {
            "path": historical["posterior_path"],
            "source_job_id": historical["source_job_id"],
            "matched_comparison": False,
            "reason": "annual WDM model, different time basis, frequency knots and gap guard",
        },
        "prediction_direct_dot_product_verified": True,
    }
    case.write_json(out / "summary.json", summary)
    np.savez_compressed(
        out / "plot_data.npz",
        days=days,
        frequency_hz=f,
        truth=truth,
        nominal=nominal,
        median=median,
        low=low,
        high=high,
        response_null_retained=retained,
        wdm_context_median=old_median,
        selected_day_indices=selected_days,
        whitened_power=whitened,
    )
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(10, 6), sharex=True, layout="constrained"
    )
    for column, index in enumerate(selected_days):
        ax = axes[0, column]
        ax.loglog(
            1000 * f,
            truth[index],
            color="#222222",
            ls=":",
            lw=1.5,
            label="Injected total PSD",
        )
        ax.loglog(
            1000 * f,
            nominal[index],
            color="#777777",
            ls="--",
            lw=1,
            label="Nominal instrument",
        )
        ax.loglog(
            1000 * f,
            median[index],
            color="#2866A2",
            lw=1.4,
            label="Hann MP median",
        )
        ax.fill_between(
            1000 * f,
            low[index],
            high[index],
            color="#2866A2",
            alpha=0.16,
            label="MP 90% interval",
        )
        ax.loglog(
            1000 * f,
            old_median[column],
            color="#B46A1D",
            ls="-.",
            lw=1.2,
            label="Annual WDM (context)",
        )
        ax.set(title=f"Day {days[index]:.1f}", ylabel=r"$S_A$ [Hz$^2$/Hz]")
        ratio = axes[1, column]
        ratio.semilogx(1000 * f, median[index] / truth[index], color="#2866A2")
        ratio.fill_between(
            1000 * f,
            low[index] / truth[index],
            high[index] / truth[index],
            color="#2866A2",
            alpha=0.16,
        )
        ratio.semilogx(
            1000 * f,
            old_median[column] / truth[index],
            color="#B46A1D",
            ls="-.",
        )
        ratio.axhline(1, color="#222222", ls=":", lw=1)
        ratio.set(xlabel="Frequency [mHz]", ylabel="Estimate / injected PSD")
        for item in (ax, ratio):
            item.grid(alpha=0.16, which="major")
    axes[0, 0].legend(fontsize=6.8, loc="best")
    fig.suptitle(
        "Simulated LISA A: gapped day 7-35 Hann moving-periodogram fit"
    )
    fig.supxlabel(
        "Annual WDM is historical context: different time basis, knots and gap guard.",
        fontsize=8,
    )
    export(fig, out, "lisa_moving_hann_spectra")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
    vmin, vmax = np.quantile(np.log10(truth), [0.01, 0.99])
    for ax, values, title in zip(
        axes[0],
        (truth, median),
        ("Injected total point PSD", "Hann MP posterior median"),
        strict=True,
    ):
        im = ax.pcolormesh(
            days,
            f * 1000,
            np.log10(values).T,
            cmap="YlOrBr",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        ax.set(
            yscale="log", title=title, xlabel="Day", ylabel="Frequency [mHz]"
        )
    fig.colorbar(
        im,
        ax=axes[0].tolist(),
        label=r"$\log_{10}(S_A/[\mathrm{Hz}^2/\mathrm{Hz}])$",
        shrink=0.8,
        extend="both",
    )
    ax = axes[1, 0]
    im = ax.pcolormesh(
        days,
        f * 1000,
        (median / truth - 1).T * 100,
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        shading="auto",
        rasterized=True,
    )
    ax.set(
        yscale="log",
        title="Posterior-median fractional error",
        xlabel="Day",
        ylabel="Frequency [mHz]",
    )
    fig.colorbar(im, ax=ax, label="Error [%]", extend="both")
    ax = axes[1, 1]
    selected_frequency = int(np.argmin(abs(f - 0.002)))
    ax.plot(
        days,
        truth[:, selected_frequency],
        color="#222222",
        ls=":",
        label="Injected total",
    )
    ax.plot(
        days, median[:, selected_frequency], color="#2866A2", label="MP median"
    )
    ax.fill_between(
        days,
        low[:, selected_frequency],
        high[:, selected_frequency],
        color="#2866A2",
        alpha=0.16,
        label="90% interval",
    )
    ax.set(
        title=f"{f[selected_frequency] * 1000:.2f} mHz",
        xlabel="Day",
        ylabel=r"$S_A$ [Hz$^2$/Hz]",
    )
    ax.legend(fontsize=7)
    schedule = json.loads(
        (case.LISA / "config/paper_gap_schedule.json").read_text()
    )["gaps_seconds"]
    for ax in axes.flat:
        for lo, hi in schedule:
            if hi / 86400 >= case.START_DAY and lo / 86400 <= case.STOP_DAY:
                ax.axvspan(lo / 86400, hi / 86400, color="#777777", alpha=0.25)
    fig.suptitle(
        "Simulated LISA A, gapped days 7-35; gray bands: missing samples"
    )
    fig.supxlabel(
        "Foreground time variation is oversmoothed in this fit.", fontsize=8
    )
    export(fig, out, "lisa_moving_hann_surface")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    axes[0].hist(
        whitened,
        bins=np.linspace(0, 7, 60),
        density=True,
        color="#2866A2",
        alpha=0.5,
        label="Retained MP powers",
    )
    z = np.linspace(0, 7, 200)
    axes[0].plot(
        z, np.exp(-z), color="#222222", ls=":", label="Exp(1) reference"
    )
    axes[0].set(
        xlabel="Power / fitted expected power",
        ylabel="Density",
        title="In-sample power check",
    )
    axes[0].legend(fontsize=8)
    axes[1].semilogx(
        saved["frequency"] * 1000,
        whitened.reshape(saved["power"].shape).mean(axis=0),
        color="#2866A2",
        lw=0.8,
    )
    axes[1].axhline(1, color="#222222", ls=":")
    axes[1].set(
        xlabel="Frequency [mHz]",
        ylabel="Mean normalized power",
        title=f"Overall mean = {whitened.mean():.3f}",
    )
    fig.suptitle(
        "Hann MP model check: fitted on the same retained observations",
        fontsize=10,
    )
    export(fig, out, "lisa_moving_hann_whitening")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("fit", "historical_wdm")
            },
            indent=2,
        )
    )


def export(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=case.OUT)
    args = parser.parse_args()
    case.OUT = args.directory.resolve()
    main()
