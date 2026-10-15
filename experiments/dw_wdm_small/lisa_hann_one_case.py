"""A bounded physical LISA A-channel Hann moving-periodogram inference case.

Exact selected time-frequency pairs, full-energy normalization, whole-cycle
gap exclusion, nominal ESA-orbit reference, and existing scalar ANOVA priors.
No annual inference or physical component separation is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import ZoomFFT

HERE = Path(__file__).resolve().parent
OUT = HERE / "lisa_hann_one_case"
LISA = Path("/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")
LEGACY = Path("/Users/avi/Documents/projects/wdm_psd/wdm_psd")
M, DT, N = 7680, 2.0, 15728640
LENGTH = 2 * M + 1
START_DAY, STOP_DAY = 7, 35
FMIN, FMAX = 1e-4, 0.1


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def physical_module():
    sys.path.insert(0, str(LEGACY))
    sys.path.insert(0, str(LISA / "src/lisa_study"))
    import run_surface_study

    return run_surface_study


class OffsetFourier:
    """Changing-frequency selected DFT, with a common frequency displacement."""

    def __init__(self, m: int, offset: float = 0.0, leaf: int = 32):
        self.m, self.length, self.nodes = m, 2 * m + 1, []
        self.wrap_factor = np.exp(-2j * np.pi * offset * self.length)
        frequency = (np.arange(m) + 1) / self.length + offset
        self.base = ZoomFFT(
            self.length,
            [frequency[0], frequency[-1] + 1 / self.length],
            m=m,
            fs=1,
        )
        self.phase = np.exp(2j * np.pi * frequency * np.arange(m))

        def build(lo: int, hi: int) -> None:
            if hi - lo <= leaf:
                k = np.arange(lo, hi)
                matrix = np.exp(
                    -2j * np.pi * frequency[lo:hi, None] * k[None, :]
                ) * (k[None, :] < k[:, None])
                self.nodes.append((lo, hi, None, matrix, None))
                return
            mid = (lo + hi) // 2
            build(lo, mid)
            build(mid, hi)
            fft = ZoomFFT(
                mid - lo,
                [frequency[mid], frequency[hi - 1] + 1 / self.length],
                m=hi - mid,
                fs=1,
            )
            phase = np.exp(-2j * np.pi * frequency[mid:hi] * lo)
            self.nodes.append((lo, hi, mid, fft, phase))

        build(0, m)

    def block(self, x: np.ndarray) -> np.ndarray:
        assert len(x) == self.length + self.m - 1
        delta = np.r_[
            self.wrap_factor * x[self.length :] - x[: self.m - 1], 0.0
        ]
        prefix = np.zeros(self.m, complex)
        for lo, hi, mid, plan, phase in self.nodes:
            if mid is None:
                prefix[lo:hi] += plan @ delta[lo:hi]
            else:
                prefix[mid:hi] += plan(delta[lo:mid]) * phase
        return self.phase * (self.base(x[: self.length]) + prefix)


class SelectedHann:
    def __init__(self, m: int):
        self.m = m
        self.h = np.hanning(2 * m + 1)
        self.plans = [
            OffsetFourier(m, offset)
            for offset in (0.0, 1 / (2 * m), -1 / (2 * m))
        ]

    def block(self, x: np.ndarray) -> np.ndarray:
        a, b, c = [p.block(x) for p in self.plans]
        return 0.5 * a - 0.25 * (b + c)


def hann_kernel(f: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """One-sided real-signal Hann spectral window, including its mirror."""

    def transform(delta: np.ndarray) -> np.ndarray:
        z = DT * delta

        def d(a: np.ndarray) -> np.ndarray:
            return (
                LENGTH
                * np.exp(-1j * np.pi * (LENGTH - 1) * a)
                * np.sinc(LENGTH * a)
                / np.sinc(a)
            )

        return 0.5 * d(z) - 0.25 * (
            d(z + 1 / (LENGTH - 1)) + d(z - 1 / (LENGTH - 1))
        )

    energy = 3 * (LENGTH - 1) / 8
    return (
        DT
        / energy
        * (abs(transform(f - fp)) ** 2 + abs(transform(f + fp)) ** 2)
    )


def inputs():
    response = physical_module()
    with h5py.File(LISA / "data/lisa.h5", "r") as h:
        assert h.attrs["n_samples"] == N and h.attrs["dt_seconds"] == DT
        t0 = float(h.attrs["t0_tcb"])
        times, freq = h["truth/time_tcb"][:], h["truth/frequency_hz"][:]
        gal = response._channel_truth(h["truth/galactic_csd"][:], "A")
        orbit_bytes = h["inputs/orbits"][:].tobytes()
        orbit_hash = hashlib.sha256(orbit_bytes).hexdigest()
        assert orbit_hash == h["inputs/orbits"].attrs["sha256"]
    orbit = OUT / "orbits.h5"
    orbit.write_bytes(orbit_bytes)
    galaxy = RegularGridInterpolator(
        (times, np.log(freq)), gal, bounds_error=False, fill_value=0.0
    )
    return response, t0, orbit, galaxy


def projected_reference(
    response,
    orbit: Path,
    epochs: np.ndarray,
    frequency: np.ndarray,
    *,
    oversample: int = 8,
    radius: int = 16,
):
    """Integrate each local kernel on a shared, oversampled Fourier grid."""
    df = 1 / (LENGTH * DT * oversample)
    fq = (
        np.arange(int(np.ceil((frequency[-1] + radius / (LENGTH * DT)) / df)))
        + 0.5
    ) * df
    spectra = response.analytic_channel_noise_psd(
        "A", orbit, epochs, fq, frequency_chunk=384
    )
    output = np.empty((len(epochs), len(frequency)))
    mass = np.empty(len(frequency))
    for i, f in enumerate(frequency):
        selected = abs(fq - f) <= radius / (LENGTH * DT)
        k = hann_kernel(f, fq[selected]) * df
        output[:, i] = spectra[:, selected] @ k
        mass[i] = k.sum()
    return output, mass


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    response, t0, orbit, galaxy = inputs()
    bins = np.array([6, 30, 93, 300, 768, 1536, 3072])
    frequency = bins / (LENGTH * DT)
    epochs = t0 + np.array([START_DAY, 14, STOP_DAY]) * 86400
    projection = []
    for oversample in (8, 16):
        nfft = oversample * LENGTH
        df = 1 / (nfft * DT)
        fq = (np.arange(nfft // 2) + 0.5) * df
        noise = response.analytic_channel_noise_psd(
            "A", orbit, epochs, fq, frequency_chunk=384
        )
        k = hann_kernel(frequency[:, None], fq[None, :]) * df
        gal = np.array(
            [
                galaxy(np.column_stack((np.full(len(fq), epoch), np.log(fq))))
                for epoch in epochs
            ]
        )
        ref, total = noise @ k.T, (noise + gal) @ k.T
        point = response.analytic_channel_noise_psd(
            "A", orbit, epochs, frequency
        )
        point_gal = np.array(
            [
                galaxy(
                    np.column_stack(
                        (np.full(len(frequency), epoch), np.log(frequency))
                    )
                )
                for epoch in epochs
            ]
        )
        factor_error = ref * (1 + point_gal / point) / total - 1
        local, mass = projected_reference(
            response, orbit, epochs, frequency, oversample=oversample
        )
        projection.append(
            {
                "oversample": oversample,
                "reference": ref.tolist(),
                "total": total.tolist(),
                "point_total": (point + point_gal).tolist(),
                "factorization_relative_error": factor_error.tolist(),
                "white_kernel_mass": k.sum(axis=1).tolist(),
                "local_reference_relative_error": (local / ref - 1).tolist(),
                "local_kernel_mass": mass.tolist(),
            }
        )
        print(
            f"Reference audit: oversample={oversample}, max factorization error={np.max(abs(factor_error)):.6g}",
            flush=True,
        )
    rng = np.random.default_rng(20261015)
    engine = SelectedHann(32)
    x = rng.normal(size=3 * 32)
    coeff = engine.block(x)
    direct = np.array(
        [
            x[j : j + 65]
            @ (engine.h * np.exp(-2j * np.pi * (j + 1) * np.arange(65) / 65))
            for j in range(32)
        ]
    )
    np.testing.assert_allclose(coeff, direct, atol=1e-10)
    error = float(
        np.max(abs(np.asarray(projection[-1]["factorization_relative_error"])))
    )
    local_error = float(
        np.max(
            abs(np.asarray(projection[-1]["local_reference_relative_error"]))
        )
    )
    refine = float(
        np.max(
            abs(
                np.asarray(projection[-1]["reference"])
                / np.asarray(projection[-2]["reference"])
                - 1
            )
        )
    )
    report = {
        "status": "pass"
        if error < 0.02 and local_error < 0.005 and refine < 0.005
        else "review_needed",
        "frequency_hz": frequency.tolist(),
        "epochs_tcb": epochs.tolist(),
        "projections": projection,
        "max_factorization_relative_error": error,
        "max_local_reference_relative_error": local_error,
        "quadrature_refinement_relative_error": refine,
        "selected_hann_direct_max_error": float(np.max(abs(coeff - direct))),
        "approximation": "nominal reference integrated through each Hann kernel; residual PSD correction locally constant over the kernel; physical response frozen within a window",
        "window_samples": LENGTH,
        "window_hours": LENGTH * DT / 3600,
        "channel": "A",
        "case": "gapped, days 7–35",
        "band_hz": [FMIN, FMAX],
    }
    write_json(OUT / "preflight.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "projections"}, indent=2
        ),
        flush=True,
    )


def prepare(*, complete: bool = False) -> None:
    """Prepare the fixed interval with original context and optional restored availability."""
    assert json.loads((OUT / "preflight.json").read_text())["status"] == "pass"
    response, t0, orbit, galaxy = inputs()
    schedule = json.loads(
        (LISA / "config/paper_gap_schedule.json").read_text()
    )
    start_s, stop_s = START_DAY * 86400, STOP_DAY * 86400
    start_index = ((int(start_s / DT) - 4 * M) // (2 * M)) * (2 * M)
    stop_index = int(stop_s / DT) + 4 * M
    with h5py.File(LISA / "data/lisa.h5", "r") as h:
        x = response.xyz_to_aet_series(
            h["tdi/total"][:, start_index:stop_index]
        )[0]
        full = hashlib.sha256()
        for channel in range(3):
            for offset in range(0, N, 1000000):
                full.update(
                    h["tdi/total"][
                        channel, offset : offset + 1000000
                    ].tobytes()
                )
        contract = json.loads(
            (LISA / "config/knots/contract.json").read_text()
        )
        assert full.hexdigest() == contract["tdi_total_sha256"]
    observed = np.ones(len(x), dtype=bool)
    for lo, hi in ([] if complete else schedule["gaps_seconds"]):
        left = max(0, int(np.floor(lo / DT)) - start_index)
        right = max(0, min(len(x), int(np.ceil(hi / DT)) + 1 - start_index))
        if left < right:
            observed[left:right] = False
    frequency = np.arange(1, M + 1) / (LENGTH * DT)
    selected = (
        ((np.arange(1, M + 1) % 3) == 0)
        & (frequency >= FMIN)
        & (frequency <= FMAX)
    )
    index = np.flatnonzero(selected)
    frequency = frequency[selected]
    prefix = np.r_[0, np.cumsum(~observed)]
    engine = SelectedHann(M)
    power, center, cycle_time, direct_errors = [], [], [], []
    rejected = 0
    started = perf_counter()
    for block in range((len(x) - 2 * M) // (2 * M)):
        start = 2 * M * block
        centers = start + M + np.arange(M) + start_index
        inside = (centers[index] * DT >= start_s) & (
            centers[index] * DT < stop_s
        )
        if not inside.all():
            continue
        left, right = start, start + LENGTH + M - 1
        if prefix[right] != prefix[left]:
            rejected += 1
            continue
        values = engine.block(x[left:right])[selected]
        if len(power) in (0, 20, 40):
            for j in index[[0, len(index) // 2, -1]]:
                direct = x[start + j : start + j + LENGTH] @ (
                    engine.h
                    * np.exp(
                        -2j * np.pi * (j + 1) * np.arange(LENGTH) / LENGTH
                    )
                )
                actual = values[np.searchsorted(index, j)]
                direct_errors.append(
                    float(abs(actual - direct) / max(abs(direct), 1e-30))
                )
        power.append(4 * DT * abs(values) ** 2 / np.sum(engine.h**2))
        center.append(centers[index])
        cycle_time.append(float(centers[index].mean() / N))
    power, center = np.array(power), np.array(center)
    assert np.max(direct_errors) < 1e-7 and np.isfinite(power).all()
    transform_seconds = perf_counter() - started
    anchors = t0 + np.linspace(start_s, stop_s, 15)
    reference_started = perf_counter()
    reference, mass = projected_reference(response, orbit, anchors, frequency)
    point = response.analytic_channel_noise_psd(
        "A", orbit, anchors, frequency, frequency_chunk=384
    )
    epochs = t0 + center * DT
    coords = np.stack(
        (epochs, np.broadcast_to(frequency, epochs.shape)), axis=-1
    )
    ref_at_sites = RegularGridInterpolator((anchors, frequency), reference)(
        coords
    )
    check_epochs = t0 + np.array([8, 15, 34]) * 86400
    direct_reference, _ = projected_reference(
        response, orbit, check_epochs, frequency[::60]
    )
    grid = np.stack(
        np.meshgrid(check_epochs, frequency[::60], indexing="ij"), axis=-1
    )
    interpolated = RegularGridInterpolator((anchors, frequency), reference)(
        grid
    )
    time_error = float(np.max(abs(interpolated / direct_reference - 1)))
    assert time_error < 0.005
    truth_point = (
        np.array(
            [
                galaxy(
                    np.column_stack(
                        (np.full(len(frequency), epoch), np.log(frequency))
                    )
                )
                for epoch in anchors
            ]
        )
        + point
    )
    np.savez_compressed(
        OUT / "prepared.npz",
        power=power,
        sample_index=center,
        time=center / N,
        frequency=frequency,
        reference=ref_at_sites,
        cycle_time=cycle_time,
        anchor_epochs=anchors,
        point_reference=point,
        projected_reference_grid=reference,
        truth_point=truth_point,
    )
    metadata = {
        "case": f"A channel, {'complete' if complete else 'gapped'}, days 7–35",
        "mask_condition": "complete" if complete else "gapped",
        "context_start_index": start_index,
        "context_stop_index": stop_index,
        "raw_path": str(LISA / "data/lisa.h5"),
        "tdi_total_sha256": full.hexdigest(),
        "raw_context_A_sha256": digest(x),
        "availability_sha256": digest(observed),
        "gap_schedule_sha256": hashlib.sha256(
            (LISA / "config/paper_gap_schedule.json").read_bytes()
        ).hexdigest(),
        "selected_cycles": len(power),
        "rejected_gap_cycles": rejected,
        "observations": power.size,
        "real_component_count": 2 * power.size,
        "frequency_rungs": (index + 1).tolist(),
        "actual_frequency_endpoints_hz": [
            float(frequency[0]),
            float(frequency[-1]),
        ],
        "selected_hann_direct_max_relative_error": float(max(direct_errors)),
        "transform_seconds": transform_seconds,
        "reference_seconds": perf_counter() - reference_started,
        "reference_time_interpolation_max_relative_error": time_error,
        "min_local_kernel_mass": float(mass.min()),
        "no_truth_in_fit_inputs": True,
        "whole_cycle_gap_exclusion": True,
        "annual_inference": False,
    }
    write_json(OUT / "preparation.json", metadata)
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k != "frequency_rungs"},
            indent=2,
        ),
        flush=True,
    )


def fit_case(
    warmup: int, samples: int, *, noncentered_eta: bool = False
) -> None:
    import arviz_stats as azs
    import jax
    from experiments.dw_wdm_small.lisa_hann_model import prepare_paired_model
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import init_to_value, initialize_model

    from log_psplines import (
        ANOVALogPSpline,
        PowerConfig,
        PowerData,
        SplineBasis,
    )
    from log_psplines.inference.nuts import MCMCResult, _mapping_to_dataset

    jax.config.update("jax_enable_x64", True)
    saved = np.load(OUT / "prepared.npz")
    original = json.loads(
        (
            LISA / "config/knots/variation_20260924/continuous_orb_A.json"
        ).read_text()
    )
    bf = SplineBasis.from_grid(
        np.linspace(FMIN, FMAX, 100), interior_knots=original["knots_hz"]
    )
    # Fixed full-case centering; gaps never choose a new reference mean.
    reference_time = np.linspace(
        START_DAY * 86400 / (N * DT), STOP_DAY * 86400 / (N * DT), 2046
    )
    bt = SplineBasis.from_grid(reference_time, 3)
    spline = ANOVALogPSpline(bf, bt, sigma_eta_prior=0.5)
    shape = saved["power"].shape
    data = PowerData(
        (saved["power"] / saved["reference"]).ravel(),
        np.full(np.prod(shape), 2.0),
        np.broadcast_to(saved["frequency"], shape).ravel(),
        saved["time"].ravel(),
    )
    config = PowerConfig(
        structure="anova",
        centered=True,
        dense_mass=True,
        roughness_scale=10,
        interaction_scale=0.5,
        n_warmup=warmup,
        n_samples=samples,
        num_chains=2,
        target_accept_prob=0.99,
        max_tree_depth=12,
        seed=20261015,
        progress_bar=True,
    )
    model, pair, init, _ = prepare_paired_model(
        data,
        spline,
        config,
        saved["cycle_time"],
        noncentered_eta=noncentered_eta,
    )
    assert sum(np.size(v) for v in init.values()) == 114
    started = perf_counter()
    model_info = initialize_model(
        jax.random.PRNGKey(config.seed),
        model,
        init_strategy=init_to_value(values=init),
    )
    flat, unravel = ravel_pytree(model_info.param_info.z)
    curvature = np.asarray(
        jax.jit(jax.hessian(lambda z: model_info.potential_fn(unravel(z))))(
            flat
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh((curvature + curvature.T) / 2)
    floor = max(1e-3, float(np.median(abs(eigenvalues))) * 1e-6)
    covariance = (
        eigenvectors / np.maximum(abs(eigenvalues), floor)
    ) @ eigenvectors.T
    if noncentered_eta:
        # Standard-normal eta coordinates retain their natural unit scale.
        # Initial likelihood curvature can otherwise freeze weak directions
        # when the seasonal hyperparameter shrinks during warmup.
        cursor = 0
        for name in sorted(model_info.param_info.z):
            size = np.size(model_info.param_info.z[name])
            if name == "g":
                g_slice = slice(cursor, cursor + size)
            cursor += size
        g_curvature = curvature[g_slice, g_slice]
        gv, gu = np.linalg.eigh((g_curvature + g_curvature.T) / 2)
        covariance = np.eye(len(flat))
        covariance[g_slice, g_slice] = (gu / np.maximum(gv, 1e-3)) @ gu.T
    write_json(
        OUT / "mass_initialization.json",
        {
            "method": "conditional g curvature, unit eta coordinates, fixed mass"
            if noncentered_eta
            else "absolute initial posterior curvature; positive floor; adaptive mass",
            "eigenvalue_min": float(eigenvalues.min()),
            "eigenvalue_max": float(eigenvalues.max()),
            "floor": floor,
            "seconds": perf_counter() - started,
        },
    )
    mcmc = MCMC(
        NUTS(
            model,
            init_strategy=init_to_value(values=init),
            inverse_mass_matrix=covariance,
            dense_mass=True,
            adapt_mass_matrix=not noncentered_eta,
            target_accept_prob=0.99,
            max_tree_depth=12,
        ),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=2,
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(config.seed),
        extra_fields=(
            "potential_energy",
            "energy",
            "num_steps",
            "accept_prob",
            "diverging",
        ),
    )
    draws = mcmc.get_samples(group_by_chain=True)
    stats = mcmc.get_extra_fields(group_by_chain=True)
    stats = {
        {"num_steps": "n_steps", "accept_prob": "acceptance_rate"}.get(k, k): v
        for k, v in stats.items()
    }
    result = MCMCResult(
        _mapping_to_dataset(draws, num_chains=2),
        _mapping_to_dataset(stats, num_chains=2),
    )
    elapsed = perf_counter() - started
    result.posterior.to_netcdf(OUT / "posterior.nc")
    result.sample_stats.to_netcdf(OUT / "sample_stats.nc")
    diag = azs.summary(
        result.posterior[
            [
                k
                for k in ("g", "eta", "z_eta", "sigma_g", "sigma_eta")
                if k in result.posterior
            ]
        ],
        round_to="none",
    )
    stats = result.sample_stats
    energy = np.asarray(stats.energy)
    ebfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(
        energy, axis=1, ddof=1
    )
    diagnostics = {
        "rhat_max": float(diag.r_hat.max()),
        "ess_bulk_min": float(diag.ess_bulk.min()),
        "ess_tail_min": float(diag.ess_tail.min()),
        "divergences": int(stats.diverging.sum()),
        "depth_hits": int((stats.n_steps >= 2**12 - 1).sum()),
        "ebfmi_min": float(ebfmi.min()),
    }
    diagnostics["pass"] = (
        diagnostics["rhat_max"] <= 1.01
        and min(diagnostics["ess_bulk_min"], diagnostics["ess_tail_min"])
        >= 400
        and diagnostics["divergences"] == 0
        and diagnostics["depth_hits"] == 0
        and diagnostics["ebfmi_min"] >= 0.3
    )
    write_json(
        OUT / "fit.json",
        {
            "config": asdict(config),
            "diagnostics": diagnostics,
            "inference_seconds": elapsed,
            "parameters": 114,
            "model": "existing H_orb scalar ANOVA prior; paired predictor adapter",
            "time_basis_scope": "full day 7–35 interval; fixed centering before applying gaps",
            "parameterization": "centered g, noncentered eta"
            if noncentered_eta
            else "centered eigen-coefficients",
            "mass_initialization": "conditional g curvature, unit eta coordinates, fixed mass"
            if noncentered_eta
            else "curvature-initialized adaptive dense mass",
            "approximation": "Hann-projected nominal reference multiplied by locally constant log correction",
            "time_transform_sha256": digest(spline.time_transform),
            "frequency_knots_sha256": digest(bf.knots),
        },
    )
    np.savez_compressed(
        OUT / "model_basis.npz",
        time_transform=spline.time_transform,
        time_knots=bt.knots,
        frequency_knots=bf.knots,
        U_time=pair["U_time"],
        U_freq=pair["U_freq"],
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare", "fit"))
    parser.add_argument("--warmup", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--noncentered-eta", action="store_true")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    if args.output_name:
        import shutil

        original = OUT
        OUT = OUT / args.output_name
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / "prepared.npz", OUT / "prepared.npz")
    if args.action == "audit":
        audit()
    elif args.action == "prepare":
        prepare()
    elif args.action == "fit":
        fit_case(
            args.warmup, args.samples, noncentered_eta=args.noncentered_eta
        )


if __name__ == "__main__":
    main()
