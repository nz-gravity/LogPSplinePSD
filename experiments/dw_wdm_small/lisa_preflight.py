"""Bounded physical-reference and MP transform preflight; no annual inference.

The large-window adapter evaluates only the existing Tang schedule. Prefix
Fourier sums use cached ZoomFFT plans; no window-by-record array is formed.
Run with the installed companion Python environment (physical response deps).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import ZoomFFT

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_study"
LISA = Path("/Users/avi/Documents/projects/wdm_psd/lisa_data_generation")


class SelectedFourier:
    """FFT evaluation of one changing frequency per adjacent window."""

    def __init__(self, m, leaf=32):
        self.m = m
        self.length = 2 * m + 1
        self.leaf = leaf
        self.nodes = []
        self.base = ZoomFFT(
            self.length, [1 / self.length, (m + 1) / self.length], m=m, fs=1
        )
        self.phase = np.exp(
            2j * np.pi * (np.arange(m) + 1) * np.arange(m) / self.length
        )

        def build(lo, hi):
            j = np.arange(lo, hi)
            if hi - lo <= leaf:
                k = np.arange(lo, hi)
                matrix = np.exp(
                    -2j * np.pi * (j[:, None] + 1) * k[None, :] / self.length
                ) * (k[None, :] < j[:, None])
                self.nodes.append((lo, hi, None, matrix, None))
                return
            mid = (lo + hi) // 2
            build(lo, mid)
            build(mid, hi)
            fft = ZoomFFT(
                mid - lo,
                [(mid + 1) / self.length, (hi + 1) / self.length],
                m=hi - mid,
                fs=1,
            )
            phase = np.exp(
                -2j * np.pi * (np.arange(mid, hi) + 1) * lo / self.length
            )
            self.nodes.append((lo, hi, mid, fft, phase))

        build(0, m)

    def block(self, x):
        """x contains L+m-1 samples; returns unnormalised selected DFTs."""
        assert len(x) == self.length + self.m - 1
        delta = np.r_[x[self.length :] - x[: self.m - 1], 0.0]
        prefix = np.zeros(self.m, complex)
        for lo, hi, mid, plan, phase in self.nodes:
            if mid is None:
                prefix[lo:hi] += plan @ delta[lo:hi]
            else:
                prefix[mid:hi] += plan(delta[lo:mid]) * phase
        return self.phase * (self.base(x[: self.length]) + prefix)


def kernel(f, fp, length, dt):
    """One-sided real-signal projection, including negative-frequency mirror."""

    def d2(delta):
        z = dt * delta
        return (length * np.sinc(length * z) / np.sinc(z)) ** 2

    return dt / length * (d2(f - fp) + d2(f + fp))


def main():
    OUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(LISA / "src/lisa_study"))
    import run_surface_study as response

    datafile = LISA / "data/lisa.h5"
    schedule = json.loads(
        (LISA / "config/paper_gap_schedule.json").read_text()
    )
    with h5py.File(datafile, "r") as h:
        dt = float(h.attrs["dt_seconds"])
        n = int(h.attrs["n_samples"])
        t0 = float(h.attrs["t0_tcb"])
        assert dt == 2 and n == 15728640
        raw_orbit = h["inputs/orbits"][:].tobytes()
        orbit = OUT / "orbits.h5"
        assert (
            hashlib.sha256(raw_orbit).hexdigest()
            == h["inputs/orbits"].attrs["sha256"]
        )
        orbit.write_bytes(raw_orbit)
        # First scheduled outage lies 14 days into a deterministic first 28 days.
        start_s = 7 * 86400.0
        stop_s = 35 * 86400.0
        first_gap = schedule["gaps_seconds"][0]
        assert start_s < first_gap[0] < first_gap[1] < stop_s
        nt = 2048
        nf = n // nt
        m = nf
        length = 2 * m + 1
        context = (
            4 * nf
        )  # Full WDM four-cell context; MP needs only its half-width.
        start_index = ((int(start_s / dt) - context) // (2 * m)) * (2 * m)
        stop_index = int(stop_s / dt) + context
        xyz = h["tdi/total"][:, start_index:stop_index]
        x = response.xyz_to_aet_series(xyz)[0]
        assert np.isfinite(x).all()
        # Confirm archive raw units from authoritative metadata, not evaluator docstring.
        units = str(h.attrs["psd_convention"])
        times = h["truth/time_tcb"][:]
        freq = h["truth/frequency_hz"][:]
        gal = response._channel_truth(h["truth/galactic_csd"][:], "A")
        galactic = RegularGridInterpolator(
            (times, np.log(freq)), gal, bounds_error=False, fill_value=0.0
        )
        input_info = {
            "path": str(datafile),
            "size_bytes": datafile.stat().st_size,
            "mtime_ns": datafile.stat().st_mtime_ns,
            "cadence_seconds": dt,
            "samples": n,
            "t0_tcb": t0,
            "units": units,
            "embedded_orbit_sha256": hashlib.sha256(raw_orbit).hexdigest(),
            "raw_28day_A_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        }
        full_digest = hashlib.sha256()
        for channel in range(3):
            for offset in range(0, n, 1000000):
                full_digest.update(
                    h["tdi/total"][
                        channel, offset : offset + 1000000
                    ].tobytes()
                )
        input_info["tdi_total_sha256"] = full_digest.hexdigest()
        contract = json.loads(
            (LISA / "config/knots/contract.json").read_text()
        )
        assert full_digest.hexdigest() == contract["tdi_total_sha256"]
    observed = np.ones(len(x), bool)
    for lo, hi in schedule["gaps_seconds"]:
        left = max(0, int(np.floor(lo / dt)) - start_index)
        right = max(0, min(len(x), int(np.ceil(hi / dt)) + 1 - start_index))
        if left < right:
            observed[left:right] = False
    # Chunking at original global cycle starts preserves absolute coordinates/phase.
    fft_start = perf_counter()
    engine = SelectedFourier(m)
    setup_seconds = perf_counter() - fft_start
    blocks = (len(x) - 2 * m) // (2 * m)
    centers = []
    powers = []
    gap_powers = []
    kept = []
    direct_error = []
    mask_error = []
    fft_start = perf_counter()
    missing_prefix = np.r_[0, np.cumsum(~observed)]
    for block in range(blocks):
        start = 2 * m * block
        values = engine.block(x[start : start + length + m - 1])
        masked_values = engine.block(
            np.where(
                observed[start : start + length + m - 1],
                x[start : start + length + m - 1],
                0.0,
            )
        )
        cc = start + m + np.arange(m)
        valid = (missing_prefix[cc + m + 1] - missing_prefix[cc - m]) == 0
        inside = (cc + start_index >= start_s / dt) & (
            cc + start_index < stop_s / dt
        )
        mask_error.append(
            float(
                np.max(abs(values[valid] - masked_values[valid]), initial=0)
                / max(np.linalg.norm(x[start : start + length + m - 1]), 1e-30)
            )
        )
        centers.append(cc[inside] + start_index)
        powers.append((4 * dt * abs(values) ** 2 / length)[inside])
        gap_powers.append((4 * dt * abs(masked_values) ** 2 / length)[inside])
        kept.append(valid[inside])
        if block in (0, blocks // 2, blocks - 1):
            for j in (0, m // 3, m // 2, m - 1):
                direct = x[start + j : start + j + length] @ np.exp(
                    -2j * np.pi * (j + 1) * np.arange(length) / length
                )
                direct_error.append(
                    abs(values[j] - direct) / max(abs(direct), 1e-30)
                )
    transform_seconds = perf_counter() - fft_start
    assert max(direct_error) < 2e-8
    assert max(mask_error) < 2e-8
    centers = np.concatenate(centers)
    power = np.concatenate(powers)
    keep = np.concatenate(kept)
    frequency = ((centers - m) % (2 * m) + 1) / (length * dt)
    np.savez_compressed(
        OUT / "lisa_smoke_powers.npz",
        sample_index=centers,
        power=power,
        power_gap=np.concatenate(gap_powers),
        keep=keep,
        frequency=frequency,
        time=centers / n,
        absolute_epoch=t0 + centers * dt,
    )
    # Preserve the physical WDM spacing in the bounded interval: NF stays
    # fixed, NT changes with interval length. Apply its original taper/guard.
    from wdm_transform import TimeSeries

    wdm_length = (len(x) // (2 * nf)) * (2 * nf)
    wdm_nt = wdm_length // nf
    local_gaps = [
        (lo - start_index * dt, hi - start_index * dt)
        for lo, hi in schedule["gaps_seconds"]
        if hi >= start_index * dt - 3600
        and lo <= (start_index + wdm_length) * dt + 3600
    ]
    wdm_start = perf_counter()
    wc = np.asarray(
        TimeSeries(x[:wdm_length], dt=dt).to_wdm(nt=wdm_nt).coeffs
    )[0, 1:-1, 1:-1]
    wg = np.asarray(
        TimeSeries(
            response.gate_gaps(x[:wdm_length], dt, local_gaps, taper_s=3600),
            dt=dt,
        )
        .to_wdm(nt=wdm_nt)
        .coeffs
    )[0, 1:-1, 1:-1]
    wtime = (start_index + np.arange(1, wdm_nt - 1) * nf) / n
    rows = response.good_time_bins(
        wtime,
        n * dt,
        schedule["gaps_seconds"],
        nt,
        buffer_pixels=4,
        taper_s=3600,
        edge_buffer_pixels=4,
    )
    rows &= (wtime * n * dt >= start_s) & (wtime * n * dt < stop_s)
    wf = np.arange(1, nf) / (2 * nf * dt)
    fkeep = (wf >= 1e-4) & (wf <= 0.1)
    assert np.isfinite(wc).all() and np.isfinite(wg).all()
    wdm_info = {
        "bounded_nt": wdm_nt,
        "fixed_nf": nf,
        "physical_frequency_spacing_hz": 1 / (2 * nf * dt),
        "retained_rows": int(rows.sum()),
        "retained_cells": int(rows.sum() * fkeep.sum()),
        "transform_seconds": perf_counter() - wdm_start,
        "real_component_count_per_cell": 1,
        "availability_mask_sha256": hashlib.sha256(
            observed.tobytes()
        ).hexdigest(),
        "policy": "one-hour cosine taper, four-cell gap/edge guard; global times retained",
    }
    # Full-Nyquist midpoint quadrature, refined before any posterior inference.
    frequency_bins = np.unique(
        np.rint(
            np.array([0.00013, 0.001, 0.003, 0.01, 0.025, 0.05, 0.1])
            * length
            * dt
        ).astype(int)
    )
    fp = frequency_bins / (length * dt)
    epoch = t0 + first_gap[0]
    projection = []
    ref_start = perf_counter()
    for oversample in (4, 8, 16):
        nfft = oversample * length
        df = 1 / (nfft * dt)
        fq = (np.arange(nfft // 2) + 0.5) * df
        nominal = response.analytic_channel_noise_psd(
            "A", orbit, np.array([epoch]), fq, frequency_chunk=384
        )[0]
        galaxy = galactic(
            np.column_stack((np.full(len(fq), epoch), np.log(fq)))
        )
        point = response.analytic_channel_noise_psd(
            "A", orbit, np.array([epoch]), fp, frequency_chunk=96
        )[0]
        gal_point = galactic(
            np.column_stack((np.full(len(fp), epoch), np.log(fp)))
        )
        k = np.array([kernel(fq, f, length, dt) for f in fp])
        norm = k.sum(axis=1) * df
        assert np.max(abs(norm - 1)) < 1e-8
        ref = k @ nominal * df
        total = k @ (nominal + galaxy) * df
        # Integrate the candidate band with its exact endpoints, avoiding a
        # discontinuous cutoff on the full-Nyquist midpoint grid.
        edges = np.r_[np.arange(1e-4, 0.1, 1 / (length * dt)), 0.1]
        nodes, weights = np.polynomial.legendre.leggauss(oversample // 2)
        widths = np.diff(edges)
        bf = (
            edges[:-1, None] + (nodes[None, :] + 1) * widths[:, None] / 2
        ).ravel()
        bw = (
            np.ones((len(widths), 1)) * weights[None, :] * widths[:, None] / 2
        ).ravel()
        bnom = response.analytic_channel_noise_psd(
            "A", orbit, np.array([epoch]), bf, frequency_chunk=384
        )[0]
        bk = np.array([kernel(bf, f, length, dt) for f in fp])
        band_ref = bk @ (bnom * bw)
        predicted = ref * (1 + gal_point / point)
        projection.append(
            {
                "oversample": oversample,
                "nodes": len(fq),
                "frequency_hz": fp.tolist(),
                "white_kernel_mass": norm.tolist(),
                "reference_mean": ref.tolist(),
                "point_reference": point.tolist(),
                "outside_candidate_band_fraction": (
                    1 - band_ref / ref
                ).tolist(),
                "total_projected_mean": total.tolist(),
                "point_total": (point + gal_point).tolist(),
                "correction_factorization_relative_error": (
                    predicted / total - 1
                ).tolist(),
            }
        )
        print(
            "projection",
            oversample,
            "factorization error",
            projection[-1]["correction_factorization_relative_error"],
            flush=True,
        )
    quadrature_relative = float(
        np.max(
            abs(
                np.array(projection[-1]["reference_mean"])
                / projection[-2]["reference_mean"]
                - 1
            )
        )
    )
    full_reuse = []
    release = json.loads(
        (
            LISA / "results/lisa_variation_A_hn10_20260928.release.json"
        ).read_text()
    )
    for row in release["replacements"]:
        if "/orb/" not in row["key"]:
            continue
        marker = Path(row["marker"])
        payload = json.loads(marker.read_text())
        out = marker.parent.parent / row["key"].replace("/", "_")
        metrics = next(out.glob("*.json"))
        meta = json.loads(metrics.read_text())
        full_reuse.append(
            {
                "key": row["key"],
                "posterior_path": str(next(out.glob("*_chains.npz"))),
                "source_job_id": marker.parent.parent.name.split("_")[-1],
                "source_command": meta.get("runtime_provenance", {}).get(
                    "command", []
                ),
                "knots_hz": meta["frequency_knots_hz"],
                "sampler_gate": payload["gate"],
                "eligible_as_matched_pair": False,
                "reason": "Separate complete/gapped knot sets; fixed common centring/prior parity must also be checked.",
            }
        )
    factor_error = float(
        np.max(
            abs(
                np.array(
                    projection[-1]["correction_factorization_relative_error"]
                )
            )
        )
    )
    blockers = []
    if quadrature_relative > 0.005:
        blockers.append("Full-Nyquist MP quadrature is not converged to 0.5%.")
    if factor_error > 0.02:
        blockers.append(
            "Pointwise injected total/reference correction cannot be pulled outside rectangular MP kernel within provisional 2% at the predeclared sites."
        )
    # Freeze the actual saved LISA-specific prior, and correct its description
    # in the companion. Historical draws are not altered by this text repair.
    result = {
        "status": "blocked_reference_preflight",
        "pid": os.getpid(),
        "inputs": input_info,
        "smoke_interval_seconds": [start_s, stop_s],
        "context_samples": context,
        "first_scheduled_gap_seconds": first_gap,
        "window_samples": length,
        "window_hours": length * dt / 3600,
        "wdm_nf": nf,
        "wdm_nt": nt,
        "wdm_time_pixel_hours": nf * dt / 3600,
        "mp_frequency_spacing_hz": 1 / (length * dt),
        "basis_raw": [7, 16],
        "basis_after_time_centering": [6, 16],
        "total_parameters": 114,
        "common_knot_policy": "Freeze continuous ORB knots for all four new cells; archived gapped knots differ.",
        "prior": {
            "sigma_g_halfnormal_scale": 10.0,
            "sigma_eta_halfnormal_scale": 0.5,
            "manuscript_consistency_issue": "Generic ANOVA description says 10 for eta; the LISA-specific value is .5 in the saved commands.",
        },
        "full_record_centering": "Same full annual reference-grid mean for all masks, never retained-site mean.",
        "transform": {
            "blocks": blocks,
            "complete_sites": len(centers),
            "gapped_sites": int(keep.sum()),
            "max_direct_relative_error": float(max(direct_error)),
            "setup_seconds": setup_seconds,
            "transform_seconds": transform_seconds,
            "estimated_annual_transform_seconds": transform_seconds
            * ((n - 2 * m) // (2 * m))
            / blocks,
            "masked_complete_coefficient_error_over_block_norm": max(
                mask_error
            ),
            "peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "wdm_plumbing": wdm_info,
        "reference": {
            "epoch_tcb": epoch,
            "approximation": "locally frozen physical response during each MP window; no within-window covariance claim",
            "projection": projection,
            "quadrature_max_relative_refinement_change": quadrature_relative,
            "maximum_factorization_relative_error": factor_error,
            "seconds": perf_counter() - ref_start,
        },
        "historical_wdm": full_reuse,
        "blockers": blockers,
        "jobs": {
            "smoke_inference": "not launched: known mean/target mismatch",
            "annual_mp_complete": "blocked",
            "annual_mp_gapped": "blocked",
            "annual_wdm_complete": "requires matched refit",
            "annual_wdm_gapped": "requires matched refit",
        },
        "sampling_profile": {
            "compile_seconds": None,
            "warmup_seconds": None,
            "sampling_seconds": None,
            "annual_fit_estimate_seconds": None,
        },
        "production_settings": {
            "chains": 2,
            "warmup": 2200,
            "draws": 4000,
            "target_accept": 0.99,
            "depth": 12,
            "wall_limit_hours": 6,
        },
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (OUT / "lisa_preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "LISA plumbing completed; inference blocked by", blockers, flush=True
    )


if __name__ == "__main__":
    main()
