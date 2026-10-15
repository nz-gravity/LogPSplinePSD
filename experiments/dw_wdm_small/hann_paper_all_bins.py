"""All-positive-bin Hann LS2 comparison with the archived full WDM fits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from experiments.dw_wdm_small import hann_paper as s
from experiments.dw_wdm_small import hann_paper_ls2 as shared

from log_psplines import PowerData

ROOT = s.ROOT / "ls2_all_bins_20261015"
LENGTHS = tuple(2**k for k in range(10, 16))
RUNGS = tuple(range(1, 33))


def cells() -> list[dict]:
    return [
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
        for n in LENGTHS
        for r in range(5)
        for method in s.METHODS
    ]


def input_path(n: int, record: int) -> Path:
    return (
        s.ROOT / "inputs" / f"LS2_n{n}_r{record}.npz"
        if n in (1024, 32768)
        else shared.ROOT / "inputs" / f"n{n}_r{record}.npz"
    )


def observations(cell: dict) -> PowerData:
    x = np.load(input_path(cell["n"], cell["record"]))["x"]
    d = s.hann_observations(x, np.ones(len(x), bool), 0.1, rungs=RUNGS)
    return PowerData(d.power, d.counts, d.frequency / 5, d.time, d.units)


def freeze() -> dict:
    ROOT.mkdir(exist_ok=True)
    destination = ROOT / "protocol.json"
    if destination.exists():
        p = s.read(destination)
        assert p["rungs"] == list(RUNGS) and p["targets"] == cells()
        return p
    old = [
        c for c in s.read(s.ROOT / "inventory.json") if c["dataset"] == "LS2"
    ] + s.read(shared.ROOT / "inventory.json")
    recovered, checks, counts = {}, {}, []
    (ROOT / "bases").mkdir(exist_ok=True)
    for n in LENGTHS:
        model = s.endpoint_model(n)
        hashes = s.save_basis(ROOT / "bases" / f"n{n}.npz", model)
        for r in range(5):
            pair = {
                c["method"]: c for c in old if c["n"] == n and c["record"] == r
            }
            assert pair["MP"]["raw_sha256"] == pair["WDM"]["raw_sha256"]
            assert (
                hashes
                == pair["MP"]["model_hashes"]
                == pair["WDM"]["model_hashes"]
            )
            source = input_path(n, r)
            x = np.load(source)["x"]
            assert s.b.digest(x) == pair["MP"]["raw_sha256"]
            recovered[f"n{n}_r{r}"] = dict(
                raw_sha256=s.b.digest(x),
                model_hashes=hashes,
                raw_file=str(source),
                raw_file_sha256=s.audit.file_hash(source),
            )
            if r == 0:
                checks[str(n)] = shared.audit_observations(x, n, rungs=RUNGS)
                mp = observations(dict(n=n, record=r))
                wdm = np.load((s.LS2 / f"wdm_n{n}_r{r}").with_suffix(".npz"))
                assert set(np.rint(mp.frequency * 32.5).astype(int)) == set(
                    RUNGS
                )
                assert int(mp.counts.sum()) == n - 64
                assert (
                    int(wdm["observed_counts"].sum()) * 32
                    == int(mp.counts.sum()) * 31
                )
                counts.append(
                    dict(
                        n=n,
                        MP_complex_ordinates=int(mp.power.size),
                        MP_nominal_real_df=int(mp.counts.sum()),
                        WDM_real_coefficients=int(wdm["observed_power"].size),
                        WDM_nominal_real_df=int(wdm["observed_counts"].sum()),
                    )
                )
    protocol = dict(
        date="2026-10-15",
        authorization="Use all bins for MP and all bins for WDM for now; LS2 comparison.",
        targets=cells(),
        rungs=RUNGS,
        recovered_inputs=recovered,
        preflight=checks,
        counts=counts,
        observation="MP: symmetric Hann L65, energy24, m32, i2, all positive j1..32 on original complete cycles. WDM: archived Nf32, all interior j1..31 and all retained time rows; DC/Nyquist excluded.",
        prior="Identical between each MP/WDM pair: full[0,1] cubic basis and penalties; sigma_time,sigma_freq independent HalfNormal10; null_precision1e-4 ridge1e-6. Original time/frequency basis sizes retained at every N.",
        scoring="Original uniform96x48 physical grid, t[.05,.95], f[.25,4.75]Hz; linear PSD medians and central90 bounds; means +/- sample SD over five paired records.",
        sampler={
            str(n): [
                asdict(s.config(dict(dataset="LS2", n=n, record=0), attempt))
                for attempt in range(2)
            ]
            for n in LENGTHS
        },
        sampler_seeds="17000+r, independent-record seeds20261001+r",
        retry_policy="one fixed sampler extension; no target/configuration selection",
        limitations="All-bin Hann powers are correlated and can have nonproper complex marginals. Retaining all bins follows the requested comparison; nominal degrees of freedom do not imply independent information or calibrated posterior intervals.",
        source_sha256={
            str(Path(p).resolve()): s.audit.file_hash(p)
            for p in (
                Path(__file__),
                Path(s.__file__),
                Path(shared.__file__),
                Path("src/log_psplines/inference/power.py"),
            )
        },
        previous_protocol_sha256=s.audit.file_hash(s.ROOT / "protocol.json"),
        six_length_extension_protocol_sha256=s.audit.file_hash(
            shared.ROOT / "protocol.json"
        ),
    )
    destination.write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(frozen=True, targets=60, new_MP_fits=30, reused_WDM=30)
        ),
        flush=True,
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        protocol = freeze()
        shared.run(
            root=ROOT,
            target_cells=cells(),
            data_function=observations,
            protocol=protocol,
        )
        shared.summarize(
            root=ROOT,
            target_count=60,
            include_original=False,
            center_band_hz=(1 / 6.5, 32 / 6.5),
        )
    else:
        shared.summarize(
            root=ROOT,
            target_count=60,
            include_original=False,
            center_band_hz=(1 / 6.5, 32 / 6.5),
        )


if __name__ == "__main__":
    main()
