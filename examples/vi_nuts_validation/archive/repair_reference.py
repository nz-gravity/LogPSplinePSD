"""One explicit NUTS tuning repair; retain original references and comparisons."""

import argparse
import json
import shutil
import tomllib
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import xarray as xr
from run import (
    DEFAULT_PREVIOUS,
    build_target,
    features,
    mmd_comparison,
    packed_reference,
    posterior_geometry,
    run_reference,
    write_json,
)

from log_psplines.diagnostics.variational import (
    VIDiagnosticState,
    compare_features,
    rebuild_guide,
    runtime_provenance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    args = parser.parse_args()
    cfg = tomllib.loads(Path(__file__).with_name("config.toml").read_text())
    # Explicit separate tuning configuration, without changing the target.
    cfg.update(
        nuts_repair_target_accept=0.999,
        nuts_repair_warmup=2000,
        nuts_repair_draws=4000,
        max_tree_depth=12,
        nuts_seed=6103,
    )
    directory = args.out / args.target
    repair = directory / "repair_2"
    if repair.exists():
        raise ValueError("refusing to overwrite previous repair")
    repair.mkdir()
    started = perf_counter()
    write_json(
        repair / "manifest.json",
        {
            "config": cfg,
            "provenance": runtime_provenance(),
            "status": "running",
        },
    )
    target = build_target(args.target, cfg, args.previous, repair)
    identity = json.loads((directory / "complete.json").read_text())[
        "target_fingerprint"
    ]
    if target.identity != identity:
        raise ValueError("target fingerprint changed")
    result, p, names, reference = run_reference(
        target, cfg, repair, repair=True
    )
    print(
        args.target,
        reference["status"],
        reference["health"]["divergences"],
        min(reference["health"]["bfmi"]),
        flush=True,
    )
    if reference["status"] != "accepted_screen":
        write_json(
            repair / "complete.json",
            {
                "status": "failed_reference",
                "wall_seconds": perf_counter() - started,
            },
        )
        return
    events = []
    for index, name in enumerate(names):
        if name.startswith("D_"):
            events.extend(
                (name + f"_below{epsilon}", index, epsilon)
                for epsilon in (0.005, 0.02, 0.08)
            )
        elif name == "time_contrast_logV_f.25":
            events.append(("positive_time_contrast", index, 0.0))
    for path in sorted(directory.glob("*_seed*/comparison.json")):
        job = path.parent
        shutil.copy2(path, job / "comparison_reference_repair.json")
        record = json.loads(path.read_text())
        state = VIDiagnosticState.load(job)
        guide = rebuild_guide(state, target.model, target_fingerprint=identity)
        pr = packed_reference(guide, result.posterior)
        checks = []
        for old in record["checkpoints"]:
            before = perf_counter()
            step = old["actual_steps"]
            params = state.checkpoints[str(step)]
            u = guide.get_posterior(params).sample(
                jax.random.PRNGKey(8301), (cfg["posterior_draws"],)
            )
            values = guide._unpack_and_constrain(u, params)
            q_dataset = xr.Dataset(
                {
                    key: xr.DataArray(
                        np.asarray(x)[None],
                        dims=(
                            "chain",
                            "draw",
                            *[f"{key}_dim{i}" for i in range(np.ndim(x) - 1)],
                        ),
                    )
                    for key, x in values.items()
                    if not key.startswith("log_likelihood")
                }
            )
            q, _ = features(target, q_dataset)
            checks.append(
                {
                    "actual_steps": step,
                    "comparison": compare_features(
                        q,
                        p,
                        names=names,
                        vi_fingerprint=identity,
                        reference_fingerprint=identity,
                        reference_status=reference["status"],
                        events=events,
                    ),
                    "functional_mmd": mmd_comparison(q, p, cfg),
                    "latent_mmd": mmd_comparison(np.asarray(u)[None], pr, cfg),
                    "geometry": posterior_geometry(guide, params, pr),
                    "checkpoint_comparison_seconds": perf_counter() - before,
                }
            )
        record.update(
            checkpoints=checks,
            selected_reference="repair_2/reference_repair",
            reference_refresh_provenance=runtime_provenance(),
        )
        write_json(path, record)
    write_json(
        directory / "reference_selection.json",
        {
            "prefix": "repair_2/reference_repair",
            "status": reference["status"],
            "target_fingerprint": identity,
        },
    )
    write_json(
        repair / "complete.json",
        {
            "status": "accepted_screen",
            "wall_seconds": perf_counter() - started,
            "provenance": runtime_provenance(),
        },
    )


if __name__ == "__main__":
    main()
