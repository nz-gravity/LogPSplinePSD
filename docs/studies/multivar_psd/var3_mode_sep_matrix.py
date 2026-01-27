"""Grid runner for VAR3 mode-separation diagnostics.

This script runs the VAR-based mode-separation datasets (separated / force_separation / bunched)
across a small grid of hyperparameters to diagnose whether the blocked sampler issues are driven by
data/likelihood geometry, prior geometry, or VI initialisation.

Defaults match the "concrete test matrix" discussed in the study notes:
  - cases: separated, force_separation, bunched
  - alpha_delta=beta_delta in {1e-4, 1, 2}
  - init_from_vi in {False, True diag, True lowrank, True flow:1}
  - n_time_blocks in {8, 16, 32}
  - n_knots in {5, 10, 15, 20}

Outputs are organised as:
  {out}/{seed_root}/case_{case}/B{blocks}/K{knots}/ad{alpha_delta}/init_{init_mode}/...

Each run writes `diagnostics_var3_mode_sep.json` and standard `run_mcmc` outputs. This script also
maintains an aggregate CSV (`var3_mode_sep_matrix.csv`) for easy comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from var3_mode_separation_study import StudyConfig, run_case


def _float_tag(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if 1e-3 <= abs_value < 1e3:
        return f"{value:g}"
    sci = f"{value:.0e}"  # e.g. 1e-04
    sci = sci.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return sci


def _sanitize_tag(value: str) -> str:
    # Keep alnum, dash, underscore; drop other separators like ':'.
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
    return "".join(out) if out else "unknown"


def _resolve_init(mode: str) -> tuple[bool, str | None, str]:
    mode = (mode or "").strip()
    key = mode.lower()
    if key in {"none", "novi", "no", "off", "false", "0"}:
        return False, None, "novi"
    if key == "diag":
        return True, "diag", "diag"
    if key == "mvn":
        return True, "mvn", "mvn"
    if key == "lowrank":
        return True, "lowrank", "lowrank"
    if key.startswith("lowrank:"):
        return True, mode, _sanitize_tag(key.replace(":", ""))
    if key.startswith("flow"):
        # e.g. flow:1
        tag = _sanitize_tag(key.replace(":", ""))
        return True, mode, tag
    # Fall back: treat as a raw vi_guide spec.
    return True, mode, _sanitize_tag(key.replace(":", ""))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _iter_runs(
    *,
    cases: Sequence[str],
    alpha_deltas: Sequence[float],
    init_modes: Sequence[str],
    time_blocks: Sequence[int],
    knots: Sequence[int],
) -> Iterable[dict[str, object]]:
    for case in cases:
        for n_time_blocks in time_blocks:
            for n_knots in knots:
                for alpha_delta in alpha_deltas:
                    for init_mode in init_modes:
                        yield {
                            "case": case,
                            "n_time_blocks": int(n_time_blocks),
                            "n_knots": int(n_knots),
                            "alpha_delta": float(alpha_delta),
                            "init_mode": str(init_mode),
                        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic grid for VAR3 mode separation."
    )
    parser.add_argument("--out", type=str, default="out_var3_mode_sep_matrix")
    parser.add_argument("--n-time", type=int, default=1024)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--vi-steps", type=int, default=8000)
    parser.add_argument("--vi-lr", type=float, default=1e-2)
    parser.add_argument(
        "--full-outputs",
        action="store_true",
        help="Enable plots/coherence/preprocessing outputs (slower).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["separated", "force_separation", "bunched"],
    )
    parser.add_argument(
        "--alpha-delta-grid",
        nargs="+",
        type=float,
        default=[1e-4, 1.0, 2.0],
    )
    parser.add_argument(
        "--init-grid",
        nargs="+",
        default=["novi", "diag", "lowrank", "flow:1"],
        help="Init modes: novi | diag | lowrank | flow:1 (or any vi_guide string).",
    )
    parser.add_argument(
        "--time-blocks-grid", nargs="+", type=int, default=[8, 16, 32]
    )
    parser.add_argument("--knots-grid", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--resume", action="store_true", help="Skip runs with existing diagnostics JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if diagnostics JSON exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing JSON diagnostics into the aggregate CSV.")
    parser.add_argument("--start", type=int, default=0, help="Start index into the run list (0-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of runs to execute/collect.")
    args = parser.parse_args()

    cases = [str(c) for c in args.cases]
    alpha_deltas = [float(x) for x in args.alpha_delta_grid]
    init_modes = [str(x) for x in args.init_grid]
    time_blocks = [int(x) for x in args.time_blocks_grid]
    knots = [int(x) for x in args.knots_grid]

    n_time = int(args.n_time)
    for n_blocks in time_blocks:
        if n_blocks < 3:
            raise ValueError("n_time_blocks must be >= 3 (n_channels=3).")
        if n_time % int(n_blocks) != 0:
            raise ValueError(
                f"n_time={n_time} must be divisible by n_time_blocks={n_blocks}."
            )

    here = Path(__file__).resolve().parent
    root_out = here / str(args.out) / (
        f"seed_{int(args.seed)}_N{n_time}_S{int(args.samples)}_W{int(args.warmup)}_C{int(args.chains)}"
    )
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "matrix_config.json").write_text(
        json.dumps(
            dict(
                base=dict(
                    n_time=int(args.n_time),
                    fs=float(args.fs),
                    seed=int(args.seed),
                    n_samples=int(args.samples),
                    n_warmup=int(args.warmup),
                    num_chains=int(args.chains),
                    target_accept_prob=float(args.target_accept),
                    max_tree_depth=int(args.max_tree_depth),
                    vi_steps=int(args.vi_steps),
                    vi_lr=float(args.vi_lr),
                    full_outputs=bool(args.full_outputs),
                ),
                grid=dict(
                    cases=cases,
                    alpha_delta=alpha_deltas,
                    init_modes=init_modes,
                    n_time_blocks=time_blocks,
                    n_knots=knots,
                ),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    all_runs = list(
        _iter_runs(
            cases=cases,
            alpha_deltas=alpha_deltas,
            init_modes=init_modes,
            time_blocks=time_blocks,
            knots=knots,
        )
    )

    if args.start < 0 or args.start > len(all_runs):
        raise ValueError(f"--start must be in [0, {len(all_runs)}], got {args.start}.")
    end = len(all_runs) if args.limit is None else min(len(all_runs), args.start + int(args.limit))
    runs = all_runs[int(args.start) : end]

    aggregate_path = root_out / "var3_mode_sep_matrix.csv"

    if args.dry_run:
        print(f"Planned runs: {len(runs)} (of total {len(all_runs)})")
        for spec in runs[:20]:
            init_from_vi, vi_guide, init_tag = _resolve_init(str(spec["init_mode"]))
            ad_tag = _float_tag(float(spec["alpha_delta"]))
            run_dir = (
                root_out
                / f"case_{spec['case']}"
                / f"B{spec['n_time_blocks']}"
                / f"K{spec['n_knots']}"
                / f"ad{ad_tag}"
                / f"init_{init_tag}"
            )
            print(f"- {run_dir} (init_from_vi={init_from_vi}, vi_guide={vi_guide})")
        return

    collected_rows: list[dict[str, object]] = []

    for spec in runs:
        init_from_vi, vi_guide, init_tag = _resolve_init(str(spec["init_mode"]))
        ad = float(spec["alpha_delta"])
        ad_tag = _float_tag(ad)
        run_dir = (
            root_out
            / f"case_{spec['case']}"
            / f"B{spec['n_time_blocks']}"
            / f"K{spec['n_knots']}"
            / f"ad{ad_tag}"
            / f"init_{init_tag}"
        )
        diag_path = run_dir / "diagnostics_var3_mode_sep.json"

        if diag_path.exists() and not args.overwrite:
            with diag_path.open("r") as handle:
                collected_rows.append(json.load(handle))
            if not args.collect_only:
                if not args.resume:
                    raise RuntimeError(
                        f"Found existing diagnostics at {diag_path} (use --resume or --overwrite)."
                    )
            continue

        if args.collect_only:
            continue

        base_cfg = StudyConfig(
            n_time=int(args.n_time),
            fs=float(args.fs),
            seed=int(args.seed),
            n_knots=int(spec["n_knots"]),
            n_samples=int(args.samples),
            n_warmup=int(args.warmup),
            num_chains=int(args.chains),
            n_time_blocks=int(spec["n_time_blocks"]),
            target_accept_prob=float(args.target_accept),
            max_tree_depth=int(args.max_tree_depth),
            alpha_delta=ad,
            beta_delta=ad,
            init_from_vi=init_from_vi,
            vi_guide=vi_guide,
            vi_steps=int(args.vi_steps),
            vi_lr=float(args.vi_lr),
        )

        row = run_case(
            cfg=base_cfg,
            outdir=run_dir,
            case=str(spec["case"]),
            skip_plot_diagnostics=not bool(args.full_outputs),
            diagnostics_summary_mode=("light" if args.full_outputs else "off"),
            diagnostics_summary_position="end",
            save_preprocessing_plots=bool(args.full_outputs),
            compute_coherence_quantiles=bool(args.full_outputs),
        )
        collected_rows.append(dict(row))
        _write_csv(aggregate_path, collected_rows)

    if collected_rows:
        _write_csv(aggregate_path, collected_rows)
        print(f"Wrote aggregate CSV to {aggregate_path}")
    else:
        print("No results collected (no existing JSONs and no runs executed).")


if __name__ == "__main__":
    main()
