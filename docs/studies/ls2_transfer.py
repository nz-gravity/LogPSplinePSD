"""Reproduce the small LS2 transfer check and render old/new posterior means.

Run: .venv/bin/python docs/studies/ls2_transfer.py (from repository root).
Uses the frozen source reference and 24 warmup / 16 retained draws per mode.
This checks numerical parity, not posterior convergence or calibrated recovery.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from log_psplines import (
    LogPSpline,
    PowerSpectrum,
    PowerSplineConfig,
    SplineBasis,
    fit,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=root / "tests/reference/ls2_wdm.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "tests/test_output/ls2_transfer"
    )
    options = parser.parse_args()
    metadata = json.loads(options.reference.with_suffix(".json").read_text())
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    rows = ["mode,max_relative_psd_difference,divergences,max_depth_hits"]
    with (
        jax.enable_x64(True),
        np.load(options.reference) as ref,
    ):
        data = PowerSpectrum(
            ref["power"],
            1,
            ref["frequency"],
            ref["time"],
            units="WDM coefficient variance",
        )
        spline = LogPSpline(
            frequency=SplineBasis.from_grid(
                data.frequency / data.frequency[-1], 4
            ),
            time=SplineBasis.from_grid(data.time, 4),
        )
        for centered in (False, True):
            mode = "centered" if centered else "noncentered"
            config = PowerSplineConfig(
                centered=centered,
                n_warmup=metadata["warmup"],
                n_samples=metadata["draws"],
                max_tree_depth=metadata["max_tree_depth"],
                progress_bar=False,
            )
            result = fit(data, config, model=spline)
            result.save(str(output / mode))
            old = ref[mode + "_psd"]
            new = result.psd
            delta = float(np.max(np.abs(new / old - 1)))
            np.testing.assert_allclose(new, old, rtol=3e-5, atol=3e-6)
            stats = result.idata["sample_stats"]
            div = int(stats["diverging"].sum())
            depth = int(
                (stats["num_steps"] >= 2**config.max_tree_depth - 1).sum()
            )
            rows.append(f"{mode},{delta:.16g},{div},{depth}")
            old_mean, new_mean = (
                np.mean(old, axis=(0, 1)),
                np.mean(new, axis=(0, 1)),
            )
            panels = [np.log(old_mean), np.log(new_mean)]
            lo, hi = min(p.min() for p in panels), max(p.max() for p in panels)
            fig, axes = plt.subplots(
                1, 2, figsize=(9, 3.5), sharey=True, layout="constrained"
            )
            for ax, field, title in zip(
                axes, panels, ["WDM source", "Shared spline port"], strict=True
            ):
                mesh = ax.pcolormesh(
                    data.time,
                    data.frequency,
                    field.T,
                    shading="auto",
                    vmin=lo,
                    vmax=hi,
                )
                ax.set(title=title, xlabel="Rescaled time")
            axes[0].set_ylabel("Frequency [Hz]")
            fig.colorbar(
                mesh, ax=axes, label="log mean WDM coefficient variance"
            )
            fig.suptitle(
                f"LS2 {mode}: {config.n_warmup} warmup / "
                f"{config.n_samples} draws (paired check)"
            )
            fig.savefig(output / f"{mode}_comparison.png", dpi=150)
            plt.close(fig)
    report = output / "comparison.csv"
    report.write_text("\n".join(rows) + "\n")
    print(report.read_text())
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
