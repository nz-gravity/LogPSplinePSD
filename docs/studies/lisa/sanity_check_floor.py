#!/usr/bin/env python3
"""Quick sanity check: eigenvalue preprocessing plots with/without Wishart floor.

Mirrors the real pipeline from run_lisa_psd_analysis.py:
  - Load TDI.h5 data (physical units, no pre-standardization)
  - Standardise → Wishart with Nb=52, tukey(0.1), fmin/fmax
  - Coarse-grain to Nc=256 (matching the preprocessing diagnostic config)
  - Produce eigenvalue separation plots

Usage:
    python sanity_check_floor.py [path/to/tdi.h5]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.diagnostics.preprocessing import (
    eigenvalue_separation_diagnostics,
    save_eigenvalue_separation_plot,
)
from log_psplines.preprocessing.coarse_grain import (
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)

OUTDIR = Path(__file__).parent / "sanity_check_output"
OUTDIR.mkdir(exist_ok=True)

# Match run_lisa_psd_analysis.py settings
FMIN = 1e-4
FMAX = 0.10809474686819218
NB = 52
NC = 256  # coarse bins for preprocessing diagnostic (matches the "good" plot)
WINDOW = ("tukey", 0.1)
FLOOR = 1e-6

# Locate tdi.h5
TDI_DEFAULT = Path("/Users/avi/Documents/projects/lisa_datagen/tdi.h5")
tdi_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TDI_DEFAULT
if not tdi_path.exists():
    print(f"ERROR: tdi.h5 not found at {tdi_path}")
    print("Usage: python sanity_check_floor.py [path/to/tdi.h5]")
    sys.exit(1)

# Load TDI data (same as run_lisa_psd_analysis.py)
import h5py

EDGE_TRIM = 500

with h5py.File(tdi_path, "r") as f:
    t = np.asarray(f["time"])
    X2 = np.asarray(f["X2"])
    Y2 = np.asarray(f["Y2"])
    Z2 = np.asarray(f["Z2"])

dt = float(t[1] - t[0])
fs = 1.0 / dt
sl = slice(EDGE_TRIM, -EDGE_TRIM)
data = np.vstack((X2[sl], Y2[sl], Z2[sl])).T
t_trim = t[sl]

# Trim for Wishart blocking
n_raw = data.shape[0]
n_trim = (n_raw // NB) * NB
data = data[:n_trim]
t_trim = t_trim[:n_trim]

ts = MultivariateTimeseries(
    y=data.astype(np.float64),
    t=t_trim.astype(np.float64),
)
std_ts = ts.standardise_for_psd()
print(
    f"Data: n={n_trim}, dt={dt}, fs={fs}, p={data.shape[1]}, Nb={NB}, Lb={n_trim // NB}"
)


def make_plot(label: str, floor_fraction=None) -> None:
    fft_full = std_ts.to_wishart_stats(
        Nb=NB,
        fmin=FMIN,
        fmax=FMAX,
        window=WINDOW,
        wishart_floor_fraction=floor_fraction,
    )
    Nl = fft_full.freq.shape[0]
    print(f"  Nl={Nl}, target Nc={NC}")

    # Coarse-grain to match the "good" plot config
    cg_spec = compute_binning_structure(fft_full.freq, Nc=NC)
    fft = apply_coarse_grain_multivar_fft(fft_full, cg_spec)
    Nh = Nl // NC
    print(f"  Coarse grid: Nc={NC}, Nh={Nh}, N_coarse={fft.freq.shape[0]}")

    diag = eigenvalue_separation_diagnostics(
        freq=np.asarray(fft.freq, dtype=float),
        matrix=np.asarray(fft.raw_psd),
    )

    slug = (
        "NO_floor" if floor_fraction is None else f"floor_{floor_fraction:.0e}"
    )
    out = str(OUTDIR / f"eigenvalue_{slug}.png")
    save_eigenvalue_separation_plot(
        diag,
        out,
        info_text=f"{label}  |  Nb={NB}, Nc={NC}, Nh={Nh}, window=tukey(0.1)",
        cholesky_matrix=np.asarray(fft.raw_psd),
    )
    print(f"  Saved: {out}")

    eig = diag.eigvals_desc
    print(f"  λ1 range: [{eig[:, 0].min():.3e}, {eig[:, 0].max():.3e}]")
    print(f"  λ3 range: [{eig[:, -1].min():.3e}, {eig[:, -1].max():.3e}]")
    for key, ratio in diag.ratios.items():
        r = ratio[np.isfinite(ratio)]
        print(
            f"  {key}: median={np.median(r):.4f}, max={r.max():.4f}, "
            f"frac>0.8={np.mean(r > 0.8):.3f}"
        )


print("\n--- Without floor ---")
make_plot("NO Wishart floor", floor_fraction=None)

print(f"\n--- With floor ({FLOOR:.0e} × median trace) ---")
make_plot(f"Wishart floor = {FLOOR:.0e} × median(trace)", floor_fraction=FLOOR)

print(f"\nDone. Plots saved to {OUTDIR}/")
