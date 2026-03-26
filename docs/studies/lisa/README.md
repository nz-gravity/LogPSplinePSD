# LISA TDI-XYZ PSD Study: Run Log

Multivariate P-spline PSD estimation on synthetic LISA TDI-2 XYZ noise.
Entry point: `main.py`. All runs use seed 0 unless noted.

---

## Known Issues

### 1. Low-frequency bias in the empirical diagnostic plot

`utils/plotting.py` computes a raw periodogram via `np.fft.rfft(y_full)` with
**no detrending and no taper**. Any residual DC offset leaks into the lowest
frequency bins (rectangular-window sidelobes), producing an apparent upward
bias below ~3×10⁻⁴ Hz that is purely an artifact of the diagnostic, not the
inference.

The Welch overlay is also marginal at low frequencies: `nperseg = fs/fmin`
gives exactly **one frequency bin at fmin**, so the lowest bins are extremely
noisy.

Fix: subtract the mean before the full-data FFT and/or double `nperseg`.

### 2. Oscillations at 0.06 Hz (transfer-null ringing)

The TDI-2 XYZ spectral matrix has deterministic zeros at
`f_k = k / (4 L/c)` (L/c ≈ 8.33 s):

| k | f (Hz) | Mechanism      |
|---|--------|----------------|
| 1 | 0.030  | sin(x) = 0     |
| 2 | 0.060  | sin(2x) = 0    |
| 3 | 0.090  | sin(x) = 0     |

At these frequencies the Wishart matrix is nearly rank-deficient, the Cholesky
factorization becomes numerically unstable, and the Welch estimate rings due to
sidelobe leakage from adjacent bins into the near-zero trough.

Fixes: `--wishart-floor-fraction 1e-6` (eigenvalue regularization) and/or
`--exclude-transfer-nulls` (excise null bins from the likelihood).

---

## Run Inventory

Runs are listed from oldest (exploratory) to newest (most current).

---

### Legacy runs (`results/`)

Old results from a previous codebase version. API and slug format are
incompatible with the current `main.py`. Kept for reference only.

Key variants explored:
- `nb14 / nb52 / nb365` — different numbers of Wishart blocks
- `cgNc1079 / cgNc2014 / cgOff` — coarse-graining experiments
- `f1e-04-6e-02` — upper frequency limit cut at 0.06 Hz to avoid the null
- `k10 / k20 / k24 / k32` — knot counts
- `d1 / d2` — penalty difference order
- `kmlog / kmdensity` — knot placement method

These runs used fractional-frequency units (no absolute-freq scaling),
so PSD values are in the ~10⁻³⁷ range. Not directly comparable to current runs.

---

### run A — sanity check: floor regularization effect (`sanity_check_output/`)

**Script:** `sanity_check_floor.py`
**Purpose:** Verify that Wishart eigenvalue flooring stabilises the Cholesky
at transfer nulls without distorting ordinary bins.
**Output:** Eigenvalue-ratio plots with and without floor.
**Result:** Floor at 1×10⁻⁶ × median(trace) cleanly clips the null-frequency
collapse while leaving all other bins unchanged. Confirms the fix is safe.

---

### run B — smoke test baseline vs excised (`smoke_out/`)

**Script:** manual smoke run
**Subdirs:** `baseline/`, `excised/`
**Purpose:** Quick end-to-end check that the null-excision path runs without
errors.
**Note:** Incomplete outputs; not a full inference run.

---

### run C — 7-day Hann, no coarse grain, no detrend (`results_lisa_7day_hann_nocoarse_nodetrend/`)

**Command:**
```bash
python main.py 0 \
  --duration-days 365 --block-days 7 \
  --wishart-window hann --wishart-detrend none \
  --coarse-Nc 0 --K 20 --knot-method density --diff-order 2 \
  --vi
```

**Intent:** Test effect of disabling per-block detrending (mean subtraction)
with a Hann taper.

| Metric           | Value  |
|------------------|--------|
| rhat_max         | 1.140  |
| ess_median       | 5410   |
| n_divergences    | 0      |
| riae_matrix      | 0.269  |
| riae_diag_mean   | 0.258  |
| coherence_riae   | 0.224  |
| coverage (90%)   | 0.239  |
| runtime          | 817 s  |

**Notes:** rhat borderline (1.14). Coverage badly low (0.24 vs expected 0.90),
indicating the posterior is severely overconfident. No-detrend + Hann does not
help with the low-frequency bias.

---

### run D — 16384-sample blocks, Hann, no coarse grain (`results_lisa_16384_hann_nocoarse/`)

**Command:**
```bash
python main.py 0 \
  --duration-days 365 --block-days 0.948148 \
  --wishart-window hann --wishart-detrend constant \
  --coarse-Nc 0 --K 20 --knot-method density --diff-order 2 \
  --vi
```

**Intent:** Very short blocks (16384 samples ≈ 0.95 days) to test whether
more Wishart averages reduces transfer-null instability. Nb ≈ 385 blocks.

| Metric           | Value  |
|------------------|--------|
| rhat_max         | 1.016  |
| ess_median       | 3205   |
| n_divergences    | 0      |
| riae_matrix      | 0.264  |
| riae_diag_mean   | 0.252  |
| coherence_riae   | 0.218  |
| coverage (90%)   | 0.251  |
| runtime          | 176 s  |

**Notes:** Good convergence. Coverage still low (0.25). Very short blocks
reduce frequency resolution (df = 1/block_duration is large), which may
hurt the spline fit at low frequencies.

---

### run E — 365-day, Tukey(0.1), coarse Nc=8192, with VI (`results_datagen_match/`)

**Command:**
```bash
python main.py 0 \
  --duration-days 365 --block-days 7 \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --wishart-detrend constant \
  --coarse-Nc 8192 --K 20 --knot-method density --diff-order 2 \
  --vi
```

**Intent:** Full-length run with the standard Tukey taper and coarse graining,
matching the data-generation configuration. Baseline for the corrected pipeline.

| Metric           | Value  |
|------------------|--------|
| rhat_max         | 1.018  |
| ess_median       | 4188   |
| n_divergences    | 0      |
| riae_matrix      | 0.276  |
| riae_diag_mean   | 0.256  |
| coherence_riae   | 0.270  |
| coverage (90%)   | 0.303  |
| runtime          | 241 s  |

**Notes:** Best convergence so far. Coverage is still low (0.30), which is
the main open issue. Transfer-null oscillations still present in the empirical
diagnostic plot (no floor, no excision).

---

### run F — 7-day diagnostic, no MCMC, step 1 of debug plan (`out_diagnostic/`)

**Command:**
```bash
python main.py 0 \
  --outdir out_diagnostic \
  --duration-days 7 \
  --K 10 --coarse-Nc 0 --no-vi \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --wishart-detrend constant
```

**Intent:** Quick single-seed run to inspect `preprocessing_psd_matrix.png`
and verify the empirical low-frequency bias. Short data so MCMC is fast.

| Metric           | Value  |
|------------------|--------|
| rhat_max         | 1.106  |
| ess_median       | 390    |
| n_divergences    | 0      |
| riae_matrix      | 0.448  |
| riae_diag_mean   | 0.403  |
| coherence_riae   | 0.289  |
| coverage (90%)   | 0.329  |
| runtime          | 7518 s |

**Notes:** 7-day data → poor accuracy and low ESS expected (little data).
rhat borderline. Low-frequency bias still visible in preprocessing plot;
transfer-null oscillations still present.

---

### run G — 365-day, null excision + VI, rect window (`results_transfer_null_excised/`)

**Command:**
```bash
python main.py 0 \
  --duration-days 365 --block-days 7 \
  --wishart-window none --wishart-detrend constant \
  --coarse-Nc 2014 --K 20 --knot-method density --diff-order 2 \
  --vi \
  --exclude-transfer-nulls --exclude-bins-per-side 1
```

**Intent:** Test null excision with the rectangular window and VI warm-start.

| Metric           | Value      |
|------------------|------------|
| rhat_max         | **12.29**  |
| ess_median       | **4.5**    |
| n_divergences    | 0          |
| riae_matrix      | **140 756**|
| coverage (90%)   | 0.001      |
| runtime          | 462 s      |

**Notes:** ⚠️ **Catastrophically bad.** The PSD values (ciw_psd_diag ~10⁻³⁵)
indicate the run used fractional-frequency units internally (unit mismatch
between the data and the analytic truth). The MCMC failed completely.
Root cause: likely `--no-absolute-freq-units` was inadvertently active, or
the unit scaling changed between runs. This result should be discarded.

---

## Current Status

Best working run: **run E** (`results_datagen_match/`).
- Convergence: good (rhat=1.018, ess=4188, no divergences)
- Accuracy: moderate (riae≈0.28)
- Coverage: **poor** (0.30 vs expected 0.90) — main open problem
- Transfer-null oscillations: still present in diagnostic plot
- Low-frequency empirical bias: still present in diagnostic plot

---

## Planned Next Runs

Use the `runs/` directory going forward with short human-readable labels.

### run H — floor regularization + null excision (`runs/run_h_floor_excise/`)

Add Wishart floor + null excision to run E configuration:

```bash
python main.py 0 \
  --outdir runs/run_h_floor_excise \
  --duration-days 365 --block-days 7 \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --wishart-detrend constant \
  --wishart-floor-fraction 1e-6 \
  --exclude-transfer-nulls --exclude-bins-per-side 3 \
  --coarse-Nc 8192 --K 20 --knot-method density --diff-order 2 \
  --vi
```

**Goal:** Eliminate the 0.06 Hz oscillations from both the diagnostic and the
inference. Compare riae and coverage to run E.

---

### run I — coverage investigation: more knots (`runs/run_i_k32/`)

The low coverage (0.30) suggests the posterior is overconfident or the spline
has insufficient flexibility. Try more knots:

```bash
python main.py 0 \
  --outdir runs/run_i_k32 \
  --duration-days 365 --block-days 7 \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --wishart-detrend constant \
  --wishart-floor-fraction 1e-6 \
  --exclude-transfer-nulls --exclude-bins-per-side 3 \
  --coarse-Nc 8192 --K 32 --knot-method density --diff-order 2 \
  --vi
```

---

### run J — coverage investigation: looser smoothing prior (`runs/run_j_loose_prior/`)

The smoothness prior (alpha=3, beta=3) may be over-shrinking the posterior.
Try a flatter prior:

```bash
python main.py 0 \
  --outdir runs/run_j_loose_prior \
  --duration-days 365 --block-days 7 \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --wishart-detrend constant \
  --wishart-floor-fraction 1e-6 \
  --exclude-transfer-nulls --exclude-bins-per-side 3 \
  --coarse-Nc 8192 --K 20 --knot-method density --diff-order 2 \
  --alpha-delta 1.0 --beta-delta 1.0 \
  --vi
```

---

### run K — multi-seed validation (`runs/run_k_multiseed/`)

Once a good configuration is identified, validate coverage across seeds:

```bash
for seed in 0 1 2 3 4; do
  python main.py $seed \
    --outdir runs/run_k_multiseed \
    --duration-days 365 --block-days 7 \
    --wishart-window tukey --wishart-tukey-alpha 0.1 \
    --wishart-detrend constant \
    --wishart-floor-fraction 1e-6 \
    --exclude-transfer-nulls --exclude-bins-per-side 3 \
    --coarse-Nc 8192 --K 20 --knot-method density --diff-order 2 \
    --vi &
done
```

---

## Directory Reference

```
lisa/
├── README.md                        # this file
├── main.py                          # single-seed entry point
├── collect_results.py               # aggregate metrics across seeds
├── utils/                           # shared helpers (data, inference, plotting…)
├── data/tdi.h5                      # cached 365-day XYZ timeseries (seed 0)
│
├── results/                         # legacy runs (old codebase, frac-freq units)
├── sanity_check_output/             # run A: eigenvalue floor sanity check
├── smoke_out/                       # run B: baseline vs excised smoke test
├── results_lisa_7day_hann_nocoarse_nodetrend/   # run C
├── results_lisa_16384_hann_nocoarse/            # run D
├── results_datagen_match/                       # run E ← best so far
├── out_diagnostic/                              # run F
├── results_transfer_null_excised/               # run G ← unit-mismatch failure
│
└── runs/                            # new structured runs going forward
    ├── run_h_floor_excise/          # planned
    ├── run_i_k32/                   # planned
    ├── run_j_loose_prior/           # planned
    └── run_k_multiseed/             # planned (after good config found)
```
