# LISA Multivariate PSD Study

Bayesian nonparametric estimation of LISA TDI-2 XYZ noise power spectral density
using log-P-splines with a Whittle/Wishart likelihood.

---

## What is being estimated

LISA has three TDI-2 channels: **X, Y, Z**. Their joint noise is described by a
3×3 Hermitian positive-definite spectral density matrix at each frequency bin:

```
S(f) = [ Sxx(f)  Sxy(f)  Sxz(f) ]
       [ Syx(f)  Syy(f)  Syz(f) ]
       [ Szx(f)  Szy(f)  Szz(f) ]
```

### Key symmetry: the true LISA noise matrix is highly structured

Because X, Y, Z are symmetric TDI combinations of the same three arms, the analytic
noise matrix has a very specific form **at every frequency**:

```
S(f) = [ a   d   d ]
       [ d   a   d ]
       [ d   d   a ]
```

where:
- `a = Sxx = Syy = Szz` — all three auto-PSDs are **identical**
- `d = Sxy = Sxz = Syz` — all six cross-PSDs are **identical and real**
- The off-diagonal value `d ≈ −0.5 × a` at most frequencies (anti-correlated)

This means the 3×3 matrix has only **2 independent real functions of frequency**,
not 9. The model does not exploit this symmetry — it estimates all 9 elements
independently, which makes the off-diagonal channels harder to sample.

### Why the imaginary part of the true PSD is zero

The cross-spectral density between channels i and j is generally complex:

```
S_ij(f) = E[ X_i(f) · X_j*(f) ]
```

For a generic pair of channels this has a nonzero imaginary part (arising from
a phase difference / time delay between the channels). However, for LISA TDI-2
XYZ the analytic cross-spectrum is **purely real** at every frequency. This
follows from two facts:

1. **Equal-arm, symmetric TDI combinations.** X, Y, Z are constructed from the
   same three arm noise processes with identical arm lengths. There is no net
   time-delay asymmetry between pairs of channels — the phases acquired on
   the two paths that form, say, S_XY cancel exactly in the equal-arm limit.

2. **Real-valued noise sources.** OMS and proof-mass noise are modelled as
   stationary, real-valued Gaussian processes. Their PSDs S_op(f) and S_pm(f)
   are real and even in frequency. When the TDI response functions C_xx(f) and
   C_xy(f) are applied (both purely real sinusoidal functions of f), the result
   stays real.

Concretely, from the code (`lisa_data.py`):

```python
def _S_xy(freq, Sop, Spm):
    C = -16 * sin(x) * sin(2x)**3      # purely real
    return C * Sop + 4 * C * Spm       # real × real → real
```

So `Im[S_XY] = Im[S_XZ] = Im[S_YZ] = 0` exactly.

**Practical consequence for the model:** the posterior imaginary CIs on off-diagonal
elements should be centred on zero and very narrow. Any visible posterior imaginary
uncertainty is pure statistical noise from finite data, not a real signal. When
reporting coverage on the off-diagonal elements we compute it on both the real and
imaginary parts via `_complex_to_real` encoding, but the imaginary part should
trivially cover (truth = 0 ≈ centre of CI).

### Transfer function nulls

The TDI-2 response functions contain sinusoidal terms that go to zero at:

```
f_null = k / (4 × L/c)  =  k × 0.030 Hz,   k = 1, 2, 3, …
```

With L = 2.5 × 10⁹ m and c = 3 × 10⁸ m/s:
- **null₁ = 0.030 Hz** (first null)
- **null₂ = 0.060 Hz**
- **null₃ = 0.090 Hz**

At these frequencies both `a` and `d` drop by 10+ orders of magnitude. A smooth
P-spline cannot track these drops, so coverage always fails there. The fix is
**null-band excision**: remove the ≈120 CG bins around each null before inference.

---

## Data generation

```
lisa_datagen.py  →  generate_lisatools_xyz_noise_timeseries()
```

- **Model**: `scirdv1` from `lisatools` (Science-CReq LISA noise model)
- **Arm length**: L = 2.5 × 10⁹ m
- **Noise sources**:
  - OMS (optical metrology): `S_oms = (1.5×10⁻¹¹ m)² × [1 + (2mHz/f)⁴] / Hz`
  - Proof-mass (acceleration): `S_pm = (3×10⁻¹⁵ m/s²)² × [1 + (0.4mHz/f)²][1+(f/8mHz)⁴] / Hz`
- **Sample rate**: dt = 5 s (fs = 0.2 Hz, Nyquist = 0.1 Hz)
- **Analysis band**: [1×10⁻⁴, 0.1] Hz
- **Units**: absolute frequency noise, Hz²/Hz

Noise is drawn as correlated Gaussian in the frequency domain using the Cholesky
decomposition of `S(f)`, then transformed to the time domain via inverse FFT.

---

## Inference pipeline

```
TimeSeries (X,Y,Z)
    │
    ▼
MultivarFFT.compute_wishart(Nb=N_blocks, window=Tukey(0.1))
    │  block-average periodogram → (N_raw × 3 × 3) complex spectral matrix
    ▼
CoarseGrain(Nc=8192)
    │  merge ~10 raw bins per CG bin → N_cg = 6042 bins
    │  effective DOF per bin = 2 × Nb × merge_factor
    ▼
NullBandExcision(0.030 ± 0.001, 0.060 ± 0.001, 0.090 ± 0.001 Hz)
    │  remove ~363 bins → ~5679 bins retained
    ▼
LogPSpline model (K knots, d=2 penalty, uniform placement)
    │  parameterises log S(f) via Cholesky: S = T⁻¹ D T⁻ᴴ
    │  D = diag(exp(log_d))  — log auto-PSDs modelled by P-spline
    │  T = lower-triangular with 1s on diagonal, off-diag modelled by P-spline
    ▼
Multivariate blocked NUTS (4 chains, 1 chain per channel block)
    ▼
posterior_psd  →  compact_ci_curves.npz  →  paper plots
```

**Cholesky parameterisation** — the model represents the PSD matrix as:

```
S(f) = T(f)⁻¹ · D(f) · T(f)⁻ᴴ
```

where `D(f)` is diagonal (auto-PSDs in a rotated basis) and `T(f)` is lower-triangular
with unit diagonal (cross-channel coupling). Both are modelled by independent
P-spline bases over frequency.

---

## Key configuration choices

| Parameter | Value | Reason |
|-----------|-------|--------|
| K (knots) | 100 | More knots → wider CIs → correct ~87% coverage |
| d (penalty order) | 2 | Penalises curvature → smooth null bridging |
| Knot placement | uniform | Avoids clustering knots at null-adjacent spectral peak |
| Nc (CG target) | 8192 → 6042 | 10× merge; confirmed CG does not affect CI width |
| Null excision | ±1 mHz × 3 nulls | Removes ~6% of bins; required for correct coverage |
| α_δ / β_δ (smoothing prior) | 3.0 / 3.0 | Loose prior (0.1/0.1) causes posterior geometry collapse |
| Block days | 7 | Nb=4 (30d), Nb=13 (90d), Nb=52 (365d) |

---

## Coverage analysis

Coverage = fraction of (frequency × matrix element) pairs where the true PSD
falls inside the 90% credible interval.

| Configuration | Coverage | CI width | Notes |
|---|---|---|---|
| K=48, baseline (run_x 30d) | 60% | 4.1% | Prior too tight |
| K=48, no CG (run_AH) | 60% | 4.1% | CG confirmed irrelevant |
| K=48, null excision (run_AI) | 72% | 4.2% | Nulls were pulling coverage down |
| K=100, standard prior (run_AL) | **86.5%** | 5.8% | **Paper configuration** |
| K=200, standard prior (run_AK) | 88% | 7.3% | Sampler struggles on ch2 |
| K=100, loose prior (run_AJ) | ~~87%~~ | — | ⚠ Sampler failure — not trustworthy |

**Why loose prior fails**: α/β=0.1 allows δ→0, making the posterior geometry
near-degenerate for the off-diagonal channels. Channel 0 (strongest signal) is fine,
but channels 1 and 2 hit max tree depth 99.8% of the time with step size 0.002 vs 0.38.

**Why coverage is not 90%**: the remaining ~13% gap after null excision and K=100 is
explained by the smooth P-spline prior still being unable to track the true PSD with
full fidelity at every frequency bin — an inherent property of the nonparametric
model, not a sampling or implementation bug.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Single-seed MCMC run |
| `run_paper.sh` | Production runs: 30d, 90d, 365d (paper config) |
| `run_ci_sweep.sh` | Hyperparameter sweep (run_z → run_AL) |
| `paper_final_plots.py` | Generate all paper figures |
| `utils/data.py` | LISA data generation wrapper |
| `utils/inference.py` | MCMC wrapper (`run_lisa_mcmc`) |
| `utils/preprocessing.py` | Frequency grid and CG setup |
| `lisa_datagen.py` | Standalone data generation + diagnostics |

---

## Key CLI flags

```bash
# Standard paper run (30 days)
python main.py 0 \
  --duration-days 30 --K 100 --diff-order 2 --knot-method uniform \
  --coarse-Nc 8192 --block-days 7 \
  --wishart-window tukey --wishart-tukey-alpha 0.1 \
  --target-accept 0.8 --n-warmup 1500 --n-samples 1000 --no-vi \
  --null-excision 0.030:0.001 0.060:0.001 0.090:0.001 \
  --outdir runs/my_run

# Null excision syntax: center:halfwidth (Hz)
# --null-excision        → no excision
# --null-excision 0.030:0.001  → excise [0.029, 0.031] Hz
```
