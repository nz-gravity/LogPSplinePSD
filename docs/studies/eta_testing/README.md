# eta-tempering study

Investigates the generalized-Bayes / Safe-Bayes correction for the
multivariate Whittle pseudo-likelihood.  The Whittle likelihood treats
each coarse-grained frequency bin as carrying `Nb * Nh` independent
Wishart replications, which over-concentrates the posterior when the
smooth P-spline model has far fewer effective parameters.

## Validated formula

```
eta = min(1, c / (Nb * Nh))     # c = 2 default
```

Implemented in `MultivarBlockedNUTSConfig.eta` / `eta_c`.

## Key findings (VAR(2) simulation, seeds 0-99)

| c     | Coverage (90% CI) | Notes                          |
|-------|-------------------|--------------------------------|
| 0.5   | > 0.90            | Over-covers, CIs too wide      |
| 1     | ~ 0.85-0.90       | Slightly under nominal         |
| **2** | **~ 0.90**        | **Best default**               |
| 4     | ~ 0.80            | Starts to under-cover          |
| 8     | ~ 0.60            | Too aggressive                 |

- `n_basis` is **not** a first-order term in the scaling rule.
- The dominant over-concentration scales with `Nb * Nh` (data replication
  factor), not the model dimension.
- eta shifts the posterior median toward the prior, not just the variance.
  Coverage, CI width, and RIAE are all coupled through eta.

## Validation results (`eta_validation_study.py`)

### Test 3: K-sweep (n_basis independence)

Coverage is stable at ~0.91-0.98 across K={10,15,20,30,40} with c=2 fixed.
Confirms n_basis is not a first-order term.

| K  | n_basis | Coverage      | RIAE   | Divergences |
|----|---------|---------------|--------|-------------|
| 10 | 11      | 0.906 ± 0.046 | 0.115  | 601         |
| 15 | 16      | 0.930 ± 0.019 | 0.113  | 79          |
| 20 | 21      | 0.939 ± 0.015 | 0.111  | 45          |
| 30 | 31      | 0.976 ± 0.026 | 0.111  | 69          |
| 40 | 41      | 0.968 ± 0.028 | 0.111  | 69          |

### Test 4: Analytic η candidates

Trace-based formulas (edf/n_basis) give c ≈ 0.75-0.94, which produces
similar overall coverage to c=2 but with more divergences and lower
off-diagonal Im coverage.  No analytic formula tested recovers c=2.

| Candidate    | η     | Coverage      | Divergences |
|--------------|-------|---------------|-------------|
| empirical c2 | 0.250 | 0.939 ± 0.015 | 45          |
| edf (φ=1)    | 0.118 | 0.943 ± 0.038 | 152         |
| edf (φ=10)   | 0.093 | 0.936 ± 0.034 | 85          |
| no tempering | 1.000 | 0.895 ± 0.029 | 0           |

### Test 5: Element-type coverage

Tempering improves all element types.  Off-diagonal imaginary is
consistently the hardest to cover.

| Config         | Overall | Diag  | OD Re | OD Im | Divergences |
|----------------|---------|-------|-------|-------|-------------|
| auto Nb4 Nh1   | 0.893   | 0.931 | 0.975 | 0.775 | 240         |
| auto Nb4 Nh2   | 0.939   | 0.981 | 0.992 | 0.845 | 45          |
| auto Nb4 Nh4   | 0.955   | 0.987 | 0.997 | 0.882 | 415         |
| noeta Nb4 Nh1  | 0.848   | 0.855 | 0.918 | 0.771 | 407         |
| noeta Nb4 Nh2  | 0.895   | 0.897 | 0.919 | 0.870 | 0           |
| noeta Nb4 Nh4  | 0.903   | 0.904 | 0.923 | 0.883 | 2           |

### Test 6: LISA XYZ noise

c=2 generalises to LISA coloured noise (non-VAR, with transfer function
structure).  Requires null excision and sufficient knots (K=50).

| Seed | Overall | Diag  | OD Re | OD Im | η     | Divergences |
|------|---------|-------|-------|-------|-------|-------------|
| 0    | 0.956   | 0.927 | 0.942 | 1.000 | 0.071 | 107         |
| 1    | 0.958   | 0.930 | 0.943 | 1.000 | 0.071 | 0           |
| 2    | 0.961   | 0.934 | 0.950 | 1.000 | 0.071 | 0           |

## Warmup-only tempering (removed)

We tested a two-stage scheme: adapt NUTS at a low eta during warmup,
then sample at eta=1 using the learned step size and mass matrix.  This
**did not show a clear benefit** and sometimes increased divergences.
The feature (`warmup_eta` config field) was removed from the sampler.

## Scripts

### `eta_tempering_study.py` (original study)

Phase 1-3 grid sweeps that established the `c/(Nb*Nh)` formula.

```bash
python eta_tempering_study.py all --seeds 0-3
```

### `eta_validation_study.py` (robustness validation)

Tests whether c=2 generalises beyond the original VAR(2) p=3 setup.

```bash
# Test 3: K-sweep — does coverage stay ~0.90 as n_basis varies?
python eta_validation_study.py test3 --seeds 0-4

# Test 4: analytic eta candidates — can we derive c from B,P matrices?
python eta_validation_study.py test4 --seeds 0-4

# Test 5: element-type coverage — diag vs offdiag, with and without tempering
python eta_validation_study.py test5 --seeds 0-4

# Test 6: LISA sanity check — non-VAR coloured noise with analytic truth PSD
python eta_validation_study.py test6 --seeds 0-2

# Run all + generate plots
python eta_validation_study.py all --seeds 0-4
python eta_validation_study.py plots
```

#### Test details

| Test  | Question                                              | Config                                           |
|-------|-------------------------------------------------------|--------------------------------------------------|
| test3 | Is coverage stable across K (n_basis)?                | K={10,15,20,30,40}, Nb=4, Nh=2, eta=auto c=2    |
| test4 | Can we derive c analytically from tr(P), tr(B'B)?     | Compares empirical c=2, edf-based, no tempering  |
| test5 | Does tempering help all element types equally?         | 3 Nb/Nh configs, auto vs eta=1, per-element cov  |
| test6 | Does c=2 work on LISA XYZ noise (non-VAR)?            | 7-day LISA, Nb=7, Nh=4, K=50, null excision      |
