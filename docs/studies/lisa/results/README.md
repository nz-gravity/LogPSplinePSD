# LISA Study Runs

This directory contains multivariate LISA PSD study runs produced by `docs/studies/lisa/lisa_multivar.py`.

## Run Name Decoder

Each run directory name encodes the main settings:

- `nb<k>`: number of time blocks `Nb`
- `lb<k>`: block length `Lb` in samples
- `cgNc<k>` or `cgOff`: coarse graining on/off, targeting `Nc` retained frequency bins
- `k<k>`: spline knot count
- `f1e-04-1e-01`: analysis band `[fmin, fmax]`
- `ta0p7`, `ta0p9`: NUTS target acceptance
- `td10`: max tree depth
- `dmOn`: dense mass matrix enabled
- `nutsW1500S1000`: 1500 warmup, 1000 posterior draws per chain
- `viOn`: VI run used for initialization and/or diagnostics
- `tau1`: `DESIGN_PSD_TAU=1.0`

## Quick Assessment

### Best current run

- `lisa_nb52_lb120960_cgNc2014_k20_f1e-04-1e-01_ta0p7_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- Why: best balance of sampler health and PSD accuracy.
- Diagnostics: `Rhat max=1.006`, `17/4000` divergences, no tree-depth hits.
- Accuracy: matrix `RIAE=0.2212`, coherence `RIAE=0.02127`.

### Also usable

- `lisa_nb52_lb120960_cgNc2014_k20_f1e-04-1e-01_ta0p7_td10_dmOn_nutsW1500S1000_steps10000_viOn_tau1`
- Nearly identical PSD accuracy to the best run.
- Slightly fewer divergences, but slightly worse `Rhat` and coverage.
- No clear improvement over the non-`tau1` variant.

- `lisa_nb2_lb120960_cgNc2014_k20_f1e-04-1e-01_ta0p7_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- Point estimates are strong: matrix `RIAE=0.2228`.
- Sampler geometry is rougher: `168/4000` divergences, `Rhat max=1.021`.

### Converges but fits badly

- `lisa_nb2_lb120960_cgNc2014_k10_f1e-04-1e-01_ta0p7_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- Sampler diagnostics look fine.
- PSD fit is poor: matrix `RIAE=0.7142`.
- Interpretation: `k=10` is too inflexible for this problem.

### Failed / not trustworthy

- `lisa_nb365_lb17280_cgNc1079_k10_f1e-04-1e-01_ta0p9_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- Catastrophic NUTS pathology.
- ESS collapses to `4`, `Rhat` explodes, tree depth hit rate is `100%`, step size collapses to `2.35e-06`.
- PSD accuracy is unusable.

- `lisa_nb14_lb17280_cgNc1079_k10_f1e-04-1e-01_ta0p9_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- `lisa_nb14_lb17280_cgOff_k10_f1e-04-1e-01_ta0p9_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- These appear to be VI-only or partial runs.
- Their `inference_data.nc` files do not include `sample_stats`, unlike full NUTS runs.
- VI diagnostics are explicitly marked not trustworthy (`PSIS k-hat` about `1.5`).

- `lisa_nb14_lb17280_cgOff_k10_f1e-04-1e-02_ta0p9_td10_dmOn_nutsW1500S1000_steps10000_viOn`
- Incomplete. Only preprocessing output is present.

## What The Runs Suggest

### 1. Long blocks matter more than short blocks

The strong runs all use `lb120960`. The short-block `lb17280` regime performs poorly or fails outright.

Working interpretation:

- short blocks reduce frequency resolution,
- off-diagonal / coherence structure is harder to identify,
- posterior geometry becomes much harder for NUTS.

### 2. `k=20` is much better than `k=10`

The `nb2` comparison isolates this cleanly:

- `k=10`: matrix `RIAE=0.7142`
- `k=20`: matrix `RIAE=0.2228`

So the main issue in the `k=10` run is model flexibility, not MCMC convergence.

### 3. More blocks help stabilize `k=20`

Comparing the `k=20` runs with `lb120960`:

- `nb2`: accurate, but many divergences
- `nb52`: equally accurate, much cleaner geometry

That makes `nb52 / lb120960 / k20` the current baseline winner.

### 4. VI is useful for initialization, not for final inference

Across the saved runs, VI diagnostics are poor:

- PSIS `k-hat` is too high,
- variance ratios are badly mismatched,
- summaries explicitly flag the approximation as unreliable.

Even when VI point estimates are close to NUTS on the good runs, uncertainty calibration is not trustworthy.

## Frequency-Resolved Notes

For the current best run (`nb52 / lb120960 / k20`):

- the saved `riae_vs_freq.png` does **not** suggest the main point-estimation error is concentrated at the lowest frequencies,
- the larger relative-error spikes are mostly in the mid/high part of the band, especially around roughly `0.04-0.05` and `0.08-0.10`,
- low-frequency coverage is still poor, but coverage is poor across most of the band, not only at low frequency.

So the main remaining problem is not "low frequency only". It is more that interval calibration is weak overall, and some diagonal/off-diagonal components have localized mid/high-frequency spikes.

## Recommended Next Run

If you want one run that is likely better than the current baseline, the safest next step is a small local refinement of the winning configuration rather than another large sweep.

Suggested next run:

- start from `nb52 / lb120960 / k20`
- keep coarse graining at `cgNc2014`
- keep dense mass on
- keep `DESIGN_PSD_TAU` off unless there is a separate prior reason to use it
- raise `target_accept` from `0.7` to `0.8` or `0.85`
- raise `max_tree_depth` from `10` to `12`

Why:

- the current best run is already in the right region,
- it still has a small number of divergences,
- higher `target_accept` is the least disruptive way to reduce those without changing the model class.

Possible follow-up experiments after that:

1. `nb52 / lb120960 / k24` or `k28`
2. same run with a narrower high-frequency cap if the `0.08-0.10` region remains unstable
3. a richer VI guide only if needed for initialization quality, not as a replacement for NUTS

## Practical Default

Until a better run is available, use:

- `lisa_nb52_lb120960_cgNc2014_k20_f1e-04-1e-01_ta0p7_td10_dmOn_nutsW1500S1000_steps10000_viOn`

and treat it as:

- good for point estimation,
- not fully satisfactory for posterior interval calibration.
