# VI development and validation

The development branch is `hacking-vi-nuts-validation`. Runtime support and
historical studies have separate commits. The current investigation is the
native frequency-only stationary AR(4) model, reusing its frozen numerical
observations, basis, penalty and target-specific NUTS references.

- Current entry point: `stationary_next.py` (bounded stages 0–2).
- Current result: fixed-smoothing full covariance passed agreement, late
  stability and seed-repeatability screens; inferred-smoothing VI was stable
  and repeatable but its physical/log roughness uncertainty was too narrow.
- [Current report](../../docs/development/vi_nuts_validation/stationary_next_report.md)
- [Essential files and archival candidates](../../docs/development/vi_nuts_validation/branch_inventory.md)
- [Opt-in fitting diagnostics](../../docs/development/vi_nuts_validation/usage.md)

The source study archives under `runs/` are ignored by Git and must be retained
separately. Checking out this branch does not recreate them. Offline reporting
uses `stationary_next.py summarize-next`; `continue` is the original bounded
controller, not authorization for new benchmarking, new records or further
optimization. No production VI preset or speed advantage is established.

## Historical four-target diagnostics and optimization studies

The commands below reproduce earlier, distinct studies. They are not the
current stationary recipe or a budget for additional experiments.


This benchmark audits optimization, posterior approximation and joint importance proposals separately. It freezes each likelihood, observation array, basis, prior, normalization and precision. It never selects windows or calibrates a variational posterior from truth.

Run in a fresh environment with the project dependencies (`numpyro>=0.22.0`), with the worktree source first on the import path:

```bash
uv venv
uv pip install -e '.[dev]'
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python -m pytest
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/run.py \
  --previous /path/to/dynamic-whittle-stages-0-3 \
  --out runs/vi-nuts-validation-new
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/summarize.py \
  --previous /path/to/dynamic-whittle-stages-0-3 \
  --out runs/vi-nuts-validation-new --fields
```

`--target exact_stationary` or `--target exact_time_varying` runs an exact Gamma power-likelihood control independently of the prior archive. The two raw targets load the earlier saved `cal_ar1_m8_nuts_none_init7101/result.nc` and `cal_slow_m32_nuts_none_init7101/result.nc`, including the exact observations, coordinates and basis. Missing inputs produce saved failure records. Completed target directories cannot be overwritten.

`config.toml` declares four targets, three guide families, three optimization seeds, three actual-step checkpoints, four-chain NUTS screens and independent diagnostic/evaluation streams. References initialize each chain independently, without VI. Failed references get one separately saved sampler-tuning repair. `repair_reference.py` offers one additional explicit tuning attempt (acceptance .999, depth 12, 2000 warmup/4000 draws per chain) and refreshes comparisons from saved guide checkpoints only when its screens pass. Original references and comparison artifacts are retained.

A completed experiment is not an accepted reference or a validated guide. Check the status and functional MCSE before reading numerical discrepancies as algorithm error. MMD baselines are descriptive for autocorrelated NUTS samples. No IID permutation test is performed. Importance-weight ESS is distinct from MCMC ESS. No reweighting is applied to raw VI outputs.

Artifacts include complete posterior coefficients/hyperparameters, chain/draw coordinates, learned guide checkpoints, repeated-seed packed draws and densities, model fingerprints, objective evaluations, NUTS energy/statistics and per-feature comparisons. Field analysis reconstructs all coefficient draws in bounded frequency chunks, including covariance contributions to selected log-spectrum projections. It does not use spectrum previews as posterior draws. Guide checkpoints use JSON tree recipes plus NPZ numerical leaves, without pickled code. NetCDF PSDResult round trips also preserve these states.

Source SHA, dirty-content hash (including untracked source), package versions, device/backend and actual dtypes are saved. Fit-phase and total workflow times are retained, including compilation-containing phases; checkpoint objective evaluations are included in the recorded historical remaining optimization phase. Additional reconstruction/comparison/field costs are separate.

Predictive adequacy, held-out whitening, simulation-based calibration, raw-process repeated coverage and large-basis accuracy are deferred. The whitening primitive has analytic contract tests, but power-only data do not provide coefficient phases. There is no optimizer-state resume recipe or flow checkpoint reconstruction. No OzSTAR access is needed.

## Fixed-target optimization follow-up

`optimize.py` loads the saved exact time-varying target and copies its accepted
four-chain reference. It checks numerical arrays, target fingerprints, file
hashes and archived log-joint evaluations before running any fits. It never
calls the data generator or NUTS. `optimization.toml` freezes a diagonal-guide
experiment with three learning-rate schedules, 1/8 optimization particles,
three matched seeds and seven checkpoints through 40,000 updates. Training,
objective-evaluation and density-diagnostic particle counts are separate.

```bash
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/optimize.py \
  --reference runs/vi-nuts-validation-v2/exact_time_varying \
  --out runs/vi-optimization-exact-tv-new
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/optimization_report.py \
  --out runs/vi-optimization-exact-tv-new
```

Completed jobs are preserved on resume; input/configuration changes and
incomplete-job overwrites are rejected. Full joint packed checkpoint draws,
derived features, learned guide parameters, final constrained draws, repeated
PSIS and fixed/independent objective evaluations are retained. The report
separates late checkpoint drift, cross-seed repeatability and NUTS accuracy.
Decreasing the step size until a guide stops moving does not establish a
repeatable optimum. No new guide or parameterization is part of this follow-up.

The exploratory `optimization_tail.toml` extends the selected eight-particle
cosine setting through 80,000 updates, holding its rate at 0.0001 after 40,000.
Use a separate output directory and retain the primary experiment for prefix
parity:

```bash
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/optimize.py \
  --config examples/vi_nuts_validation/optimization_tail.toml \
  --out runs/vi-optimization-exact-tv-tail
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/optimization_report.py \
  --out runs/vi-optimization-exact-tv-tail \
  --report docs/development/vi_nuts_validation/optimization_tail_report.md
```

`prefix_reference` in the tail configuration identifies the primary artifact
directory used for the 40,000-step checkpoint identity check; change it for a
new reproduction directory. The tail experiment was chosen after the primary
results and does not constitute untouched-seed confirmation. The combined
interpretation is in `docs/development/vi_nuts_validation/optimization_findings.md`.
