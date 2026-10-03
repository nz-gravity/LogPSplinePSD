# VI branch: what to keep and what can be retired

The VI development is preserved on `hacking-vi-nuts-validation`, based on
`63b5c7a`. Runtime changes and research studies have separate commits. This is
a research branch: fixed-smoothing full covariance passed the declared screens,
but the inferred-smoothing posterior remains underdispersed in roughness.
There is no validated production preset or demonstrated speed advantage.

## Essential fitting code

| Files | Purpose | Keep |
|---|---|---|
| `src/log_psplines/inference/vi.py` | One shared SVI engine, full joint draws, data-based initialization, Adam/clipping, callable schedules, independent training particles, actual update counts and checkpoint callback | Yes. The scheduled eight-particle recipe is configured by the study runner through `fit_vi`; setting public `vi_steps=60000` alone does not reproduce it. |
| `src/log_psplines/config.py`, `fit.py`, `inference/power.py` | Opt-in VI dispatch to the existing model and reconstruction; diagnostic/stopping controls | Yes for public VI use. The power backend also serves the older moving-periodogram study; it is not needed to define the current frequency-only stationary target. |
| `src/log_psplines/results.py` | Preserve losses, timings, guide parameters and complete joint posterior information through result persistence | Yes. A guide checkpoint is not optimizer state. |
| `src/log_psplines/diagnostics/variational.py` | Target identity, guide reconstruction, packed density/Jacobian contracts, PSIS and explicit nonfinite/unavailable statuses | Yes for diagnosed, inspectable VI. The same module currently also contains offline comparisons; those can eventually be separated without changing the inference engine. |
| `tests/unit/test_variational.py`, `tests/time_varying/test_power_vi.py` | Density, RNG, persistence and public backend contracts | Yes. These software tests do not establish posterior accuracy. |

Existing release code already supported Cholesky/stationary VI, diagonal/full/
low-rank/flow guides and `PSDResult.vi`. This move isolates the new extensions
and investigations; it does not delete the released public API. No new flow
experiment or different parameterisation is introduced.

`inference/parametric_power.py` explicitly rejects unsupported parametric VI.
Keep that guard while `PowerConfig` exposes a method selector. The NumPyro
0.22.0 lower bound is retained for the installed native diagnostic contract;
basic SVI alone does not explain that constraint. Relaxing it would require a
separate compatibility check.

## Essential to reproduce and diagnose the current stationary result

The active dependency boundary is now small and explicit:

```text
stationary_next.py -> stationary.py, study_common.py, comparison.py
extensions.py -> extension_targets.py, study_common.py, comparison.py
```

`study_common.py` contains the exact inherited schedule, packed/joint draws,
MMD, quadrature and persistence functions. `comparison.py` contains the unchanged
MC-aware accuracy and paired checkpoint formulas, with explicit feature roles
for the new native cases. Synthetic nontrivial comparisons were bitwise equal
to the pre-cleanup implementations. Active imports do not load historical drivers.

The former `run.py`, `optimize.py`, report generators, extra-reference repair and
sweep specifications are retained under `examples/vi_nuts_validation/archive/`.
The complete old moving-periodogram study is in `archive/dynamic_whittle/`.
Numerical archives and historical failures were not deleted or rerun.

Keep `tests/unit/test_stationary_vi_investigation.py` and
`test_stationary_vi_continuation.py`. The `run_nuts(init_strategy=...)` addition
and corrected trajectory-cap convention are needed by the diagnosed references,
not by the VI optimizer itself. The cap correction is also a general NUTS fix.

Keep the frozen observations, numerical basis/penalty, accepted references,
all guide checkpoints, failed attempts, source snapshots and timing records.
They live in the ignored `runs/` directories, so **checking out the Git branch
alone does not restore the numerical study archive**. The stationary continuation
loads these saved inputs; do not substitute newly generated data or a fresh
reference and call it a replay.

## No longer needed as active development paths

| Item | Current role | Disposition |
|---|---|---|
| `examples/vi_nuts_validation/archive/repair_reference.py` | Extra reference repair for the earlier four-target study | Archived. It is not the current stationary repair policy and must not be used to spend an exhausted budget. |
| `summarize.py`, `optimization_report.py` | Earlier saved-study report generators | Archived with their matching reports; inputs remain under ignored runs/. Neither is imported by the stationary continuation. |
| `config.toml`, `optimization.toml`, `optimization_tail.toml` | Earlier target/guide/optimizer sweeps | Keep as historical specifications, not current recipes. No sweep or additional tail is needed for the present question. |
| Old diagonal, time-varying, particle and schedule sweeps | Explain why the current candidate and controls were selected | Preserve outcomes; do not rerun them as part of stationary validation. |
| `examples/vi_nuts_validation/archive/dynamic_whittle/` and matching development reports | Original mixed VI/NUTS moving-periodogram investigation | Historical study on this branch. Its coupled VI backend means moving the complete study is safer than leaving an unusable runner in the primary checkout. |
| `whiten_coefficients` in `diagnostics/variational.py` | Analytic tests for a deferred predictive-adequacy primitive | Not needed for current inference or posterior comparisons. Candidate to move out of the runtime module; tests are its only current consumer. |
| Noise-aware early stopping | Optional, tested optimization heuristic | Not used by the fixed 60k study. Keep separate from posterior validation and do not present it as an established stopping recipe. |
| Low-rank and flow guide options | Existing public functionality | Not needed for this study; retain for compatibility rather than deleting unrelated released support. |

None of this historical evidence was discarded. No study inference, benchmark,
untouched-seed confirmation, dependency upgrade or scientific target change is
part of this branch move.

## Shared changes that are not specifically VI

Bounded-row scattered PLS initialization, restored observed-data units, the
NUTS trajectory-cap correction, and `diagnostics/stationarity.py` have independent
uses. The primary checkout retains these, along with its expected-power forward
map, expected-power study, dynamic-Whittle contracts and unrelated staged edits.
They should not be removed merely because VI also used them.

The primary checkout's scalar VI backend extensions, associated public docs,
VI tests and complete mixed VI/NUTS study are moved onto the VI branch. A full
pre-move file/index backup is retained locally under
`runs/vi-branch-move-20261003/primary-before/`. Original scientific archives are
untouched. No main/release branch or remote is updated.

## Cleanup completed and remaining work

The active helper consolidation and historical driver relocation are complete.
Keep the public fitting engine, opt-in diagnostics, persistence contracts and
current bounded studies. Preserve public low-rank/flow options for compatibility;
they are not part of these experiments. The deferred whitening primitive and
optional early stopping remain separate from posterior validation.

The extension study tests one stationary two-channel VAR record and the archived
scalar exact-power time-varying control, each with its own fixed/hierarchical
reference and gate. See `extensions_report.md` for actual outcomes and budgets.
The earlier AR(4) underdispersion remains a retained finding. No production
preset or efficiency benchmark is implied by cleanup or stable optimization.

## Move verification

The assembled VI branch passed 173 tests with 29 warnings in 82.32 s. The
cleaned primary checkout passed 144 tests with 19 warnings in 49.92 s, including
its expected-power and stationary/NUTS regressions. Both runs used the existing
VI worktree environment with CPU float64 and the tested checkout first on
`PYTHONPATH`; no dependencies were changed. Ruff and Git whitespace checks
passed. No scientific study was rerun.

The archived continuation's 31 source hashes still match. The move verified
that every unrelated primary file and index entry remained byte-identical,
that the expected-power forward map and scattered PLS functions were unchanged,
and that observed-unit restoration remained present. The primary branch HEAD
stayed at `63b5c7a`; its unrelated work remains staged/unstaged as before.

Move manifests, the original primary files/index and both test logs are saved
under `runs/vi-branch-move-20261003/`. The branch remains local; no push, merge
or release was performed.

## Cleanup and extension verification

The cleanup and new native-target contracts passed 176 tests with 28 warnings
in 64.80 s. Archived CLI help imports passed after relocation. Extracted
MC statistics, accuracy and stability calculations were bitwise equal to the
original formulas. The new study verified native/conditional gradients,
raw-latent health of both accepted references, six guide round trips,
36 parameter checkpoints and the original 31 stationary source hashes.
Numerical failures and stop gates are recorded in `extensions_report.md`.
The public fitting API and native priors/likelihoods were unchanged by cleanup.
