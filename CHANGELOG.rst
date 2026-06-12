.. _changelog:

=========
CHANGELOG
=========


.. _changelog-v0.1.0:

v0.1.0 (2026-06-12)
===================

Bug Fixes
---------

* fix: update workflow (`7af67ce`_)

* fix: update workflow (`09b80a9`_)

* fix: update tests (`c42299b`_)

* fix: vi diagnostics and plotting (`ea1a087`_)

* fix: add more MCMC sampler configuration (chain method) and diagnostics options (`2c5d09d`_)

* fix: improve diagnostics for multivar-blocks (`08ad439`_)

* fix: adjust freq regions (`f71d2f6`_)

* fix: CI fixes (`598a80e`_)

* fix: chnage release (`bcec12d`_)

* fix: chnage release (`2a4da4b`_)

* fix: chnage release (`d9a12cf`_)

* fix: chnage release (`eedaa20`_)

* fix: fmax error correction (`55f35de`_)

* fix: demos (`45f3b46`_)

* fix: plotly plotting (`5253690`_)

* fix: adjust default settings (`b15f183`_)

* fix: made plotting changes (`5e8185b`_)

* fix: refactor line-locator name to lvk-allocator (`13e7416`_)

* fix: LVK knot allocation fix (`37e2d20`_)

* fix: pytests (`825ea13`_)

* fix: refactoring (`159d387`_)

* fix: add sampler option (`a6a45c0`_)

* fix: add RNG logging and verbosity (`b5c37b6`_)

* fix: update runner (`fa742fa`_)

* fix: add benchmarking fix for cli (`8c05631`_)

* fix: update inference result saving/loading (`551685d`_)

* fix: fix pypi name bug (`a9d0101`_)

* fix: pypi readme fix and updating demo (`6040a5c`_)

* fix: add gwpy for dev options (`6b73f16`_)

* fix: add arviz (`da8d428`_)

* fix: add diagostics and ar dataset for tstig (`efdbcaf`_)

* fix: add demo to docs (`81c0bbb`_)

* fix: init weights with mse istead of lnl (`9df1e5d`_)

Chores
------

* chore(release): 0.0.14 (`48aecf4`_)

* chore(release): 0.0.13 (`2d993d9`_)

* chore(release): 0.0.12 (`bfa1388`_)

* chore(release): 0.0.11 (`877e6c8`_)

* chore(release): 0.0.10 (`0290b42`_)

* chore(release): 0.0.9 (`b4f97a5`_)

* chore(release): 0.0.8 (`f838433`_)

* chore(release): 0.0.7 (`4cf5f52`_)

* chore(release): 0.0.6 (`6c4c175`_)

* chore(release): 0.0.5 (`d0e06f6`_)

* chore(release): 0.0.4 (`d7d7355`_)

* chore(release): 0.0.3 (`f03de7d`_)

* chore(release): 0.0.2 (`9e906dc`_)

* chore(release): 0.0.1 (`2837da2`_)

Continuous Integration
----------------------

* ci: update pypi to only release for specific tags (`278441c`_)

Documentation
-------------

* docs: update docs (`0c518bf`_)

* docs: fix docs (`62c196a`_)

* docs: add plots (`185e176`_)

* docs: add colab button (`f6ef8ae`_)

* docs: add some notes for commits (`2aee2f3`_)

Features
--------

* feat(plotting): shade excluded frequency bands in diagnostic plots

Add _shade_excluded_bands helper to both the eigenvalue separation
diagnostic and PSD matrix plots. Excluded regions are highlighted with
a tan axvspan so it is immediately clear which frequency bins were
removed from the likelihood during preprocessing. (`34858e4`_)

* feat(multivar): add frequency exclusion, Wishart floor, and detrend control

Introduce three new capabilities needed for LISA TDI null handling:

- exclude_freq_bands: post-coarse-grain frequency band exclusion on
  MultivarFFT via exclude_frequency_bands / filter_frequency_mask.
  Bands are normalised (sorted, merged) and propagated through the
  full preprocessing pipeline including Welch overlays.
- wishart_floor_fraction: per-bin eigenvalue flooring on Wishart
  matrices to stabilise Cholesky / likelihood near deterministic
  spectral nulls (e.g. TDI transfer zeros). Uses per-frequency trace
  as the reference to preserve eigenvalue structure across the full
  dynamic range.
- wishart_detrend: expose detrend mode ("constant", "linear", False)
  on block-averaged FFT, previously hardcoded to constant subtraction.

Also fix _derive_vi_coarse_grain_config to snap explicit Nc/Nh to the
nearest divisor of the full frequency count instead of silently
producing a non-integer coarse-grain ratio. (`616e469`_)

* feat: add alpha_phi_theta and beta_phi_theta parameters for off-diagonal P-spline blocks (`65e5c0a`_)

Refactoring
-----------

* refactor: avoid registering large deterministic sites to reduce memory usage (`25b0cc4`_)

* refactor(lisa): update study for TDI null exclusion and Wishart floor

Wire up the new exclude_freq_bands and wishart_floor_fraction options
throughout the LISA study. Key changes:

- main.py: add --exclude-transfer-nulls and --wishart-floor-fraction
  CLI flags; physics-informed null bands derived from TDI arm length.
- utils/preprocessing.py: compute null-band edges from L/c and expose
  them for both likelihood exclusion and plot shading.
- utils/metrics.py: skip null-band regions when computing coverage
  and other scalar metrics to avoid penalising known singularities.
- utils/plotting.py: pass excluded_bands to PSDMatrixPlotSpec so
  shading is consistent between inference and diagnostic figures.
- utils/inference.py, utils/data.py: minor plumbing updates.
- diagnose_wishart_bias.py: standalone script to audit eigenvalue
  flooring impact vs raw Wishart matrices.
- vi_transfer_null_smoke.py: quick smoke test for VI initialisation
  near transfer nulls.
- run_study.sh: convenience wrapper for batch seed runs.
- README.md: document null-frequency issue, fixes, and run inventory. (`89db68e`_)

* refactor: streamline imports and improve frequency handling in VARMAData; update tests for frequency validation (`1903a98`_)

Testing
-------

* test: cover frequency exclusion, Wishart floor, and coarse VI adjustment

New test modules:
- test_multivar_frequency_exclusion: exclude_frequency_bands and
  filter_frequency_mask correctness.
- test_multivar_wishart_detrend_unit: detrend modes ("constant",
  "linear", False) and wishart_floor_fraction eigenvalue flooring.
- test_lisa_study_preprocessing_unit: LISA study preprocessing helpers.
- test_lisa_study_plotting_unit: LISA study plotting helpers.

Extended existing suites:
- test_coarse_vi_init: Nc/Nh auto-adjustment to nearest divisor.
- test_plotting_base: excluded_bands shading rendered as axvspan patches.
- test_preprocessing_diagnostics: excluded_bands passed to eigenvalue
  separation plot; updated diagnostic output path assertion. (`c9e6360`_)

* test: implement simulation of independent AR(1) channels and add consistency test for univariate and multivariate scaling

Added an AR(1)-based regression test that simulates identical independent channels, runs univariate and multivariate samplers, and checks frequency alignment plus percentile ratio statistics on the diagonal PSDs to guard against scaling mismatches. (`4390cd8`_)

Unknown
-------

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`f118ecd`_)

* Merge pull request #50 from nz-gravity/copilot/skip-lnz-test

skip: disable failing LnZ tests while LnZ is not in use (`662efed`_)

* skip: mark failing LnZ tests as skipped, disable compute_lnz in multivar integration test (`0e95fb8`_)

* KILL UNIVAR.py (`b8e58fd`_)

* confirm that we are adding eta to the LnZ kwargs (`72db98c`_)

* fix LnZ using a whitening step (`cbbd2b7`_)

* nuke old plotter (`4334af0`_)

* add logging fixes (`7515bd5`_)

* cleanup (`0fa25a6`_)

* cleanup (`b687361`_)

* remove lvk lisa (`52c4eba`_)

* more junk removal (`1655acd`_)

* Refactor sampler factory: Move sampler factory helpers to preprocessing.py

- Moved functions related to sampler creation and configuration from sampler_factory.py to preprocessing.py.
- Removed the sampler_factory.py file as it is no longer needed.
- Updated imports in relevant modules to reflect the new locations of the moved functions.
- Simplified the NUTSSampler class by removing the VIInitialisationMixin.
- Adjusted the initialization artifacts and runner functions to accommodate the refactor.
- Updated tests to align with the new structure and ensure functionality remains intact. (`80d6afb`_)

* adding in more pipeline stuff (`5436178`_)

* cleanup (`6eb5f77`_)

* cleaup VI plotting (`54c62ef`_)

* add energy plot (`af1940f`_)

* cleanup (`bfa242f`_)

* make plotting work for univar (`4e58da9`_)

* cleanup loading/saving (`1b477b2`_)

* simplify vi (`e58ed5e`_)

* addd mcmcs (`4a63c49`_)

* add shorter diagnostics (`1391b04`_)

* cleanup (`c8b2391`_)

* cleanup (`b0765bd`_)

* cleanup (`95e0621`_)

* break up adapters for VI (`1f08ce2`_)

* cleanup (`6768865`_)

* add delta for psline block (`ec291d8`_)

* edit eta comments (`c8890d0`_)

* add enbw and eta to vi (`bd1df91`_)

* fix test (`10b35b4`_)

* remove psis warning (`527251c`_)

* nuke useless test (`22a51b2`_)

* add ignore (`21dc4b3`_)

* add tests (`1b89dec`_)

* add runner (`cd6d246`_)

* add eta validation (`0379145`_)

* add tests for arious eta settings (`360ffb0`_)

* Enhance VI diagnostics with additional coarse-grained context and PSD extraction (`4447764`_)

* Merge branch 'main' into eta_tempering (`fd3689c`_)

* add some changes to VI init (`bf4f8d0`_)

* REMOVE EXTRA STUDIES (`26735ad`_)

* cleanup (`30c28d5`_)

* cleanup result reading (`251cc4e`_)

* fixes for VI runner (`51768dc`_)

* fix factory bug (`ee9a417`_)

* Merge pull request #47 from nz-gravity/better_memory_handling

Better memory handling (`bea276d`_)

* fix uni vs multi overlap (`67d3dde`_)

* cleanup (`033be92`_)

* lazy loading of posterior-psd (`f8f7cfa`_)

* add comment for pspline (`01357db`_)

* removed: large full-frequency deterministic state duplication in memory (`db45fa4`_)

* fix mem issue (`4de5b5b`_)

* add pytest-memray to dev dependencies; enhance testing for PSD reconstruction and weight sampling (`eec8130`_)

* fix mem issue (`708105f`_)

* nuke old run file (`24bfab2`_)

* cleanup for multivar (`4ce4cf0`_)

* Merge pull request #46 from nz-gravity/multivar_knot_alloc

Improve multivar knot alloc (`f5fdb6f`_)

* fix knot alloc (`2fc0b75`_)

* add fixes for knot alloc (`bd6c544`_)

* Merge remote-tracking branch 'origin' into multivar_knot_alloc (`f9936a9`_)

* Merge pull request #45 from nz-gravity/different_number_of_knots_per_cholesky_element

Allow different number of knots per cholesky-component (`3919f28`_)

* cleanup (`6ad2cf4`_)

* allow differnt numbner of cholesky components (`d4119c3`_)

* knot alloc: remove spectal method, fixes to cholesky method (`f8b3725`_)

* add more runners (`309e75e`_)

* start eta tempering studies (`a4a28de`_)

* add improved analysis (`a9fc38d`_)

* Merge pull request #43 from nz-gravity/wishart_hacking

Add frequency exclusion, Wishart floor, and TDI null handling (`9f2f10c`_)

* Refactor multivariate sampler to support per-component theta basis and penalties

- Updated the MultivarBaseSampler to log the number of B-spline basis models.
- Modified the _blocked_channel_model function to accept separate real and imaginary theta basis and penalty components.
- Introduced a method to retrieve per-theta basis and penalty arrays for blocked channels in MultivarBlockedNUTSSampler.
- Adjusted the _channel_model_kwargs method to pass the new per-component theta basis and penalties.
- Enhanced the VI initialization functions to accommodate the new structure of theta weights and basis.
- Updated tests to reflect changes in theta basis handling and ensure compatibility with new component structure.
- Removed legacy code related to shared theta basis and penalties, ensuring a cleaner implementation. (`7f50d88`_)

* Enhance multivariate PSD workflow and preprocessing

- Added support for additional command-line arguments in ``run_headnode_segment_sweep.sh`` to customize sampling, warmup, chains, and VI steps.
- Introduced a new script ``run_knot_compare_workflow.sh`` for executing knot comparison workflows with posterior options.
- Modified ``_preprocess_with_run_config`` in ``preprocessing.py`` to apply frequency exclusion after coarse graining, ensuring proper data alignment.
- Updated ``multivar_psplines.py`` to utilize channel-space Wishart matrices for knot scoring, preserving cross-spectral structure.
- Added unit tests in ``test_multivar_knot_methods_unit.py`` to verify the use of Wishart matrices in multivariate density scoring. (`9e5ddd5`_)

* more test jobs (`0daba3f`_)

* Enhance multivariate PSD analysis and diagnostics

- Updated subproject commit to indicate dirty state.
- Refactored metric extraction in ``metrics.py`` to streamline handling of multiple metrics.
- Modified ``3d_study.py`` to allow overriding of time samples and block counts, improving flexibility in simulations.
- Enhanced ``collect_results.py`` to accept a custom results directory for better organization.
- Added coherence metrics to ``from_arviz.py`` and adjusted ``to_arviz.py`` for complex truth handling.
- Introduced new methods in ``multivar.py`` for applying masks and cutting frequency ranges.
- Implemented frequency exclusion in ``preprocessing.py`` to enhance data preprocessing capabilities.
- Updated configurations in ``configs.py`` to include frequency exclusion bands.
- Added tests for frequency exclusion functionality in `test_preprocessing_diagnostics.py`.
- Created a new SLURM script for exploring coverage with varying segment sizes and coarse-graining configurations. (`16d5449`_)

* add lisa testing (`66659bc`_)

* hacking (`b47719d`_)

* adjustments to vi (`efe9227`_)

* update L2 logging, fix slurm mempry (`aa685d4`_)

* adjust slurm (`6b1bd12`_)

* adjust VI (`55787e3`_)

* more VI cleanup (`d807f24`_)

* simplify VI (`a0d80ae`_)

* hacking on VI: adding a coarse-grain VI (`fb88156`_)

* add prior plot (`34804bb`_)

* add prior plot (`021538f`_)

* Merge pull request #42 from nz-gravity/testing_different_phi

Use exact transformed-Gamma prior for log_phi in pspline_block (`e10d126`_)

* add more comments (`cbf2f5d`_)

* ran tests (`5436ff2`_)

* add least_squares_weight_initialiser (`4d5fbc0`_)

* Simplify docs (`d952114`_)

* remove lisa LnZ test and simplify logs (`8eec160`_)

* changes to lisa run (`692ab2e`_)

* Refactor LISA study into CLI-driven sim study with utils package

Replace the monolithic lisa_multivar.py (no CLI, hard-coded config) with a
clean, flexible simulation study setup suitable for 100-seed coverage analysis.

Deleted:
- lisa_multivar.py  → logic moved to main.py + utils/
- evidence_utils.py → lnZ always disabled in sim study
- check_preprocessing.py → one-off diagnostic, not needed
- run.slurm → replaced by lisa_sim.slurm

Added:
- main.py: single CLI entry point with flags for all key settings
  (--K, --knot-method, --diff-order, --coarse-Nc, --block-days,
   --duration-days, --wishart-window, --welch-window, --vi/--no-vi,
   --tau, --target-accept, --max-tree-depth)
- utils/data.py: inline data generation via generate_lisatools_xyz_noise_timeseries
- utils/preprocessing.py: block structure + coarse-grain setup
- utils/inference.py: run_mcmc wrapper; fixes knot_kwargs key (strategy→method,
  which was silently ignored before — all old runs used density placement)
- utils/metrics.py: RIAE/coverage/CI-width extraction; saves JSON+CSV+NPZ
- utils/plotting.py: PSD matrix plot with blocked Welch overlay
- utils/windows.py: window_spec/window_slug/welch_window_arg helpers
- lisa_sim.slurm: SLURM array job (--array=0-99, 1 seed/task, 7h walltime)
- collect_results.py: aggregates seed results, handles nested slug/seed layout (`6dba63b`_)

* Refactor sampler utilities: move functions to pspline_block.py and update imports

Also move LnZ drom mutlivar_block to multivar_base (`808d911`_)

* Revert default knot scoring to spectral, default window to rect; document findings

The Cholesky knot scoring refactor (246d57b) caused a ~5.5pp coverage drop
(0.872 → 0.816) confirmed over 100 seeds. Spectral energy scoring places knots
where PSD curvature is highest, giving better-calibrated CIs. Reverts
3d_study.py defaults to spectral scoring and rect window. Documents the full
knot scoring comparison in the study README. (`bc67aab`_)

* add more testing (`a215fc6`_)

* Add configurable knot scoring (cholesky vs spectral vs uniform) for coverage comparison

Restores the pre-refactor spectral-energy knot scoring as a selectable
option alongside the current Cholesky-based scoring. Adds --knot-method
and --knot-scoring CLI args to 3d_study.py. Includes SLURM scripts for
all three variants (uniform, density+spectral, density+cholesky). (`ecc9f7e`_)

* Change SLURM script to rect + Nh=4 for VAR3 coverage study (`caae21c`_)

* Add Tukey(0.1) + Nh=4 SLURM script for VAR3 coverage study (`f4d83b5`_)

* Reorganise README results table: separate rect baseline from Hann exploratory runs (`be27aeb`_)

* cleanup (`6610919`_)

* Add --outdir out_var3_sanity to var3d.slurm for scikit-fda baseline sanity check (`407fe61`_)

* Remove GPS penalty code; add post-mortem; update study README

The GPS general difference matrix (Li & Cao 2022) experiment is closed.
100-seed VAR3-3D studies showed a consistent ~5% coverage regression vs
the scikit-fda baseline (0.819 vs 0.873) with no compensating benefit.
Root cause: GPS D^T D approximates ∫B''B''dx with ≤1.5% error for
non-uniform knots, which is enough to shift posterior φ and narrow CIs.

Removed:
  - src/log_psplines/psplines/penalty.py (GPS penalty + phantom knots)
  - tests/test_gps_penalty.py (35 tests for removed code)
  - docs/studies/multivar_psd/var3d_gps_basis.slurm
  - docs/studies/multivar_psd/var3d_gps_basis_shrink.slurm

Added:
  - gps_penalty_postmortem.md (full experimental record for GitHub issue)

Updated:
  - docs/studies/multivar_psd/README.md (GPS section replaced with summary
    table and pointer to post-mortem)

181 tests pass, 6 skipped. (`6961c35`_)

* Revert to scikit-fda basis+penalty; drop GPS general difference matrix

GPS D^T D (Li & Cao 2022) with clamped knots gave 0.819 ± 0.059 coverage
across 99 seeds vs 0.873 ± 0.041 with the original scikit-fda O-spline
penalty -- a statistically significant 9-sigma regression.

The root cause: GPS D^T D and scikit-fda's exact integral penalty
∫ B_i''(x) B_j''(x) dx agree on the diagonal after max-norm but differ
on off-diagonal elements, changing the weight-prior covariance structure
and posterior φ in a way that over-smooths credible intervals.

Restore initialisation.py to use BSplineBasis + L2Regularization
(LinearDifferentialOperator) from scikit-fda, and add scikit-fda back
to pyproject.toml dependencies.  All 216 tests pass. (`ad61b63`_)

* Switch GPS basis/penalty back to clamped knots, drop phantom knots

Phantom knots made the interior penalty 1.67x stronger (interior diag = 1.0
vs 0.6 for clamped after max(|P|) normalisation), shifting the effective phi
prior and reducing coverage from 0.87 to 0.81.

Use _build_full_knot_vector (clamped, degree+1 boundary multiplicity) in
both build_bspline_basis_scipy and build_gps_penalty, exactly matching the
scale that scikit-fda's L2Regularization produced.  The GPS general
difference matrix (Li & Cao 2022) is still used — clamped knots do not
affect its correctness for uniform/near-uniform interior spacing.

Update test_endpoint_basis_not_pinned_to_one → test_endpoint_basis_pinned_to_one
and fix test_null_space_contains_linear to use the clamped knot vector. (`962c74a`_)

* Restore max(|P|) normalization in GPS penalty

Removing normalization entirely (commit 6263611) left raw D^T D values
~780,000 on the diagonal, causing φ to over-smooth catastrophically
(coverage dropped from 0.81 to 0.60).

Restore the max(|P|) = 1 normalization that matches the convention used
by scikit-fda's L2Regularization.penalty_matrix(). This keeps the φ prior
well-calibrated and should recover the 0.81 GPS coverage baseline. (`d4fdf0e`_)

* Fix GPS penalty normalization: remove max(|P|) scaling, use relative ridge

The previous max(|P|) normalization distorted the relative structure of the
penalty matrix because phantom knots place the maximum in the interior while
clamped knots placed it at the boundary.  After normalization the GPS penalty
was ~1.7× stronger in the interior, systematically narrowing credible intervals
and reducing coverage from ~0.87 to ~0.81.

Fix: return the raw D^T D penalty so knot-spacing geometry is preserved.
φ (phi) in the model already provides the scalar precision scaling, so no
overall normalization is needed.  Replace the fixed epsilon * I ridge with
epsilon * max(diag(P)) * I so the regularisation remains effective regardless
of knot spacing or scale.

All 35 GPS penalty tests pass; full suite 203 passed, 6 skipped. (`6263611`_)

* Add --tau shrinkage to 3d_study, GPS+shrink slurm, update README with 100-seed results

3d_study.py:
- Add --tau CLI flag (float, default None) to enable design-PSD shrinkage
- Wire tau through simulation_study() into run_mcmc() kwargs
- When tau is set, true PSD is automatically used as design target

var3d_gps_basis_shrink.slurm:
- New OzStar array job: GPS basis + rect window + tau=1.0 shrinkage
- 100 seeds (array=0-9, 10 seeds each), outdir=out_var3_gps_shrink

README.md:
- Add GPS penalty section with knot sensitivity probe table (K=10/20/30/50)
- Add 100-seed comparison: old basis+Hann (0.87) vs GPS+Rect (0.81)
- Note window confound and RIAE/ESS observations
- Document next step: GPS + Rect + light shrinkage (tau=1.0) (`52f687c`_)

* differnt savepath (`8c9c6a0`_)

* GPS study: use rect window, add --label flag for per-seed folder signage

- Switch var3d_gps_basis.slurm from hann → rect window (matches the
  existing baseline runs for apples-to-apples comparison)
- Add --label CLI arg to 3d_study.py: appends a short tag to each
  per-seed output folder (e.g. seed_0_short_nb4_N2048_K20_rect_noNh_gps/)
  so GPS vs old-basis results are unambiguous when scanning output dirs
- GPS slurm passes --label gps (`3eb0069`_)

* Prep GPS basis study: --outdir flag, fix slurm output dir, drop scikit-fda

- Add --outdir CLI argument to 3d_study.py so different study variants
  write to separate directories (e.g. out_var3_gps_basis vs out_var3)
- Update var3d_gps_basis.slurm to pass --outdir out_var3_gps_basis,
  preventing collision with existing hann-window baseline runs
- Remove scikit-fda from pyproject.toml dependencies (replaced by
  scipy BSpline + GPS penalty in previous GPS penalty PR) (`57750b9`_)

* Add SLURM script for VAR3-3D study with GPS penalty basis (100 seeds, K=20, Nb=4, hann window) (`804ec1e`_)

* Merge GPS penalty + comprehensive arviz 1.0.0 migration from claude/mystifying-moore

Our branch features (taking precedence):
- GPS penalty (penalty.py): Li & Cao 2022 with phantom knots & D_m^T D_m
- Comprehensive arviz 1.0.0: dict-style from_dict, DataTree, _psislw fix
- 176 passing tests, 50% coverage improvement (K=20: 0.71 vs old 0.47)

Main branch features (integrated):
- Pre-commit hook improvements (.venv/bin/python for mypy)
- Diagnostics cleanup and plotting enhancements
- Window parameter handling improvements

Conflict resolution: took our arviz + GPS penalty work, main's infrastructure
improvements. Removed compare_results.py (deleted on main).

Ready for VAR3-3D study runs on OzStar. (`cdcd610`_)

* Update README with GPS penalty coverage probe results (K=10-50 at N=2048) (`a570d64`_)

* Migrate to arviz 1.0.0 (arviz-base/arviz-stats) and fix GPS penalty

arviz migration (breaking API changes):
- az.from_dict(): keyword args → dict-first positional arg throughout
- idata.add_groups(): replaced with idata["name"] = xr.DataTree(dataset=ds)
- az.to_netcdf(): replaced with idata.to_netcdf(path, engine="h5netcdf")
- az.InferenceData(): replaced with xr.DataTree() + child assignment
- az.psislw(): replaced with _psislw() using arviz_stats._ps_tail directly
  with tail='right' to correctly detect heavy upper tails (the psislw
  wrapper in arviz_stats uses an inverted convention)
- az.hdi(hdi_prob=): renamed to prob=; coord 'hdi'/'higher' → 'ci_bound'/'upper'
- posterior.__class__(subset, attrs=...): replaced with xr.Dataset() in
  rhat.py and mcmc.py to avoid DataTree constructor errors
- az.plot_pair() API: removed divergences/kind/marginals kwargs; use
  marginal=True and aes_by_visuals for divergences
- ess_result.to_array(): replaced with iteration over data_vars
- idata.groups(): DataTree .groups is a property with '/'-prefixed paths;
  replaced with idata.children membership checks

GPS penalty (from previous session):
- New penalty.py with build_gps_penalty, build_bspline_basis_scipy using
  phantom knot vectors and Li & Cao (2022) D_m^T D_m difference matrix
- initialisation.py: replaced skfda with scipy BSpline + GPS penalty
- tests/test_gps_penalty.py: comprehensive unit tests

Other fixes:
- np.trapz → np.trapezoid (removed in NumPy 2.0)
- pyproject.toml: arviz[h5netcdf] dependency
- .pre-commit-config.yaml: fix mypy hook for systems without python symlink (`6f6a7f7`_)

* Enhance window parameter handling: support tuple input for wishart_window and add parsing function; introduce SLURM script for tukey window simulation. (`baeb884`_)

* Cleanup diagnostics output: merge plots, fix ArviZ 1.0, reorganize files

- Fix ArviZ 1.0 breaking change in plot_trace: remove deprecated combined/
  compact/figsize/divergences kwargs; use figure_kwargs and visuals instead;
  extract figure via PlotCollection.viz["figure"] for compatibility
- Merge VI diagnostics into diagnostics_summary.txt (append) instead of
  writing a separate vi_diagnostics_summary.txt
- Add composite_images_vertical() helper to plotting/base.py for stitching
  saved PNGs into a single figure
- Create sampling_diagnostics.png compositing summary_dashboard,
  ess_rhat_profiles, nuts_diagnostics, nuts_block_diagnostics
- Create accuracy.png compositing psd_truth_error_vs_freq, riae_vs_freq,
  coverage_vs_freq; save all three to diagnostics/ instead of outdir root
- Move preprocessing_eigenvalue_ratios.png into diagnostics/ subfolder
- Remove metrics_summary.csv from 3d_study.py (keep only JSON)
- Fix .pre-commit-config.yaml mypy hook: use .venv/bin/python
- Fix mypy: guard os.path.join against None outdir in multivar_base.py
- Fix mypy: add arviz/arviz_plots/requests to ignore_missing_imports;
  suppress pre-existing errors in known-noisy modules via ignore_errors (`59440bd`_)

* Refactors Cholesky mapping and improves diagnostics

Moves the PSD-to-Cholesky transformation into a shared utility for reuse and clarity, updating preprocessing and knot placement to leverage it. Enhances diagnostics plotting by subsampling trace plots for large posteriors, adds better plot layout and customization, and improves figure saving robustness. Updates logging and default plotting behaviors for more informative and manageable outputs. (`246d57b`_)

* chnage settings (`51c268a`_)

* Ignores PSD endpoint bins in diagnostics and plots

Prevents unreliable or outlier endpoint frequency bins from skewing
diagnostic metrics, coverage, RIAE, and plots by consistently excluding
the first and last bins when possible. Applies this logic to univariate
and multivariate PSD comparison, all plot renderings, and related test
assertions.

Adds utility for generating endpoint-excluding frequency slices and makes
diagnostics more robust, especially in presence of edge artifacts.

Also enhances NUTS block diagnostics with combined plot output and
optional per-block file saving, and adds informative metadata to
preprocessing diagnostics figures. (`7266799`_)

* changes to slurm (`58d6ed4`_)

* fix stupid typo (`a9977dd`_)

* improve preproc (`7d07ce4`_)

* arviz saving fixes (`7ae7535`_)

* arviz saving fixes (`cc6f34d`_)

* add fixes for preproc (`1a9c2a9`_)

* add result collection (`bf0f497`_)

* add 3D study for coarse grain (`65e26f7`_)

* fix tests (`976d515`_)

* bug fixes (`4982305`_)

* add new file (`0db59a5`_)

* add new file (`f0e6cb4`_)

* add new file (`7472888`_)

* Merge branch 'improved_basis' (`ede4b1f`_)

* add new file (`ff5fdbb`_)

* reduce logs (`6be8807`_)

* add **version** (`56fa75e`_)

* fixed window energy (`1d69109`_)

* Merge pull request #40 from nz-gravity/improved_basis

Improves multivariate PSD calibration and ArviZ compatibility (`0b6e053`_)

* typo fix (`3fefb46`_)

* Improves multivariate PSD calibration and ArviZ compatibility

Introduces window tapering support for multivariate PSD estimation and corrects posterior calibration by scaling the Whittle likelihood using equivalent noise bandwidth (ENBW). Refactors output and diagnostics to use xarray.DataTree for compatibility with ArviZ >=1.0.0, adds detailed coverage breakdowns, and enhances CLI experiment support for rank-deficiency analysis.

Clarifies credible interval coverage limitations in documentation and enables flexible window configuration to address spectral leakage and posterior miscalibration. (`4ab79be`_)

* edit settings (`31a685f`_)

* Add design_psd + tau soft shrinkage to LISA multivariate study

Wire up the recently-merged soft-shrinkage feature in the LISA
analysis script. With DESIGN_PSD_TAU=1.0, each P-spline component is
softly pulled toward the true LISA sensitivity model (true_psd_source),
which should reduce divergences and improve sampler convergence.

- Add DESIGN_PSD_TAU constant (float | None) for easy tuning/disabling
- Include tau value in the run slug so runs produce distinct directories
- Pass design_psd=true_psd_source and tau=DESIGN_PSD_TAU to run_mcmc() (`cd198ea`_)

* Merge pull request #38 from nz-gravity/add_soft_shrinkage

add soft shrinkage (`1f0336a`_)

* add docs (`8103f1d`_)

* Implement soft shrinkage toward design weights in multivariate sampler and add corresponding tests (`45047ac`_)

* hacking (`f020e3f`_)

* Enhance multivariate PSD diagnostics and streamline environment variable handling

- Removed environment variable functions in  and replaced with direct assignments for clarity and simplicity.
- Updated hyperparameters and configurations in  for improved model performance.
- Introduced new functions in  for generating per-frequency diagnostic plots when a true PSD is available, including relative error and coverage diagnostics.
- Added validation for positive definiteness of covariance matrices in .
- Integrated true PSD diagnostics into the multivariate sampler's plotting routine.
- Created a new markdown file for documenting large study results.
- Added tests for truth-aware PSD diagnostics to ensure functionality. (`daf5507`_)

* hacking on LnZ (`61f1635`_)

* add 3d stiudy (`2a3e72a`_)

* fix knot alloc (`b82a327`_)

* adjust LISA datagen (`ff5e8d7`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`4ae2430`_)

* Add files via upload (`8522040`_)

* Refactor and enhance multivariate P-spline functionality

- Updated the ``_posterior_subset_for_rhat`` function to avoid passing parent coordinates, preventing NaN issues during alignment.
- Introduced a new utility function ``_to_flat_finite_array`` to convert ArviZ/xarray outputs to flat finite float arrays, improving handling of R-hat and ESS calculations.
- Enhanced the ``VARMAData`` class to include empirical stationarity checks, adding flags and metrics for better diagnostics.
- Refactored the PSD matrix plotting functions to support per-panel knot extraction for multivariate cases, improving flexibility in visualizations.
- Implemented a new method for initializing knots based on raw frequency/power arrays, allowing for more robust knot placement strategies.
- Added tests for multivariate knot methods, ensuring support for various knot placement strategies and validating their behavior against empirical data.
- Created a new shell script for running VAR3 simulations, facilitating reproducibility in experiments. (`6a9763e`_)

* Refactor multivariate utilities and enhance VARMAData validity checks

- Moved _interp_complex_matrix import to multivar_utils.py for better organization.
- Added u_re_im_to_U function to combine real and imaginary components into a complex matrix.
- Updated VARMAData class to include validity checks for stationarity and data integrity.
- Enhanced test coverage for VARMAData validity checks and u_re_im_to_U functionality. (`254e135`_)

* Merge branch 'main' into copilot/fix-mypy-static-type-errors (`1fdb89a`_)

* more refactoring (`dae452a`_)

* use lambda and v to match up with paper (`45e2baf`_)

* Improve code comments for clarity

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`0f0da2c`_)

* Address code review feedback - improve comments and warnings

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`2fb7c24`_)

* Simplify sampler loop logic in runtime_benchmark

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`df536b3`_)

* Fix mypy type checking errors

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`3488eeb`_)

* Initial plan (`45c38c7`_)

* make coarse grain clearer (`8faa1ac`_)

* more cleanup (`79f3063`_)

* break up preprocessing (`4b9aafe`_)

* Add preprocessing module and refactor coarse grain functionality

- Introduced a new preprocessing module with configurations for MCMC and diagnostics.
- Implemented data preprocessing functions including alignment of true PSD and coarse graining.
- Refactored existing coarse grain functions to reside within the new preprocessing structure.
- Updated tests to reflect changes in module imports and removed deprecated sampler parameters.
- Ensured compatibility with existing tests by adjusting configurations and function calls. (`340ed5e`_)

* remove preprocessed imputs. Always pass in timeseries (`2e98313`_)

* Refactor MCMC diagnostics and configuration: enforce full diagnostics mode, remove unused options, and streamline preprocessing checks (`be30185`_)

* Add MCMC utility functions and enhance PSD matrix plotting

- Introduced a new module ``mcmc_utils.py`` containing various configurations and utility functions for MCMC sampling, including model configuration, diagnostics, and sampler factory configurations.
- Implemented functions to handle true PSD unpacking, interpolation, and normalization of coarse grain configurations.
- Enhanced the ``_prepare_processed_data`` function to handle different data types and added checks for frequency truncation.
- Updated the PSD matrix plotting functionality in ``psd_matrix.py`` to extract and plot spline knots from inference data, improving visualization of model fits.
- Added options to display knots in various panels of the PSD matrix plot, ensuring better representation of model structure. (`4bc22bc`_)

* remove old legacy stuff (`c2dede8`_)

* remove old legacy stuff (`cd1ff15`_)

* Merge pull request #36 from nz-gravity/hacking

hacking (`2b9f675`_)

* Merge remote-tracking branch 'origin/hacking' into hacking (`ec5f505`_)

* hacking (`0fd0481`_)

* hacking (`c7f67bc`_)

* simplify diagnostics (`116f59a`_)

* Unifies and refactors diagnostics and interpolation logic

Consolidates diagnostics code by removing redundant modules and integrating energy/E-BFMI and time-domain checks into a single diagnostics pipeline. Unifies frequency-indexed interpolation for PSDs and matrices, improving consistency and correctness when grids are unsorted or have duplicates. Refactors plotting and summary logic to support expanded NUTS convergence checks, clearer reporting, and more robust handling of high-dimensional outputs. Updates tests accordingly to match new diagnostics and interpolation paths.

Addresses scalability, maintainability, and reliability of core diagnostics and output routines. (`7b91d50`_)

* Merge pull request #35 from nz-gravity/typing

Type checking (`26af587`_)

* Refactor and enhance type checking across the log_psplines module

- Updated type hints to use Optional and Union for better clarity.
- Introduced runtime type checking for various functions to ensure input validity.
- Added assertions to check for None values in critical variables.
- Improved diagnostics handling in the save_vi_diagnostics functions.
- Enhanced the initialization of weights and basis matrices in the initialization module.
- Added tests for runtime type checking to ensure robustness against invalid inputs.
- Refactored plotting functions to ensure compatibility with updated type hints.
- Improved error handling and assertions in the multivariate sampler classes. (`5ff2a29`_)

* Refactor coarse graining functions and update imports to use multivar_utils; adjust tests for new configurations (`aa35d05`_)

* Merge branch 'refactoring_weights' (`d8b16e9`_)

* work more on docs (`bab7745`_)

* Merge pull request #34 from nz-gravity/refactoring_weights

Refactor freq_weights to Nh (`ed82fcf`_)

* more refactoring (`b599c15`_)

* Refactor frequency weights handling to use Nh instead of freq_bin_counts across multivariate samplers (`103684d`_)

* More changes (`120e901`_)

* Refactor frequency weights handling to use per-bin counts in multivariate samplers and related documentation updates (`6b4bb27`_)

* remove LISA specific code (`349a2f7`_)

* Simplify code (`7ec854d`_)

* Merge pull request #33 from nz-gravity/diagnostics_refactor

Refactor diagnostics modules and remove unused code (`f740a20`_)

* Refactor diagnostics modules and remove unused code

- Deleted psd_metrics.py and psd_posterior_diagnostics.py as they were no longer needed.
- Removed time_domain_moments.py and associated tests for similar reasons.
- Updated run_all.py to check for PSD datasets more efficiently.
- Adjusted diagnostics summary generation to utilize cached attributes for improved performance.
- Enhanced the BaseSampler class to compute and cache full diagnostics after inference.
- Modified multivar_blocked_nuts.py and nuts.py to ensure full diagnostics are cached after data creation.
- Removed outdated tests related to deleted modules and added new tests to verify caching behavior in diagnostics summary. (`6471534`_)

* Merge pull request #31 from nz-gravity/copilot/clean-up-unnecessary-pytests

Consolidate redundant PSD matrix test files (`d032bfb`_)

* Address code review feedback: use OUTDIR constant and remove trailing whitespace

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`b17556f`_)

* Consolidate PSD matrix tests into single refactor file

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`350fe2c`_)

* Initial plan (`5dc8dad`_)

* Merge pull request #32 from nz-gravity/copilot/improve-documentation-structure

Reorganize documentation into hierarchical structure (`7159d09`_)

* Update docs .gitignore to exclude _build directory (`b16a42a`_)

* Add Quick Links section and remove duplicate conventions file

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`51a29d6`_)

* Reorganize documentation structure with clear sections

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`e181b63`_)

* Initial plan (`7839d8a`_)

* Merge pull request #30 from nz-gravity/death_to_non_factorised_multivar

remove non-factorised multivar sampler (`845c919`_)

* remove non-factorised multivar sampler (`8f9afcc`_)

* Merge pull request #29 from nz-gravity/death_to_MH

remove MH (`c2e0cb1`_)

* remove MH (`2ad3756`_)

* Merge pull request #28 from nz-gravity/more_refactoring

more refactoring (`64ea085`_)

* nuke that noise-floor (`cd7914e`_)

* Refactor BaseSampler and VI initialization logic

- Improved the structure of the BaseSampler class by extracting methods for attaching VI diagnostics, computing chain summaries, and logging summary metrics.
- Enhanced error handling and logging for attaching VI diagnostics.
- Simplified the process of preparing buffers for blocked VI initialization by introducing a dedicated function.
- Refactored the initialization of values for blocked channels to improve clarity and maintainability.
- Updated tests to reflect changes in the psd_compare and psd_bands modules, ensuring they call the new private run methods.
- Added new unit tests for MCMC and PSD matrix functionalities to ensure robustness and correctness. (`065de15`_)

* add more conventions (`bb5f7d9`_)

* Merge branch 'refactor_nu_to_Nb' (`95ca9a8`_)

* more refactoring (`cbe644b`_)

* Merge pull request #27 from nz-gravity/refactor_nu_to_Nb

Refactoring to match paper (`76094ab`_)

* more refactoring (`a83a366`_)

* more refactoring (`55e51f1`_)

* Renames 'nu' to 'Nb' for Wishart degrees of freedom

Standardizes terminology across codebase by replacing 'nu' with 'Nb' to clarify its role as the number of averaged blocks in Wishart statistics.

Updates function signatures, class attributes, documentation, and all usages for consistency.

Improves code readability and reduces confusion about statistical parameters. (`fd0c06e`_)

* Add conventions documentation for variables and methods

This document outlines naming conventions and eigendecomposition methods used in the codebase. (`ca97cf1`_)

* Merge pull request #26 from nz-gravity/refactor_coarse_graining

refactor coarse graining (`60e7343`_)

* fixes to tests (`b5c7ac0`_)

* Refactor coarse graining functions and update tests

- Updated coarse graining function names for consistency: changed ``coarse_grain_multivar_fft`` to `apply_coarse_grain_multivar_fft`.
- Modified test cases to reflect changes in coarse graining configuration parameters from ``n_bins`` and ``n_freqs_per_bin`` to ``Nc`` and `Nh`.
- Removed obsolete test file ``test_coarse_grain.py`` and `test_multivar_blocked_noise_floor_unit.py`.
- Updated various tests to ensure compatibility with the new coarse graining structure and logging.
- Commented out plotting tests in ``test_plot_fitted_data.py`` to prevent execution during testing. (`05d58f7`_)

* Delete unused code (`ff704b0`_)

* Merge pull request #25 from nz-gravity/add_duration_to_multivar_lnl

add 1/T to multivar LnL (`ea29298`_)

* add 1/T (`13939d6`_)

* add welch (`10b62de`_)

* Fix typo (`8b7ff8a`_)

* Adds support for extra empirical PSD overlays in plots

Enables passing additional empirical PSD estimates—such as Welch overlays—to multivariate PSD matrix plots with customizable labels and styles. Improves attribute sanitization to prevent serialization issues with complex or non-serializable config objects, and adds unit tests for attribute filtering and plot overlay behavior. Enhances logging for effective sample size summaries and ensures robust handling of overlay data. (`27082ac`_)

* Simplify CLI (`fd09172`_)

* Add some runners. (`8b50491`_)

* Check semantic commit, if not, skip (`dce8398`_)

* improve docs (`d3efaed`_)

* Improves coarse binning to handle odd/even remainders

Updates coarse-graining logic to allow final bins to absorb frequency remainders while preserving odd bin sizes, eliminating divisibility constraints and improving flexibility. Adds corresponding tests for various remainder scenarios and expands CLI/config support for related options.

Enables more robust and general bin structure for spectral estimation workflows. (`8106069`_)

* add docs to follow math (`2d44d97`_)

* Add tex (`f5d757d`_)

* Merge pull request #24 from nz-gravity/linear_coarse_bins

Change coarse-binning to match linear-bins from paper draft (`db4fee1`_)

* fix coarse binning (`0e7abb7`_)

* coarse_grain: linear binning + midpoint reps + bin counts (`6a6f7af`_)

* add noise floor handling and aggregation script for LISA diagnostics (`6ec4c42`_)

* add energy diagnostics functions and utilities for multivariate PSD analysis (`b6f36fd`_)

* add noise floor controls to multivariate blocked NUTS sampler and update diagnostics (`a1ea558`_)

* add energy diagnostics module and integrate into run_all_diagnostics (`3ced440`_)

* add utility for inspecting delta3 variance scaling and generating diagnostics (`a8bfe14`_)

* add documentation for coarse graining in multivariate PSD pipeline (`0769cd0`_)

* add diagnostics (`c235f2f`_)

* refactor compute_ci_coverage_multivar to improve handling of complex PSD matrices (`04a73a5`_)

* allow faster diagnostic summary (`90ff4b2`_)

* add more multivar tests (var3) (`87bb4b7`_)

* add more lisa tests (`c0e6621`_)

* change LISA study settings (`89928f6`_)

* tests: add more tests (`d11bfbb`_)

* add matrix runs (`f09711f`_)

* Add eigenvalue separation diagnostics and preprocessing plots; enhance MCMC with preprocessing checks (`04fb02f`_)

* Enhance MCMC diagnostics with precomputed Rhat and ESS support; add optional PSIS computation toggle (`9aed2bd`_)

* Add numerical guards (`df2fde8`_)

* Speed up diagnostics (`71add51`_)

* Clip delta (Guard against delta underflow to 0.0 in float32 which makes log(delta)) (`69cda54`_)

* Add gradient clipping for VI (helps avoid NaNs when the ELBO has very steep regions) (`940f51d`_)

* Add VAR3 sim study (`3de2d2d`_)

* Print lowest ESS var (`aa9e056`_)

* add eigenvalue diagnostics (`bbbefc4`_)

* hacking on multivar case (`3e13f48`_)

* add more diagnostics warning for max tree depth (`90cbebc`_)

* add more tests (`7c7f0fe`_)

* remove timedomain moments from top (`e407824`_)

* improve tests (`fa618a7`_)

* run slow on CI (`e94ca19`_)

* add marker (`dc45907`_)

* cleanup tests (`7dd1161`_)

* speed up tests (`746d843`_)

* Merge pull request #22 from nz-gravity/diagnostics_refactoring

Refactor diagnostics modules (`7f4aff1`_)

* trapeizod to trapz (`00423d3`_)

* remove unused (`f394e9e`_)

* remove unused (`fac0fdf`_)

* remove unused (`ba018a3`_)

* fix autocorrelation computation (`ec4ca47`_)

* Add whitening diagnostics and refactor diagnostics modules

- Implemented a new module for whitening diagnostics based on autocorrelation in `whitening.py`.
- Created a new empty file ``whitening_diagnostics.py`` for future diagnostics.
- Updated ``diagnostics.py`` to integrate new diagnostics and streamline the summary generation process.
- Refactored the acceptance rate and Rhat calculations to utilize new diagnostics.
- Enhanced the PSD accuracy diagnostics to include RIAE and coverage metrics.
- Updated the VI initialization mixin to remove deprecated functions and integrate new diagnostics.
- Added a backward-compatible import wrapper for time-domain moment utilities in `time_domain_moments.py`.
- Refactored tests to validate the new diagnostics and ensure proper functionality.
- Added tests for time-domain moment computations to ensure accuracy and reliability. (`6aa5b80`_)

* more VI diagnostics (`fbb8387`_)

* add more diagnostics (`2e1aed0`_)

* hacking on lisa example (`d5a088c`_)

* add lisa example (`c139df8`_)

* add windowing (`b90ea79`_)

* add rhat (`6958493`_)

* fix scaling (Aagain) (`a57de7b`_)

* enhance scaling factor handling in VI adapters and improve test for PSD consistency (`b487b7f`_)

* Updated multivariate PSD rescaling to keep units consistent and added a dedicated regression test.

Adjusted src/log_psplines/samplers/multivar/multivar_base.py _rescale_psd to remove the global scaling_factor before applying channel standard deviations, aligning empirical PSDs with the observed/true scale.
Fixed src/log_psplines/arviz_utils/to_arviz.py periodogram handling to stop double-applying scaling_factor when Wishart PSDs are already scaled, while preserving the standardized-to-physical conversion for channel-standardized data.
Added tests/test_multivar_scaling.py, which builds a synthetic 2-channel VAR(1), checks the internal rescaling path, runs run_mcmc with blocked NUTS, and reports/ asserts posterior vs periodogram vs analytic PSD ratios.
Tests: pytest tests/test_multivar_scaling.py -k scaling -q (ArviZ shape warnings appear because only one chain is used, but the assertions pass).

Next steps: run the broader test suite if desired, and rerun with multiple chains to silence the ArviZ warning. (`8c8e96d`_)

* VI improvements (`5dfbc9c`_)

* Merge pull request #21 from nz-gravity/chains

Allow multiple chains (`36dbee0`_)

* add arviz saving mutliple chains (`80f5010`_)

* hackign on lisa multivar example (`3ab1220`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`4fdfdda`_)

* add files (`f95b84e`_)

* lisa (`b09aa82`_)

* add freq range check (`dc1b3f9`_)

* add more demos (`1034233`_)

* add more demos (`c0991d3`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`967cafe`_)

* add eeg plots (`e6c5dd7`_)

* add finance dataloader (`129187e`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`1030c62`_)

* Merge branch 'other_demos' into main (`d4f4749`_)

* more hacking (`62d453c`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`62b758b`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`2b4e933`_)

* add vscode (`3efc069`_)

* add improved multivar analysis (`4230090`_)

* fix VI scaling (`b921877`_)

* fix VI init (`f63f553`_)

* fix VI init (`5910de8`_)

* Merge pull request #20 from nz-gravity/allow_vi_standalone

allow VI standalone (`449e681`_)

* allow VI standalone (`c79f40e`_)

* Merge pull request #19 from nz-gravity/extend-run_mcmc-with-only_vi-parameter

Add variational-only execution mode (`66c3556`_)

* Add variational-only execution mode (`cffd8bd`_)

* fix scaling and one sided psd (`e554c4a`_)

* Merge pull request #17 from nz-gravity/testing_with_lisa

Add testing with LISA (`8a26726`_)

* add multivar study (`c0c10f4`_)

* add tests (`389a839`_)

* refactoring (`c141f26`_)

* refactoring (`3c53319`_)

* Merge pull request #16 from nz-gravity/multivar_coarse_fixes

Add multivar coarse fix (`fc23bf1`_)

* add fixes (`5938097`_)

* Add new knots

Coauthor: @pmat747 (`b6ada0b`_)

* add better output (`bbe0e4c`_)

* Merge branch 'add_multivar_coarse_grain' (`7dd4077`_)

* add lisa slurm (`790d32e`_)

* Merge pull request #15 from nz-gravity/add_multivar_coarse_grain

add multivar coarse grain lnl (`68c4925`_)

* add coarse grain lnl (`79d451b`_)

* Merge pull request #14 from nz-gravity/add_multivar_averaged

add averaged data for multivar case (`3a2815f`_)

* fix conftext (`ddf0450`_)

* add change time blocks (`9ffe960`_)

* save quantiles directly instead of saving individaul PSD samples (`92aaf96`_)

* remove z matrix (`592c055`_)

* add averaged (`9d5de29`_)

* fix tests (`09a87fd`_)

* Merge pull request #11 from nz-gravity/coarse_lnl

Add Coarse-graining (univar) (`e0a1daf`_)

* add functional coarse lnl (`3aa784e`_)

* start working on coarse-LnL (`2eb4b64`_)

* Merge pull request #10 from nz-gravity/save_vi_diagnostics_before_sampling

save VI plots at the start (`0d7bc92`_)

* save VI plots at the start (`0f46599`_)

* cleanup plotting (`11047d7`_)

* Increase settings (`79e7d33`_)

* Refator plotting a bit to reuse code for VI (`7f2be90`_)

* Add traceback (`35a155c`_)

* Add welch PSD (`c654281`_)

* autospectrum to PSD (`166aac2`_)

* Add gitbranch check (`2caa70f`_)

* Remove recomputation of ESS (`8a7ba42`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`104a256`_)

* hacking on the logger' (`d163705`_)

* add few logs for VI init (`2ecc397`_)

* fix PSD matrix --> real (`f5b3f26`_)

* add extra debugs (`34ca975`_)

* add logger (`6306b1c`_)

* fix logger (`36a26da`_)

* from print-->logger (`bf31899`_)

* Fix plotting error (`8fa9087`_)

* Add demo images (`dc9ee0b`_)

* Merge pull request #9 from nz-gravity/vi

Vi to init params (`41d564d`_)

* init with VI (`c64f5a5`_)

* refactoring vi (`27b0bd3`_)

* fix vi plotting (`b50d467`_)

* add vi (`35f6268`_)

* add more diagnostics for multivar case (`4d50468`_)

* add blocked (`193f278`_)

* add fixes for plotters (`633370e`_)

* Refactor NUTS samplers for shared blocks and log-phi reparam (`4c3c9e1`_)

* add txt file for faster simulation (`f867b51`_)

* refactor pspline-sampling into its own reusable block and use einsum instead of @ (`3c3ed56`_)

* testing speedup attempts (`c5041a1`_)

* addd blocking (`9cb304e`_)

* hacking on lisa demo (`0081ff6`_)

* add plotting of results (`34e4304`_)

* reduce slurm requirements (`c700da4`_)

* add result extractor (`e1ede71`_)

* refactored multivar sampler so its a bit easeir to read (`b47f356`_)

* hcking on lisa sim (`e8f9a1f`_)

* Add sim study slurm (`28d5d75`_)

* fix sampler scaling (`7e4c1cf`_)

* Merge pull request #7 from nz-gravity/nuts_lnz

add lnz computation for univar NUTS + MH (failing for multivar still) (`d56e0c0`_)

* starting to work on multivar LnZ computation (`247e24a`_)

* functioning morphZ lnZ computation (`aecc053`_)

* add lnz computation using MH posterior function (`85aea6f`_)

* improve latex description (`eb5389a`_)

* add caching (`a17a3c3`_)

* add GW tests (`b50131f`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD

Also ran pre-commit formatter (`f6a710d`_)

* add coverage check (`b2a3d0a`_)

* Add some docs (`7ba3ed4`_)

* fix 'duplicate channel key' arviz error, and add ci_coverage functions (`cd51f18`_)

* Remove old multivar study (`7bb6d29`_)

* increase test size (`c94e430`_)

* Add coherence plotter (`3e648ec`_)

* Add simulation study for multivar (`e42d8fd`_)

* fix psd sccaling (`5d7d721`_)

* adding PSD plotting for multivar (`6896728`_)

* add IAE (`4f38bfc`_)

* add a simple check for the 2d PSD nazeela (`07ddea1`_)

* Merge pull request #6 from nz-gravity/add_rescaling

add auto rescaling of data (`deb463a`_)

* add test (`64c0590`_)

* add auto rescaling of data (`541e8da`_)

* load posterior-psd from idata (`6310b6a`_)

* convert pdgrm to numpy from jax (`1e42487`_)

* Allow plotting with saved PSD in idata (`eb62a25`_)

* Add different scaling for datasets (`7fcf8f3`_)

* add plotting fixes (`64a0a47`_)

* remove dead function (`b2b9cd7`_)

* adjust 'slow' settings (`4d359f6`_)

* acceptance_rate to accept_prob (`28823c3`_)

* Add num-chains (`91ed697`_)

* Add extra-fields (`26cfab9`_)

* Ave log-posterior (`dfc6edd`_)

* Add simulated dataset (`aff0d18`_)

* Testing improvements (`9d70c82`_)

* remove dead code (`ab07e24`_)

* use run_mcmc for the multivar dataset (`1d8fa25`_)

* unified mcmc structure (`c2a915c`_)

* improve diagnostics (`fe2b728`_)

* refactor to_arviz for better maintainability (`299a5b8`_)

* simplify to_arviz interface (`f26fce2`_)

* remove sparcity comments (`853ca48`_)

* create unified structure for creation of inference_data (`8e52ca3`_)

* refactoring kwargs to run_mcmc (removed TypedDict of kwargs) (`bd0eeee`_)

* add idata saving (`0428c0f`_)

* Batch spline eval (`24721a1`_)

* Use plotting function (`8a17868`_)

* Allow different plotting scales (`ef8b862`_)

* refactor psd-matrix plotting (`08cae1a`_)

* some small speedups (`9765714`_)

* Merge pull request #5 from nz-gravity/adding_new_sampler_base

Add multivar PSD estimator (`0c9e226`_)

* add multivar (`4667e02`_)

* started adding new base sampler (`c4b0979`_)

* refactor location for multivar code (`87a6365`_)

* refactor location of samplers (`31b96ff`_)

* Add MultivariateLogPSplines class (`047dbf7`_)

* Add MultivarFFT and MultivariateTimeseries (`bd38705`_)

* refactor datatypes into new module (`fe45df5`_)

* Cant explicitly requested dtype <class 'jax.numpy.float64'>  -- users have to use JAX_ENABLE_X64 (`8282035`_)

* Add tqdm for reconstruction (`3ae8df2`_)

* Add varma dataset (`98733ab`_)

* fixed workflow (`6d7e21f`_)

* add multivar test (`3ec2933`_)

* add multivar example (`500b28d`_)

* add nsamples hack (`fd1ddfa`_)

* more hacking (`dee923f`_)

* add multivar PSD (`0acac8c`_)

* more hacking on multivar PSD (`74d41e3`_)

* start working on multivar demo (`f6f24cd`_)

* adding freq-grid for knot allocation (only knots at freq grid values) (`649f071`_)

* Merge pull request #4 from nz-gravity/add_morph_Lnz

add Morph-LnZ computation as an option (`241fa26`_)

* add Morph-LnZ computation as an option (`217d1d0`_)

* add better docs (`6f63046`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`c23ed06`_)

* Add patricio's knot allocation

Co-authored-by: Patricio Maturana-Russel <pmat747@users.noreply.github.com> (`2fdaf71`_)

* Improve type hints (`a2a1505`_)

* Add patricio's knot allocation

Co-authored-by: Patricio Maturana-Russel <pmat747@users.noreply.github.com> (`e091162`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`ffbb214`_)

* fix conflicts (`56337c6`_)

* qol changes (`40ae72d`_)

* add simulation study (`5f983b0`_)

* testing with higher diff matrix order (`ff0807f`_)

* Add info on penalty matrix in repr (`4be1039`_)

* Add simulation study files (`2c93927`_)

* Remove jax version fix (`3db9e55`_)

* Allow for higher penalty matrix (`bd61544`_)

* Print the model init (`778d9a2`_)

* Add a script to explor N-knots vs the IAE (`754cbea`_)

* Specify dtype for penalty matrix (`a2db6f5`_)

* Load posterior PSD from arviz, allow passing of path to inferenec objec (`b10cda8`_)

* Add units (`4ce0997`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`ec7c4b7`_)

* run precommits (`fe6bbf0`_)

* hacking on lvk knot loc (`47a11f5`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`94ab56e`_)

* pypi onl after pytest passes (`f865ac0`_)

* add LVK allocation (`4e50458`_)

* add more tests for PSD diagnostics (`7185750`_)

* add: add LVK code testing (`13ce77f`_)

* add basis comparison (`4610f02`_)

* add cut (`119a1f7`_)

* refactor docs to work with new API (`4286687`_)

* refactor preprocessing (`624a87b`_)

* add SVI testing (`d7d598d`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`86771ee`_)

* fix typo (`32f41e6`_)

* t push
Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`600387f`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`6d4dbc4`_)

* add benchmarking code (`aedfd5a`_)

* add ESS comparison (`9a98372`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`c9d3799`_)

* update readme links (`3f32876`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`4d042c1`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`6ec4029`_)

* Update pypi.yml (`939e855`_)

* edit readme (`537f8d4`_)

* add: add option for mh and nuts (`3d1208b`_)

* refactoring to use a common parent class (`3c6d879`_)

* change to just vanilla metropolis-hastings (get rid of covar matrix adaptation) (`8ca7f3d`_)

* Merge pull request #3 from nz-gravity/adding_adaptive_mcmc

Adding adaptive MCMC (`21e1be2`_)

* init (`18b2bfe`_)

* fix tests (`02de5a7`_)

* Update docs.yml (`affd908`_)

* Update README.rst (`b8f91d7`_)

* add line locator (`f451b80`_)

* add fix (`b8a94a2`_)

* refactor (`134c0cd`_)

* add docs (`474785e`_)

* add examples (`d8c0a68`_)

* add psd approx (`9143b01`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD (`ccc1f48`_)

* Create LICENSE (`3107114`_)

* fix readme (`0314ffd`_)

* add workflows (`c8e8f49`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD (`d7de401`_)

* Merge pull request #1 from avivajpeyi/pre-commit-ci-update-config

[pre-commit.ci] pre-commit autoupdate (`34951cf`_)

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci (`23c3aa0`_)

* [pre-commit.ci] pre-commit autoupdate

updates:
- [github.com/pre-commit/pre-commit-hooks: v4.5.0 → v5.0.0](https://github.com/pre-commit/pre-commit-hooks/compare/v4.5.0...v5.0.0)
- https://github.com/pre-commit/mirrors-isort → https://github.com/PyCQA/isort
- [github.com/PyCQA/isort: v5.10.1 → 6.0.1](https://github.com/PyCQA/isort/compare/v5.10.1...6.0.1)
- https://github.com/ambv/black → https://github.com/psf/black
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0)
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0) (`ff3b289`_)

* add welch psd (`d7121d6`_)

* add LVK plots (`f818caa`_)

* add LVK example and parametric model (`0666415`_)

* hackig on alternative model (`4197563`_)

* add LVK example (`922f870`_)

* add LVK example (`4944aa1`_)

* add lvk noise (`d93f36b`_)

* add tests (`c9e3c79`_)

* more hacking (`fda820d`_)

* add ci (`3539ffb`_)

* add whitepsace (`3274b74`_)

* hacking with Benjamin (`23210a3`_)

* init project packaging (`5685aac`_)

* improve knot allocation (`8e4ad33`_)

* optimise starting weights (`1942d60`_)

* generate data for testing (`0d619ce`_)

* start hacking (`cd4026f`_)

.. _7af67ce: https://github.com/nz-gravity/LogPSplinePSD/commit/7af67ce4bb02e69b5ff302fcbd7c234c10b05359
.. _09b80a9: https://github.com/nz-gravity/LogPSplinePSD/commit/09b80a9737ece14bbb9b8606e486635883a27e69
.. _c42299b: https://github.com/nz-gravity/LogPSplinePSD/commit/c42299b5aae21a34ce5077bd18430c4887d3244d
.. _ea1a087: https://github.com/nz-gravity/LogPSplinePSD/commit/ea1a0877d8af15aa70c834ce4a3d66b13b6248ff
.. _2c5d09d: https://github.com/nz-gravity/LogPSplinePSD/commit/2c5d09dd50326f74a676f57870a16e0654b29f2f
.. _08ad439: https://github.com/nz-gravity/LogPSplinePSD/commit/08ad439c68937d85ee95d3d4f0d73e1059b2c789
.. _f71d2f6: https://github.com/nz-gravity/LogPSplinePSD/commit/f71d2f6269cd9c8714052edf0b0ba73628b3a084
.. _598a80e: https://github.com/nz-gravity/LogPSplinePSD/commit/598a80ed1e8de7b9195f8db2541e02c74621b922
.. _bcec12d: https://github.com/nz-gravity/LogPSplinePSD/commit/bcec12d99c47f126e454473d3699fa32b83d8573
.. _2a4da4b: https://github.com/nz-gravity/LogPSplinePSD/commit/2a4da4b73f11c892a8da78edee8ecc2a9701cb6a
.. _d9a12cf: https://github.com/nz-gravity/LogPSplinePSD/commit/d9a12cff7fcb434235cca562cf708291cffe34f2
.. _eedaa20: https://github.com/nz-gravity/LogPSplinePSD/commit/eedaa206e9f1fcca52bb1fff78cdf40906ef58f0
.. _55f35de: https://github.com/nz-gravity/LogPSplinePSD/commit/55f35de18e3ccbc32ccb7a30aec9bae4253b88eb
.. _45f3b46: https://github.com/nz-gravity/LogPSplinePSD/commit/45f3b467a600f43e2ff181fb1debad9b8d5d7bfb
.. _5253690: https://github.com/nz-gravity/LogPSplinePSD/commit/5253690c9d0474a6b85db4ef837c3ba4589b5061
.. _b15f183: https://github.com/nz-gravity/LogPSplinePSD/commit/b15f183541e1be9a6e30e6c374806a55892651a2
.. _5e8185b: https://github.com/nz-gravity/LogPSplinePSD/commit/5e8185b37ae4de1783b2dc3acb8a96c1456e1785
.. _13e7416: https://github.com/nz-gravity/LogPSplinePSD/commit/13e7416874b3d29478294f6503c9655f6580e7ba
.. _37e2d20: https://github.com/nz-gravity/LogPSplinePSD/commit/37e2d20080ab7a0413c653a322813b6989c52fbc
.. _825ea13: https://github.com/nz-gravity/LogPSplinePSD/commit/825ea13a7b3b3abe61df806e1a98df3d8b94a069
.. _159d387: https://github.com/nz-gravity/LogPSplinePSD/commit/159d387c52a69bb26e0f8216febb247adfa9f3d2
.. _a6a45c0: https://github.com/nz-gravity/LogPSplinePSD/commit/a6a45c045b78c69ccb6e40263910069d0fc97834
.. _b5c37b6: https://github.com/nz-gravity/LogPSplinePSD/commit/b5c37b6cf244c2d33f5007223877be37e304a2f2
.. _fa742fa: https://github.com/nz-gravity/LogPSplinePSD/commit/fa742faec0e6c969e0ea0456f302a0b583e4a7db
.. _8c05631: https://github.com/nz-gravity/LogPSplinePSD/commit/8c0563131ade5c507eb90dba055bc192e5ffeb7c
.. _551685d: https://github.com/nz-gravity/LogPSplinePSD/commit/551685deeb3828178803d9c1ed479f7c5abaaf3d
.. _a9d0101: https://github.com/nz-gravity/LogPSplinePSD/commit/a9d01010f5e90ab8ef2def443bfdb12b3d5cb0e6
.. _6040a5c: https://github.com/nz-gravity/LogPSplinePSD/commit/6040a5c429749c72d088b06ac75807eafb2070ca
.. _6b73f16: https://github.com/nz-gravity/LogPSplinePSD/commit/6b73f1635dba98478d6f5e54ece91ed455775d84
.. _da8d428: https://github.com/nz-gravity/LogPSplinePSD/commit/da8d4289af932418d52d7208129fa58cba0a873c
.. _efdbcaf: https://github.com/nz-gravity/LogPSplinePSD/commit/efdbcafec3e14f4f9895ae59889c7a8f5ef0c070
.. _81c0bbb: https://github.com/nz-gravity/LogPSplinePSD/commit/81c0bbbdfd23ce00a06ecc1cc531114f589c65fe
.. _9df1e5d: https://github.com/nz-gravity/LogPSplinePSD/commit/9df1e5d7527d08602a4402cb038e88c8aa474128
.. _48aecf4: https://github.com/nz-gravity/LogPSplinePSD/commit/48aecf46d6d6064e4b5744c73525f9a302ec8cf1
.. _2d993d9: https://github.com/nz-gravity/LogPSplinePSD/commit/2d993d939be50cb038e48d0fc25e3eeb44eb4ba4
.. _bfa1388: https://github.com/nz-gravity/LogPSplinePSD/commit/bfa138891af4a67e7509b9589cc6c4f719965b52
.. _877e6c8: https://github.com/nz-gravity/LogPSplinePSD/commit/877e6c804248a40aa1be2a353b93a351967b8270
.. _0290b42: https://github.com/nz-gravity/LogPSplinePSD/commit/0290b4268296812f7216f9d3c2de97fa8186f9b9
.. _b4f97a5: https://github.com/nz-gravity/LogPSplinePSD/commit/b4f97a5c5da0d5f720ed623f6a81d0fd0107a42d
.. _f838433: https://github.com/nz-gravity/LogPSplinePSD/commit/f83843301dbe457becf90f00e370a6ef10d7677a
.. _4cf5f52: https://github.com/nz-gravity/LogPSplinePSD/commit/4cf5f52a05232f1afb7e14d1d543b0d79ec11ced
.. _6c4c175: https://github.com/nz-gravity/LogPSplinePSD/commit/6c4c1757ad2284b8bb4d3232db91c7cc332b3e82
.. _d0e06f6: https://github.com/nz-gravity/LogPSplinePSD/commit/d0e06f61ac6117166670fffb80028f2f69a2e7de
.. _d7d7355: https://github.com/nz-gravity/LogPSplinePSD/commit/d7d73557b9c286f6c11857025def9e60b9fe2b0e
.. _f03de7d: https://github.com/nz-gravity/LogPSplinePSD/commit/f03de7dee3c2ddddf256f8da0414c2c76da54fcf
.. _9e906dc: https://github.com/nz-gravity/LogPSplinePSD/commit/9e906dc4b821a60b1ae9a2b9b05c0376239921cb
.. _2837da2: https://github.com/nz-gravity/LogPSplinePSD/commit/2837da2541b1b1f23f767e5001e52a5fe67f8e28
.. _278441c: https://github.com/nz-gravity/LogPSplinePSD/commit/278441c995a1ebb460cb053954a06e67e2c19a9e
.. _0c518bf: https://github.com/nz-gravity/LogPSplinePSD/commit/0c518bf5df82db2f295d91a4f9aa256cf052f5c0
.. _62c196a: https://github.com/nz-gravity/LogPSplinePSD/commit/62c196aff27bd37e910a55521f84b610a9ec4306
.. _185e176: https://github.com/nz-gravity/LogPSplinePSD/commit/185e176503f522df62ad726d7306f1d696d3dcb9
.. _f6ef8ae: https://github.com/nz-gravity/LogPSplinePSD/commit/f6ef8ae79c8e6e52f10b8ba88601ec991b540851
.. _2aee2f3: https://github.com/nz-gravity/LogPSplinePSD/commit/2aee2f37bed4fd09d59f7070794592e12bb1b435
.. _34858e4: https://github.com/nz-gravity/LogPSplinePSD/commit/34858e4a88107f7487201dc437e688e13236022c
.. _616e469: https://github.com/nz-gravity/LogPSplinePSD/commit/616e469968749ae958ac33691befb52ddd29f709
.. _65e5c0a: https://github.com/nz-gravity/LogPSplinePSD/commit/65e5c0a19e6322f761e0e838cf7a8f27fc2746e0
.. _25b0cc4: https://github.com/nz-gravity/LogPSplinePSD/commit/25b0cc49741ca8b0b5b61c88b8008ab1754afca1
.. _89db68e: https://github.com/nz-gravity/LogPSplinePSD/commit/89db68e7e7419c72f0118f4ca6960774126d7928
.. _1903a98: https://github.com/nz-gravity/LogPSplinePSD/commit/1903a9822723db11ee8e449c21d2bea5756da964
.. _c9e6360: https://github.com/nz-gravity/LogPSplinePSD/commit/c9e6360fa7057f020efb4604c8dd65fb1b1455c2
.. _4390cd8: https://github.com/nz-gravity/LogPSplinePSD/commit/4390cd8aa854d6658ffd65b3eea3ed6e9f31ded4
.. _f118ecd: https://github.com/nz-gravity/LogPSplinePSD/commit/f118ecd851d6910c35530e45a5ca691ee9cb69f0
.. _662efed: https://github.com/nz-gravity/LogPSplinePSD/commit/662efed4c44d07c81afa70ef50614a075c3f2e6e
.. _0e95fb8: https://github.com/nz-gravity/LogPSplinePSD/commit/0e95fb8b39a1d5c2bf6239d5e15f75e956ce1955
.. _b8e58fd: https://github.com/nz-gravity/LogPSplinePSD/commit/b8e58fd8aaef0c0179499cf7b6a3c2482667626b
.. _72db98c: https://github.com/nz-gravity/LogPSplinePSD/commit/72db98c59e716faee0b6a3bec9943ff782b248ae
.. _cbbd2b7: https://github.com/nz-gravity/LogPSplinePSD/commit/cbbd2b7a173159b37a5303128a15fd6ea948e0ac
.. _4334af0: https://github.com/nz-gravity/LogPSplinePSD/commit/4334af0047fa93d17c1fd842ae48c790ac0933e4
.. _7515bd5: https://github.com/nz-gravity/LogPSplinePSD/commit/7515bd540df88d108e7bb976ce119cea828b81be
.. _0fa25a6: https://github.com/nz-gravity/LogPSplinePSD/commit/0fa25a6a9c32451fa93c8f7c6ba73a0a2f461acf
.. _b687361: https://github.com/nz-gravity/LogPSplinePSD/commit/b6873613b2cd386eeebb2160bfe191058e2bfa88
.. _52c4eba: https://github.com/nz-gravity/LogPSplinePSD/commit/52c4eba2e31956a4c69b5bb37d4c4eec9fa658d9
.. _1655acd: https://github.com/nz-gravity/LogPSplinePSD/commit/1655acdbf1c4e56cbb5d357cc1398174625827ce
.. _80d6afb: https://github.com/nz-gravity/LogPSplinePSD/commit/80d6afb1342bc834d70c19fdabd10a51ea883fb5
.. _5436178: https://github.com/nz-gravity/LogPSplinePSD/commit/543617807912a79bcf87d8a9e17ed89ce343c3a0
.. _6eb5f77: https://github.com/nz-gravity/LogPSplinePSD/commit/6eb5f7797c684ae9081e40a462aaf05f47fcc8d5
.. _54c62ef: https://github.com/nz-gravity/LogPSplinePSD/commit/54c62ef76ff1feb1bf84ceeaa153f1df57e984a4
.. _af1940f: https://github.com/nz-gravity/LogPSplinePSD/commit/af1940f687e45e9780b52ddcb14c54834ebc3ecb
.. _bfa242f: https://github.com/nz-gravity/LogPSplinePSD/commit/bfa242f5fcc664e5e6ac1d8d80c0a682d072977e
.. _4e58da9: https://github.com/nz-gravity/LogPSplinePSD/commit/4e58da96ae64cbe0bc1fd9bc2bafeb403cd86ebe
.. _1b477b2: https://github.com/nz-gravity/LogPSplinePSD/commit/1b477b29dc8ade65e6646bd5bec48c660d96f69e
.. _e58ed5e: https://github.com/nz-gravity/LogPSplinePSD/commit/e58ed5e8db4474f08ac15485b727404bb399a669
.. _4a63c49: https://github.com/nz-gravity/LogPSplinePSD/commit/4a63c495dddaf87f1d347439aa61b9a3fabaa765
.. _1391b04: https://github.com/nz-gravity/LogPSplinePSD/commit/1391b04d776906a6fd3681ec522f061f035c8407
.. _c8b2391: https://github.com/nz-gravity/LogPSplinePSD/commit/c8b2391b9d210e921eddcaf0b6d0aabd223d4916
.. _b0765bd: https://github.com/nz-gravity/LogPSplinePSD/commit/b0765bd161e11791acac9d5ee02f5246df7c5720
.. _95e0621: https://github.com/nz-gravity/LogPSplinePSD/commit/95e0621902320292005beb524f0f5b9b0bd15cee
.. _1f08ce2: https://github.com/nz-gravity/LogPSplinePSD/commit/1f08ce2c52d0ce0737dccc4d2acb58661d2dc929
.. _6768865: https://github.com/nz-gravity/LogPSplinePSD/commit/676886557a1c8ccae37441540a4660440fd57cc9
.. _ec291d8: https://github.com/nz-gravity/LogPSplinePSD/commit/ec291d85af0410108eab9c9412f64de5fa464f84
.. _c8890d0: https://github.com/nz-gravity/LogPSplinePSD/commit/c8890d0f0d3398447648b47526959f8836174d50
.. _bd1df91: https://github.com/nz-gravity/LogPSplinePSD/commit/bd1df91b7b83064b7a6f2cb3d66e8df092d72cb0
.. _10b35b4: https://github.com/nz-gravity/LogPSplinePSD/commit/10b35b4c1b88489ff98d62c769573724899c6a81
.. _527251c: https://github.com/nz-gravity/LogPSplinePSD/commit/527251cdd712b22479e7f562ac1256e72879f779
.. _22a51b2: https://github.com/nz-gravity/LogPSplinePSD/commit/22a51b226736c363f7d3838861dafac9afaa06b4
.. _21dc4b3: https://github.com/nz-gravity/LogPSplinePSD/commit/21dc4b36c9868595713931a73bea889d6890a512
.. _1b89dec: https://github.com/nz-gravity/LogPSplinePSD/commit/1b89dec56ebb51a4bd7d43cea8e31bf1e8d92b7c
.. _cd6d246: https://github.com/nz-gravity/LogPSplinePSD/commit/cd6d24618c5c2d7f86cba692585e1ea3a54e9d28
.. _0379145: https://github.com/nz-gravity/LogPSplinePSD/commit/0379145fd8a00f3885456e61899799fe2f824571
.. _360ffb0: https://github.com/nz-gravity/LogPSplinePSD/commit/360ffb0ad2ba5cd2f9c9bd7ffe472fc815e46737
.. _4447764: https://github.com/nz-gravity/LogPSplinePSD/commit/444776407e6640505fabea83e3d89fd72468e766
.. _fd3689c: https://github.com/nz-gravity/LogPSplinePSD/commit/fd3689c7221d72b69b871747e2bd8d2265338240
.. _bf4f8d0: https://github.com/nz-gravity/LogPSplinePSD/commit/bf4f8d0205eb8ad1c89ba81ec2c5ca5ff3207529
.. _26735ad: https://github.com/nz-gravity/LogPSplinePSD/commit/26735ad484e5e3f202c9096b4c4e0e69bc955c90
.. _30c28d5: https://github.com/nz-gravity/LogPSplinePSD/commit/30c28d523d7bd69b32e047c1da32ca8fa5c30651
.. _251cc4e: https://github.com/nz-gravity/LogPSplinePSD/commit/251cc4e38d46e8490a2b9e5eb9c199fa994f2e8a
.. _51768dc: https://github.com/nz-gravity/LogPSplinePSD/commit/51768dc08ece7d1357796fd96889ab11ba0cd6c9
.. _ee9a417: https://github.com/nz-gravity/LogPSplinePSD/commit/ee9a4171f5b0cd2d23dd370e1419598da26c6efd
.. _bea276d: https://github.com/nz-gravity/LogPSplinePSD/commit/bea276dde42b8113c4364c859dfbe2c4b7ee13b4
.. _67d3dde: https://github.com/nz-gravity/LogPSplinePSD/commit/67d3dde1069b41948691f4b17e90988f97d847d3
.. _033be92: https://github.com/nz-gravity/LogPSplinePSD/commit/033be924ada1de3fe24092adf79149f8b07d9067
.. _f8f7cfa: https://github.com/nz-gravity/LogPSplinePSD/commit/f8f7cfadc742be3efb96e51842bc0bb72beb0174
.. _01357db: https://github.com/nz-gravity/LogPSplinePSD/commit/01357db54538cf9bf35daf61cb45db953722fb50
.. _db45fa4: https://github.com/nz-gravity/LogPSplinePSD/commit/db45fa45ef42b32f462fedddb503e9a369939678
.. _4de5b5b: https://github.com/nz-gravity/LogPSplinePSD/commit/4de5b5b6698dab7c5423bddbac247e2dd4164aea
.. _eec8130: https://github.com/nz-gravity/LogPSplinePSD/commit/eec81300f4135f6ba757529e7edaa740a2de9089
.. _708105f: https://github.com/nz-gravity/LogPSplinePSD/commit/708105fcb4fd461c1ffa887a5d1cbae835111038
.. _24bfab2: https://github.com/nz-gravity/LogPSplinePSD/commit/24bfab261928d0d7dd99a887bee3aee53f1ef176
.. _4ce4cf0: https://github.com/nz-gravity/LogPSplinePSD/commit/4ce4cf00ad22d5a54a9224ed2adb65bae0a5d060
.. _f5fdb6f: https://github.com/nz-gravity/LogPSplinePSD/commit/f5fdb6f5d9268edc43ec75e9907dadcebf3ed451
.. _2fc0b75: https://github.com/nz-gravity/LogPSplinePSD/commit/2fc0b7527420835c2e14b2373a037351da960f32
.. _bd6c544: https://github.com/nz-gravity/LogPSplinePSD/commit/bd6c54424a03537ba65ddf789a9aec073a5e0c19
.. _f9936a9: https://github.com/nz-gravity/LogPSplinePSD/commit/f9936a9786b62a946f0e9b6e0a42cf2628bd85b7
.. _3919f28: https://github.com/nz-gravity/LogPSplinePSD/commit/3919f2897428dfc0f0d650f7c26af4949ce1253e
.. _6ad2cf4: https://github.com/nz-gravity/LogPSplinePSD/commit/6ad2cf4f3c950d888590e134549e32c0a0b945cb
.. _d4119c3: https://github.com/nz-gravity/LogPSplinePSD/commit/d4119c3c60339c099832da6df7441a569ac68e76
.. _f8b3725: https://github.com/nz-gravity/LogPSplinePSD/commit/f8b372577231cb35f4977ce057609b3d891a72cc
.. _309e75e: https://github.com/nz-gravity/LogPSplinePSD/commit/309e75eac28a555399f8e654bddfb00995b71a7d
.. _a4a28de: https://github.com/nz-gravity/LogPSplinePSD/commit/a4a28de9de381b6f5e2bb754230636ad26dbd1bf
.. _a9fc38d: https://github.com/nz-gravity/LogPSplinePSD/commit/a9fc38d2f3ede5f0e9ecc2842e7ef422ef4eb9c0
.. _9f2f10c: https://github.com/nz-gravity/LogPSplinePSD/commit/9f2f10c8360efa09b37eaead01bdaa2350dbdcc7
.. _7f50d88: https://github.com/nz-gravity/LogPSplinePSD/commit/7f50d88b44efec4ce265e40b8157bdb93f7fa985
.. _9e5ddd5: https://github.com/nz-gravity/LogPSplinePSD/commit/9e5ddd57468dd399d30d385b7d5449aeab688c54
.. _0daba3f: https://github.com/nz-gravity/LogPSplinePSD/commit/0daba3fa4496eb22f9849d5cb6436257df180ad0
.. _16d5449: https://github.com/nz-gravity/LogPSplinePSD/commit/16d5449b04180e17d03617b5a381527d2683cddd
.. _66659bc: https://github.com/nz-gravity/LogPSplinePSD/commit/66659bc8a8fdf65adb2301f29e8694064eaf269f
.. _b47719d: https://github.com/nz-gravity/LogPSplinePSD/commit/b47719d385ae81f4fcf376e7365dbd5a217bba94
.. _efe9227: https://github.com/nz-gravity/LogPSplinePSD/commit/efe9227741428cd11723846f1a0b4c3e47a1dda0
.. _aa685d4: https://github.com/nz-gravity/LogPSplinePSD/commit/aa685d48ed2b3d78bdc7ab155ce39aabc63bb68f
.. _6b1bd12: https://github.com/nz-gravity/LogPSplinePSD/commit/6b1bd1203a378c588682e11238fbd804d4d9fc74
.. _55787e3: https://github.com/nz-gravity/LogPSplinePSD/commit/55787e3d59023cd95f5c4468932e74e71822cf2a
.. _d807f24: https://github.com/nz-gravity/LogPSplinePSD/commit/d807f246260076005386a497fe86f90ea6f54e0f
.. _a0d80ae: https://github.com/nz-gravity/LogPSplinePSD/commit/a0d80aec5f4a023bf572276ae1a5f6a3f0613f08
.. _fb88156: https://github.com/nz-gravity/LogPSplinePSD/commit/fb88156833241a1f20f5d2ef650ce12c2e71b775
.. _34804bb: https://github.com/nz-gravity/LogPSplinePSD/commit/34804bbfaf4cc74094d91d760410bac933b99c21
.. _021538f: https://github.com/nz-gravity/LogPSplinePSD/commit/021538fbaa436dc1215acdb63a67f01edd165050
.. _e10d126: https://github.com/nz-gravity/LogPSplinePSD/commit/e10d126302e476843481b872df0e1d499c60c5c3
.. _cbf2f5d: https://github.com/nz-gravity/LogPSplinePSD/commit/cbf2f5d10cb9c1c750d86d1ad27c817b8967464a
.. _5436ff2: https://github.com/nz-gravity/LogPSplinePSD/commit/5436ff204e86ae44054af525fe397d127ebec1a3
.. _4d5fbc0: https://github.com/nz-gravity/LogPSplinePSD/commit/4d5fbc0a47a57a29a789acb4e5eeea5de7bb17dd
.. _d952114: https://github.com/nz-gravity/LogPSplinePSD/commit/d95211499a6f74682e590fcf72178d878b54d082
.. _8eec160: https://github.com/nz-gravity/LogPSplinePSD/commit/8eec160222d7a6360f559ffb6ee8541a0ff62d49
.. _692ab2e: https://github.com/nz-gravity/LogPSplinePSD/commit/692ab2ec4fa41a57ca1a1cbe121b7b05992b861d
.. _6dba63b: https://github.com/nz-gravity/LogPSplinePSD/commit/6dba63b5671e7c3305aac3ddee749f34edc89de8
.. _808d911: https://github.com/nz-gravity/LogPSplinePSD/commit/808d911ffcec50d608ec3219db3b7f99e95fd9f4
.. _bc67aab: https://github.com/nz-gravity/LogPSplinePSD/commit/bc67aabbd7be01f9361c525afd675300d09b8585
.. _a215fc6: https://github.com/nz-gravity/LogPSplinePSD/commit/a215fc63695d1b8a10d7155c112268443a377baa
.. _ecc9f7e: https://github.com/nz-gravity/LogPSplinePSD/commit/ecc9f7e9e3a01afb8b0707262c3ed1dffa00f53f
.. _caae21c: https://github.com/nz-gravity/LogPSplinePSD/commit/caae21c799b840540f35171336dc777d175608ff
.. _f4d83b5: https://github.com/nz-gravity/LogPSplinePSD/commit/f4d83b508ec67e74c888d03173eee413f5c38b84
.. _be27aeb: https://github.com/nz-gravity/LogPSplinePSD/commit/be27aebb63ac96b6d9f4fd3fdd11e8f5331d054a
.. _6610919: https://github.com/nz-gravity/LogPSplinePSD/commit/6610919ce741a47824a75b4956e32f40c44ef9bb
.. _407fe61: https://github.com/nz-gravity/LogPSplinePSD/commit/407fe615c45ce7e64e1b4257f64fa82c2c8eb28d
.. _6961c35: https://github.com/nz-gravity/LogPSplinePSD/commit/6961c355e161d4edc48ee4e2f6c02c491f0cd67d
.. _ad61b63: https://github.com/nz-gravity/LogPSplinePSD/commit/ad61b63b3f9a1472f47fccd5cbc40a4b47765991
.. _962c74a: https://github.com/nz-gravity/LogPSplinePSD/commit/962c74ad91a3a4888553b455beb725bc82673145
.. _d4fdf0e: https://github.com/nz-gravity/LogPSplinePSD/commit/d4fdf0e53eafbfe9b137f00fd0cca9dd8277151c
.. _6263611: https://github.com/nz-gravity/LogPSplinePSD/commit/62636112faef4cb1fc746a44751ada1eb8a3a4e7
.. _52f687c: https://github.com/nz-gravity/LogPSplinePSD/commit/52f687cd3dd38cdbaf0c42226526a1ca87b15d83
.. _8c9c6a0: https://github.com/nz-gravity/LogPSplinePSD/commit/8c9c6a0974961a546102c1dd23b85b3dc82ceaf3
.. _3eb0069: https://github.com/nz-gravity/LogPSplinePSD/commit/3eb0069c065496b551889b24911050380409c4bf
.. _57750b9: https://github.com/nz-gravity/LogPSplinePSD/commit/57750b99998f47759b311b93b44a15a21227b425
.. _804ec1e: https://github.com/nz-gravity/LogPSplinePSD/commit/804ec1e626947a83e9f285e25567df49ed0e9a12
.. _cdcd610: https://github.com/nz-gravity/LogPSplinePSD/commit/cdcd610eae1c03cf8903a45172bd09bcaee62f6a
.. _a570d64: https://github.com/nz-gravity/LogPSplinePSD/commit/a570d64316ed67d1658ba359c030a035810c27d4
.. _6f6a7f7: https://github.com/nz-gravity/LogPSplinePSD/commit/6f6a7f7873e058f5d9faeabf0bd62a5f882fd4df
.. _baeb884: https://github.com/nz-gravity/LogPSplinePSD/commit/baeb884a0951ed58d451a5ff95ced933b62a6f2b
.. _59440bd: https://github.com/nz-gravity/LogPSplinePSD/commit/59440bdb5805b22058bd332ed009e1a99490cc8b
.. _246d57b: https://github.com/nz-gravity/LogPSplinePSD/commit/246d57b933e2ba25327d05e7fb9fd17fd03baf55
.. _51c268a: https://github.com/nz-gravity/LogPSplinePSD/commit/51c268a35c584912b7bf196543321d5af9907355
.. _7266799: https://github.com/nz-gravity/LogPSplinePSD/commit/7266799581f4e6d610ebae106c3faa826d6162d6
.. _58d6ed4: https://github.com/nz-gravity/LogPSplinePSD/commit/58d6ed4a8b8daea79980c1450c8af39d6a0223c7
.. _a9977dd: https://github.com/nz-gravity/LogPSplinePSD/commit/a9977dd77bf071aa8742c53192027884e2eadaaf
.. _7d07ce4: https://github.com/nz-gravity/LogPSplinePSD/commit/7d07ce4294a062ee9126fc8236d0ac4afed57ed6
.. _7ae7535: https://github.com/nz-gravity/LogPSplinePSD/commit/7ae7535b72ea3f9a6913f10f0f735430180600c6
.. _cc6f34d: https://github.com/nz-gravity/LogPSplinePSD/commit/cc6f34dadedcdae97b66fbbb1fe94ad34870b8a8
.. _1a9c2a9: https://github.com/nz-gravity/LogPSplinePSD/commit/1a9c2a931ec741f4493f8f053812966096abc299
.. _bf0f497: https://github.com/nz-gravity/LogPSplinePSD/commit/bf0f497ffd24d6ace11c4effa64f827170595bee
.. _65e26f7: https://github.com/nz-gravity/LogPSplinePSD/commit/65e26f797905b2402ff16dd58be347271fd7b231
.. _976d515: https://github.com/nz-gravity/LogPSplinePSD/commit/976d515edbdf3ad5ebfbd7ed01d35203086fbc6c
.. _4982305: https://github.com/nz-gravity/LogPSplinePSD/commit/4982305e52d23d43cec4c450f147bb839eae9a3c
.. _0db59a5: https://github.com/nz-gravity/LogPSplinePSD/commit/0db59a5566dc7ac23a28dfd116107ed8e956e2f8
.. _f0e6cb4: https://github.com/nz-gravity/LogPSplinePSD/commit/f0e6cb40fc7f81d98b4d55c5e8f3596ca4ee756c
.. _7472888: https://github.com/nz-gravity/LogPSplinePSD/commit/74728886f25438288694a4bf0ab639dd5d7af4b7
.. _ede4b1f: https://github.com/nz-gravity/LogPSplinePSD/commit/ede4b1f755246d273fd0227e79a46fbcb7366fa1
.. _ff5fdbb: https://github.com/nz-gravity/LogPSplinePSD/commit/ff5fdbbb593370c577f59fabcf42ddb29caa9381
.. _6be8807: https://github.com/nz-gravity/LogPSplinePSD/commit/6be880732192fd090cf4b924e42f2a2087b69cb0
.. _56fa75e: https://github.com/nz-gravity/LogPSplinePSD/commit/56fa75e1d865a4061698d87510d158de054240ca
.. _1d69109: https://github.com/nz-gravity/LogPSplinePSD/commit/1d691096979e8e070de623aa42dd9da47b689b4e
.. _0b6e053: https://github.com/nz-gravity/LogPSplinePSD/commit/0b6e0535401a6838d8ae27d8650d0c7bd2495910
.. _3fefb46: https://github.com/nz-gravity/LogPSplinePSD/commit/3fefb462c6d965ef00c6bf2350120b080f867daa
.. _4ab79be: https://github.com/nz-gravity/LogPSplinePSD/commit/4ab79be74ecd0954db2c6b46182af497e2399488
.. _31a685f: https://github.com/nz-gravity/LogPSplinePSD/commit/31a685f6b04282888564e44a6f50071a7eec2560
.. _cd198ea: https://github.com/nz-gravity/LogPSplinePSD/commit/cd198eae83ddf69f3eba0c0382cc2a26c5f55143
.. _1f0336a: https://github.com/nz-gravity/LogPSplinePSD/commit/1f0336a258d2e15f72692db816452dcda39d216f
.. _8103f1d: https://github.com/nz-gravity/LogPSplinePSD/commit/8103f1d1d7780ec8dc18cb8851a635309fa07d22
.. _45047ac: https://github.com/nz-gravity/LogPSplinePSD/commit/45047ac2e13c597472117a9aa28859e4e6090cb4
.. _f020e3f: https://github.com/nz-gravity/LogPSplinePSD/commit/f020e3fa978aeb698c5670ec2f41a4b1fa460f8d
.. _daf5507: https://github.com/nz-gravity/LogPSplinePSD/commit/daf550765f8ac760ed7c4d66d9a55151e20d957d
.. _61f1635: https://github.com/nz-gravity/LogPSplinePSD/commit/61f1635e6297c5e02c715eb7a3369f67ca6a00b8
.. _2a3e72a: https://github.com/nz-gravity/LogPSplinePSD/commit/2a3e72ac0bfc698eb4aaf94df1d86efa06e8cee3
.. _b82a327: https://github.com/nz-gravity/LogPSplinePSD/commit/b82a327b6fa15a989e69baf42e2484457f65f727
.. _ff5e8d7: https://github.com/nz-gravity/LogPSplinePSD/commit/ff5e8d72614d2516814fce7b370e63df29053e76
.. _4ae2430: https://github.com/nz-gravity/LogPSplinePSD/commit/4ae243098010ea93789f49a91f5faf7958035894
.. _8522040: https://github.com/nz-gravity/LogPSplinePSD/commit/85220401569f1c80e8aa79bba000be2e71ddc6d5
.. _6a9763e: https://github.com/nz-gravity/LogPSplinePSD/commit/6a9763edfa5abefc15162ee66f6146c97f438d05
.. _254e135: https://github.com/nz-gravity/LogPSplinePSD/commit/254e135e838c69fa80aeee4275b53b66cfc0b749
.. _1fdb89a: https://github.com/nz-gravity/LogPSplinePSD/commit/1fdb89ac22684336ae3db5878aad214f6b36a701
.. _dae452a: https://github.com/nz-gravity/LogPSplinePSD/commit/dae452a03a252091fec4c4d53535e62e98139d49
.. _45e2baf: https://github.com/nz-gravity/LogPSplinePSD/commit/45e2baf7d6a40267b6b14408a7a579362a25e7ca
.. _0f0da2c: https://github.com/nz-gravity/LogPSplinePSD/commit/0f0da2c5cf9a186b9e2e16b3aa948490e4d57ca5
.. _2fb7c24: https://github.com/nz-gravity/LogPSplinePSD/commit/2fb7c244b2c2aa004c1df46ed8d723788ce82eae
.. _df536b3: https://github.com/nz-gravity/LogPSplinePSD/commit/df536b3e4f21ebc99ebe702ae34254c073606a97
.. _3488eeb: https://github.com/nz-gravity/LogPSplinePSD/commit/3488eeb362551fd058c2e0e1aba1bc8a7b583037
.. _45c38c7: https://github.com/nz-gravity/LogPSplinePSD/commit/45c38c787f8553a3c8fcb9de8c0ac52adbc19a76
.. _8faa1ac: https://github.com/nz-gravity/LogPSplinePSD/commit/8faa1acfada7da10d5b2e0391cbad9c07db9a28f
.. _79f3063: https://github.com/nz-gravity/LogPSplinePSD/commit/79f30633c8edafae383194c986be5a965ef5940a
.. _4b9aafe: https://github.com/nz-gravity/LogPSplinePSD/commit/4b9aafea6291a71d763bad9cde12edc3786bcf3a
.. _340ed5e: https://github.com/nz-gravity/LogPSplinePSD/commit/340ed5e75f8e084d562db96f097d099a8e325820
.. _2e98313: https://github.com/nz-gravity/LogPSplinePSD/commit/2e983131470b09bd154374c56e5e4de7751875e7
.. _be30185: https://github.com/nz-gravity/LogPSplinePSD/commit/be30185851ca7e8309c0e550ad0aaa853ae7ab0c
.. _4bc22bc: https://github.com/nz-gravity/LogPSplinePSD/commit/4bc22bccc956dfd6c18070371d6512867b7fe1ba
.. _c2dede8: https://github.com/nz-gravity/LogPSplinePSD/commit/c2dede89582c07e011258f51dfbf75a470ead392
.. _cd1ff15: https://github.com/nz-gravity/LogPSplinePSD/commit/cd1ff1512559769c5b1d3243a5454c39c77d492c
.. _2b9f675: https://github.com/nz-gravity/LogPSplinePSD/commit/2b9f6757da031441e79b4dfb967bf24a72e8ff67
.. _ec5f505: https://github.com/nz-gravity/LogPSplinePSD/commit/ec5f505d4d900cec9237c8563af3d1a21c5cb34c
.. _0fd0481: https://github.com/nz-gravity/LogPSplinePSD/commit/0fd04813a9be0c32a38a71c641ffd60840407560
.. _c7f67bc: https://github.com/nz-gravity/LogPSplinePSD/commit/c7f67bcf0d971cf4bd6ef53f71afcddeb7f85bd6
.. _116f59a: https://github.com/nz-gravity/LogPSplinePSD/commit/116f59acfc58b84d8fc5fb37c3f851d92dc5a3db
.. _7b91d50: https://github.com/nz-gravity/LogPSplinePSD/commit/7b91d505ccbd17f1478e9e1912eb443334cd9e3f
.. _26af587: https://github.com/nz-gravity/LogPSplinePSD/commit/26af5872d617f63d381f9832f6876e50cd561722
.. _5ff2a29: https://github.com/nz-gravity/LogPSplinePSD/commit/5ff2a29dcfbd84d7faab1e3e2538a66c9fa20074
.. _aa35d05: https://github.com/nz-gravity/LogPSplinePSD/commit/aa35d052d42cc99d50f013207a7b9c51c6988d2e
.. _d8b16e9: https://github.com/nz-gravity/LogPSplinePSD/commit/d8b16e915b10c9f7358c54fa86915a1ddd105d22
.. _bab7745: https://github.com/nz-gravity/LogPSplinePSD/commit/bab774594eeed346de4b9b1ddabba42f2cc01822
.. _ed82fcf: https://github.com/nz-gravity/LogPSplinePSD/commit/ed82fcf06c563bc903b975e769c5b9ceb9e1dcfa
.. _b599c15: https://github.com/nz-gravity/LogPSplinePSD/commit/b599c150a6dd810a54f912e07a3133c3d6969503
.. _103684d: https://github.com/nz-gravity/LogPSplinePSD/commit/103684d90b909975a97529121ce90f7b127acf48
.. _120e901: https://github.com/nz-gravity/LogPSplinePSD/commit/120e9011eaea37eba468e14a640a8b5360bbf9e6
.. _6b4bb27: https://github.com/nz-gravity/LogPSplinePSD/commit/6b4bb2724f7e47709ee53c2bb8c5501674fe03e4
.. _349a2f7: https://github.com/nz-gravity/LogPSplinePSD/commit/349a2f7d172d4b59fc8a95356c656e74aa098787
.. _7ec854d: https://github.com/nz-gravity/LogPSplinePSD/commit/7ec854daad981b7e042f714d018be98f731b52bf
.. _f740a20: https://github.com/nz-gravity/LogPSplinePSD/commit/f740a20c87e855d56771845f80d76a3f35eb6892
.. _6471534: https://github.com/nz-gravity/LogPSplinePSD/commit/6471534808a1bb246bf536e7c3ead664719fb2cf
.. _d032bfb: https://github.com/nz-gravity/LogPSplinePSD/commit/d032bfbcec8f6978618357a21d4766f624f5a410
.. _b17556f: https://github.com/nz-gravity/LogPSplinePSD/commit/b17556ff3eb96dedbc139acbf2ee9333a3349fed
.. _350fe2c: https://github.com/nz-gravity/LogPSplinePSD/commit/350fe2c28eb8de07053ef56cc260ed1950ee4db3
.. _5dc8dad: https://github.com/nz-gravity/LogPSplinePSD/commit/5dc8dadef3308591700822e996f966490dd3b8a1
.. _7159d09: https://github.com/nz-gravity/LogPSplinePSD/commit/7159d09ada6fcdbe49b0a26d76c206bc1ad8c476
.. _b16a42a: https://github.com/nz-gravity/LogPSplinePSD/commit/b16a42aa4253b94c0375b681d20763b98cad8ad3
.. _51a29d6: https://github.com/nz-gravity/LogPSplinePSD/commit/51a29d67ec7acc85d37e30ddb1f572daeae93751
.. _e181b63: https://github.com/nz-gravity/LogPSplinePSD/commit/e181b6312cc6588605250a5141dc710b67ad1cea
.. _7839d8a: https://github.com/nz-gravity/LogPSplinePSD/commit/7839d8a769002eb9bb2c602602f3ac7d49396de2
.. _845c919: https://github.com/nz-gravity/LogPSplinePSD/commit/845c91969708aaea11dcf236ed4555e79bda0171
.. _8f9afcc: https://github.com/nz-gravity/LogPSplinePSD/commit/8f9afcc89bdd5761a3bf1e3f741933c91d32cae1
.. _c2e0cb1: https://github.com/nz-gravity/LogPSplinePSD/commit/c2e0cb1d9342809657c3f9dedbb0e255c97cb453
.. _2ad3756: https://github.com/nz-gravity/LogPSplinePSD/commit/2ad3756f25443f92f5dcd5c91a56760608273f6f
.. _64ea085: https://github.com/nz-gravity/LogPSplinePSD/commit/64ea085d5b7c663a7a23e7ed9a0ca78658fd841a
.. _cd7914e: https://github.com/nz-gravity/LogPSplinePSD/commit/cd7914e35f2376bee545ca2a76b0cdac00b3d5a1
.. _065de15: https://github.com/nz-gravity/LogPSplinePSD/commit/065de15a3cfce2017727e189d441b772a50deb9d
.. _bb5f7d9: https://github.com/nz-gravity/LogPSplinePSD/commit/bb5f7d986dd616209a08d0d0001c05a491e4c2a3
.. _95ca9a8: https://github.com/nz-gravity/LogPSplinePSD/commit/95ca9a827cdb511ab803255e7d05d7c7ce27c15f
.. _cbe644b: https://github.com/nz-gravity/LogPSplinePSD/commit/cbe644b2bc877b191761697e09247388c63f894f
.. _76094ab: https://github.com/nz-gravity/LogPSplinePSD/commit/76094ab897840f90112b3db48bdd4e4fc5045da8
.. _a83a366: https://github.com/nz-gravity/LogPSplinePSD/commit/a83a366cf9dc8ec35919dab73e32087cef81d020
.. _55e51f1: https://github.com/nz-gravity/LogPSplinePSD/commit/55e51f12ed5e1236a62399e3ef34a89c11d1e5c2
.. _fd0c06e: https://github.com/nz-gravity/LogPSplinePSD/commit/fd0c06eef97c17899077ad9b3c190290fb328f8d
.. _ca97cf1: https://github.com/nz-gravity/LogPSplinePSD/commit/ca97cf18e9a6f1fe3f2e08f7c1a2701191655b62
.. _60e7343: https://github.com/nz-gravity/LogPSplinePSD/commit/60e7343257ffd44b826b09c6679c9994c1a60f37
.. _b5c7ac0: https://github.com/nz-gravity/LogPSplinePSD/commit/b5c7ac007f2865d30837b3c854a43e47ae976ec4
.. _05d58f7: https://github.com/nz-gravity/LogPSplinePSD/commit/05d58f7033d49baf534eacab737ecd9a860f2283
.. _ff704b0: https://github.com/nz-gravity/LogPSplinePSD/commit/ff704b0d25608bcf4c87113f14bf04182289ecad
.. _ea29298: https://github.com/nz-gravity/LogPSplinePSD/commit/ea29298f74c5e57367dc342429d2e7564e6d3dbc
.. _13939d6: https://github.com/nz-gravity/LogPSplinePSD/commit/13939d61e023f288fd0a481aa8108d6a53a830b2
.. _10b62de: https://github.com/nz-gravity/LogPSplinePSD/commit/10b62de75bc9ce627b758ecc883beb5c9de6473f
.. _8b7ff8a: https://github.com/nz-gravity/LogPSplinePSD/commit/8b7ff8aba9d14c192e70a318da7b94ee1732487e
.. _27082ac: https://github.com/nz-gravity/LogPSplinePSD/commit/27082ac8201e0ac5f902a2e5b4bebd21dacdcd7c
.. _fd09172: https://github.com/nz-gravity/LogPSplinePSD/commit/fd09172829f4a7cc5fcdbc5655b84c99a6fa36ee
.. _8b50491: https://github.com/nz-gravity/LogPSplinePSD/commit/8b50491c77093473d07c0d38f9c49432188ce8f7
.. _dce8398: https://github.com/nz-gravity/LogPSplinePSD/commit/dce83986291d6fbb94ef49339f8221d52a1da932
.. _d3efaed: https://github.com/nz-gravity/LogPSplinePSD/commit/d3efaedf6eee56b89d0cbfac4c07930de54ed3ab
.. _8106069: https://github.com/nz-gravity/LogPSplinePSD/commit/81060696a49b9c97145fb5e6f4e620539d07abb1
.. _2d44d97: https://github.com/nz-gravity/LogPSplinePSD/commit/2d44d971b32f40e04f3700d1b429b9b2e9e45fcc
.. _f5d757d: https://github.com/nz-gravity/LogPSplinePSD/commit/f5d757dc8c8357c2721708b87c047e6b0b679ec0
.. _db4fee1: https://github.com/nz-gravity/LogPSplinePSD/commit/db4fee1373ce84fa4d789df848c2e91b8fef25e4
.. _0e7abb7: https://github.com/nz-gravity/LogPSplinePSD/commit/0e7abb755089f813410210dc15bcbe452e57f004
.. _6a6f7af: https://github.com/nz-gravity/LogPSplinePSD/commit/6a6f7af8a7b15125078766a72dcf90982edce70a
.. _6ec4c42: https://github.com/nz-gravity/LogPSplinePSD/commit/6ec4c4245a323c7e6e0b86e0831c6bd2014aa27a
.. _b6f36fd: https://github.com/nz-gravity/LogPSplinePSD/commit/b6f36fdf3b60b734341e86e7e42de20366d765ac
.. _a1ea558: https://github.com/nz-gravity/LogPSplinePSD/commit/a1ea558dd416594115648dada3e3b2ddb296b283
.. _3ced440: https://github.com/nz-gravity/LogPSplinePSD/commit/3ced4406225233ed8f4be23fc529ee825bc646b5
.. _a8bfe14: https://github.com/nz-gravity/LogPSplinePSD/commit/a8bfe14311f377f7679b26e019457037c1b7a2f2
.. _0769cd0: https://github.com/nz-gravity/LogPSplinePSD/commit/0769cd089153419cf49f8597fe08cb262c0184df
.. _c235f2f: https://github.com/nz-gravity/LogPSplinePSD/commit/c235f2fc07b9d8d8ff23e993caf0930254f02b9c
.. _04a73a5: https://github.com/nz-gravity/LogPSplinePSD/commit/04a73a5932fb3246b50cc4e0ac21d8ce4b4175c3
.. _90ff4b2: https://github.com/nz-gravity/LogPSplinePSD/commit/90ff4b28c2f9b716cfd70d6bc36b84c404fbd354
.. _87bb4b7: https://github.com/nz-gravity/LogPSplinePSD/commit/87bb4b7cb1249676a13200eb81bd37ac93b84ea4
.. _c0e6621: https://github.com/nz-gravity/LogPSplinePSD/commit/c0e66217f76f71b31809a1e0290675beb87fe36a
.. _89928f6: https://github.com/nz-gravity/LogPSplinePSD/commit/89928f64456b59784da7aa2a09069dd751e66567
.. _d11bfbb: https://github.com/nz-gravity/LogPSplinePSD/commit/d11bfbb3fe3d3be92619f00704034f4882c50f16
.. _f09711f: https://github.com/nz-gravity/LogPSplinePSD/commit/f09711fae20513bdba7458bd28a3946d98f0e55b
.. _04fb02f: https://github.com/nz-gravity/LogPSplinePSD/commit/04fb02f39f73d26083e037f2bc7b27a7161c82f9
.. _9aed2bd: https://github.com/nz-gravity/LogPSplinePSD/commit/9aed2bd238a86fbd025fef1429e89a7ebdb86a04
.. _df2fde8: https://github.com/nz-gravity/LogPSplinePSD/commit/df2fde8d8e9e450b425b989d2b0026bff14ec2db
.. _71add51: https://github.com/nz-gravity/LogPSplinePSD/commit/71add5161d0a81b15010ae7a7dd4649ac1b097a0
.. _69cda54: https://github.com/nz-gravity/LogPSplinePSD/commit/69cda5495b07b73f132f033ec295ff54a8a4ae8b
.. _940f51d: https://github.com/nz-gravity/LogPSplinePSD/commit/940f51db26cc199c9f0a28e473d2afa7e84c149c
.. _3de2d2d: https://github.com/nz-gravity/LogPSplinePSD/commit/3de2d2d606fc6980e9be5c34db8e8235ab32a455
.. _aa9e056: https://github.com/nz-gravity/LogPSplinePSD/commit/aa9e05603db478319c881da0829fc5db24a21e31
.. _bbbefc4: https://github.com/nz-gravity/LogPSplinePSD/commit/bbbefc469ebf1f8a0032408531409b8b72835d0e
.. _3e13f48: https://github.com/nz-gravity/LogPSplinePSD/commit/3e13f481fc437f01cfa7ff0ae8ec12a220e46b1d
.. _90cbebc: https://github.com/nz-gravity/LogPSplinePSD/commit/90cbebc809aedde51c1d45dfd643b19c2e427135
.. _7c7f0fe: https://github.com/nz-gravity/LogPSplinePSD/commit/7c7f0fe98a14b533e266fb4f14143ce2840734c7
.. _e407824: https://github.com/nz-gravity/LogPSplinePSD/commit/e4078247a9eedaff53f6359035f817cd9704976b
.. _fa618a7: https://github.com/nz-gravity/LogPSplinePSD/commit/fa618a717f6dab372981587ea644cbb7849f19a5
.. _e94ca19: https://github.com/nz-gravity/LogPSplinePSD/commit/e94ca1937e6a74cba150d3a4fce6ed1398b7f59e
.. _dc45907: https://github.com/nz-gravity/LogPSplinePSD/commit/dc45907e980a524885346f2db51e06dcfd6a51f4
.. _7dd1161: https://github.com/nz-gravity/LogPSplinePSD/commit/7dd1161e440f3972cff845b63e584ae7c0e04d1e
.. _746d843: https://github.com/nz-gravity/LogPSplinePSD/commit/746d843dd3663741a7588e3a8bb39d73fcce2242
.. _7f4aff1: https://github.com/nz-gravity/LogPSplinePSD/commit/7f4aff16e07b04d5cc9ba3e9e5517fd982f23241
.. _00423d3: https://github.com/nz-gravity/LogPSplinePSD/commit/00423d3e096312878b91752894c48a8f189c3f0d
.. _f394e9e: https://github.com/nz-gravity/LogPSplinePSD/commit/f394e9e936850911b121617f7dcaec4eed634223
.. _fac0fdf: https://github.com/nz-gravity/LogPSplinePSD/commit/fac0fdf581a817252f9bf768ae90a21c8b49af77
.. _ba018a3: https://github.com/nz-gravity/LogPSplinePSD/commit/ba018a345609b20f20a87aff2ed85dee9ef6176b
.. _ec4ca47: https://github.com/nz-gravity/LogPSplinePSD/commit/ec4ca4789ac6edbafd75bbd3efee841f3e020c09
.. _6aa5b80: https://github.com/nz-gravity/LogPSplinePSD/commit/6aa5b80a9885203b549a309870bffb7e1690c059
.. _fbb8387: https://github.com/nz-gravity/LogPSplinePSD/commit/fbb8387583ee9b0f2c0f08d1507aae17b9ff10b6
.. _2e1aed0: https://github.com/nz-gravity/LogPSplinePSD/commit/2e1aed07b5889bd4798514df02608a9e2e5bb450
.. _d5a088c: https://github.com/nz-gravity/LogPSplinePSD/commit/d5a088c2ff35570bbab22262fac3651bd48c2727
.. _c139df8: https://github.com/nz-gravity/LogPSplinePSD/commit/c139df85b706e19494c2585a9d08c16860e92b83
.. _b90ea79: https://github.com/nz-gravity/LogPSplinePSD/commit/b90ea79492ba13dba28911b33065575fc4b6a818
.. _6958493: https://github.com/nz-gravity/LogPSplinePSD/commit/6958493d3858c50b724aa016599529caf94af856
.. _a57de7b: https://github.com/nz-gravity/LogPSplinePSD/commit/a57de7b8b96298cb2a38097cd93615e2beafcbb8
.. _b487b7f: https://github.com/nz-gravity/LogPSplinePSD/commit/b487b7f653e0c9ffad095808e4a8c66e9be1773e
.. _8c8e96d: https://github.com/nz-gravity/LogPSplinePSD/commit/8c8e96dbd1054716206b2a60ae0270b5e69009de
.. _5dfbc9c: https://github.com/nz-gravity/LogPSplinePSD/commit/5dfbc9ce87b3790701f313862f91c409a7cffa2c
.. _36dbee0: https://github.com/nz-gravity/LogPSplinePSD/commit/36dbee0b9aebb1621504ddfa5e4f9ec392829ac9
.. _80f5010: https://github.com/nz-gravity/LogPSplinePSD/commit/80f50104a040d7c2152d9f70bce914b99dc5da38
.. _3ab1220: https://github.com/nz-gravity/LogPSplinePSD/commit/3ab122097f93f2f713381b5a625945be95cac3b3
.. _4fdfdda: https://github.com/nz-gravity/LogPSplinePSD/commit/4fdfddaac3ddffb35eb7cb349fafad2808e1cb0f
.. _f95b84e: https://github.com/nz-gravity/LogPSplinePSD/commit/f95b84e9ad67e39f92716a4243078e65d4318594
.. _b09aa82: https://github.com/nz-gravity/LogPSplinePSD/commit/b09aa82d328e29740dacf3d1d690dcad41be2c85
.. _dc1b3f9: https://github.com/nz-gravity/LogPSplinePSD/commit/dc1b3f9e7831e1e6302d39dc90c4986614944082
.. _1034233: https://github.com/nz-gravity/LogPSplinePSD/commit/10342337acfb921d06e41c1125fbc8805c19dfdc
.. _c0991d3: https://github.com/nz-gravity/LogPSplinePSD/commit/c0991d3863ab388cbb360b817819b26d26a5fd5f
.. _967cafe: https://github.com/nz-gravity/LogPSplinePSD/commit/967cafe5718bec705af17e0552f8eeb7661e6adb
.. _e6c5dd7: https://github.com/nz-gravity/LogPSplinePSD/commit/e6c5dd7b9ab056ca07cded759a3ccfaf54cda249
.. _129187e: https://github.com/nz-gravity/LogPSplinePSD/commit/129187e74b7b7b3a177e893fd8ea125550a0f18c
.. _1030c62: https://github.com/nz-gravity/LogPSplinePSD/commit/1030c62bb3667cbfdf1cc3343ef9292e825560c3
.. _d4f4749: https://github.com/nz-gravity/LogPSplinePSD/commit/d4f47498dd8074e2f59f294b761b83b39fa8dab5
.. _62d453c: https://github.com/nz-gravity/LogPSplinePSD/commit/62d453c0b0e6dbe62a01cf252d75a45c34a6012b
.. _62b758b: https://github.com/nz-gravity/LogPSplinePSD/commit/62b758b2ff8fe9ea25c94912655801a615357c0f
.. _2b4e933: https://github.com/nz-gravity/LogPSplinePSD/commit/2b4e9336168fadf4fcb246e50dd0e7a3838193eb
.. _3efc069: https://github.com/nz-gravity/LogPSplinePSD/commit/3efc069da1a147ef9083ef93baed1df22b320f1c
.. _4230090: https://github.com/nz-gravity/LogPSplinePSD/commit/4230090dd7ab4167de2af49cdf149361cd57ee29
.. _b921877: https://github.com/nz-gravity/LogPSplinePSD/commit/b921877192ef24215531c39c57e83118f462c1c7
.. _f63f553: https://github.com/nz-gravity/LogPSplinePSD/commit/f63f553bcf00ff8e9ff45f92b3098ec7b65b884f
.. _5910de8: https://github.com/nz-gravity/LogPSplinePSD/commit/5910de8d5eb8b8d3e76b217440e4adb473856640
.. _449e681: https://github.com/nz-gravity/LogPSplinePSD/commit/449e68129bc205441f3dd05a653481a002718280
.. _c79f40e: https://github.com/nz-gravity/LogPSplinePSD/commit/c79f40e60ad181aeb5d8fc59f573d4819b24cbe9
.. _66c3556: https://github.com/nz-gravity/LogPSplinePSD/commit/66c3556c7face5bfbc1b80197f20af01d91efb6c
.. _cffd8bd: https://github.com/nz-gravity/LogPSplinePSD/commit/cffd8bdad70a7e4763a9b44f6ad830af7d299a6c
.. _e554c4a: https://github.com/nz-gravity/LogPSplinePSD/commit/e554c4af595240502a8d8ac91558b03990d9cbf0
.. _8a26726: https://github.com/nz-gravity/LogPSplinePSD/commit/8a2672690cc43049def0e05dd983365fa1d2f4f8
.. _c0c10f4: https://github.com/nz-gravity/LogPSplinePSD/commit/c0c10f40a511f91710782d9f5e5b1011393777a2
.. _389a839: https://github.com/nz-gravity/LogPSplinePSD/commit/389a839c90ff42645ad9be3042b269bef6ba90be
.. _c141f26: https://github.com/nz-gravity/LogPSplinePSD/commit/c141f26fc3ff25824e646cd2f334af106e35de47
.. _3c53319: https://github.com/nz-gravity/LogPSplinePSD/commit/3c53319a2da609276b59dc2b6b573c828ae716c0
.. _fc23bf1: https://github.com/nz-gravity/LogPSplinePSD/commit/fc23bf16c38d4bd4e8a8a91d4f410751cafbfc6b
.. _5938097: https://github.com/nz-gravity/LogPSplinePSD/commit/5938097026bd86e4a3d1ce6fed71361406875458
.. _b6ada0b: https://github.com/nz-gravity/LogPSplinePSD/commit/b6ada0b6a134a5cfbc14bdd11b5dc6e0652027de
.. _bbe0e4c: https://github.com/nz-gravity/LogPSplinePSD/commit/bbe0e4cc8cd07f67aa8dca37997c22ee33acbaf4
.. _7dd4077: https://github.com/nz-gravity/LogPSplinePSD/commit/7dd4077f4b0a665959350328715d934f423fe201
.. _790d32e: https://github.com/nz-gravity/LogPSplinePSD/commit/790d32ef117de7835332264d4021b7affc21d646
.. _68c4925: https://github.com/nz-gravity/LogPSplinePSD/commit/68c4925dcdfecfdb6ec95266ccbf6c936b43e4e6
.. _79d451b: https://github.com/nz-gravity/LogPSplinePSD/commit/79d451b37ad44e7db9ea2f56af4e6b7c53e75dd3
.. _3a2815f: https://github.com/nz-gravity/LogPSplinePSD/commit/3a2815f1f2a7951c22276cd0008d3eff2efb4b07
.. _ddf0450: https://github.com/nz-gravity/LogPSplinePSD/commit/ddf04501f8f00589574612ab41d66abd2657f73d
.. _9ffe960: https://github.com/nz-gravity/LogPSplinePSD/commit/9ffe960cf90c1a73eda36c9388f693e61072de14
.. _92aaf96: https://github.com/nz-gravity/LogPSplinePSD/commit/92aaf96fc9a429c9941c684a9996cd6865d43c61
.. _592c055: https://github.com/nz-gravity/LogPSplinePSD/commit/592c055d121145efcb4f6f7db0304596b34ae472
.. _9d5de29: https://github.com/nz-gravity/LogPSplinePSD/commit/9d5de29d545886c5c7fb20052c9ed2a019544d77
.. _09a87fd: https://github.com/nz-gravity/LogPSplinePSD/commit/09a87fd0ce1281fbd4d8b4325139b8d61a6c583b
.. _e0a1daf: https://github.com/nz-gravity/LogPSplinePSD/commit/e0a1daf758cb283b0dc8242cd55ca94324107935
.. _3aa784e: https://github.com/nz-gravity/LogPSplinePSD/commit/3aa784ec2d38d4750c2d8fdb7a5860b77d450a31
.. _2eb4b64: https://github.com/nz-gravity/LogPSplinePSD/commit/2eb4b6426f3325672c961ebf5ff3fe72993f2544
.. _0d7bc92: https://github.com/nz-gravity/LogPSplinePSD/commit/0d7bc92ae91fe362fdffc997336323cf0f9a3f50
.. _0f46599: https://github.com/nz-gravity/LogPSplinePSD/commit/0f4659929ba47459f83c7e1794a43d0d8660ac34
.. _11047d7: https://github.com/nz-gravity/LogPSplinePSD/commit/11047d7effb24696323dc2ee94b84fab7faed624
.. _79e7d33: https://github.com/nz-gravity/LogPSplinePSD/commit/79e7d337bab54bf946a4dee83424a8cb7e9ba729
.. _7f2be90: https://github.com/nz-gravity/LogPSplinePSD/commit/7f2be908d3c69269ff32a8522dfd0136c335952c
.. _35a155c: https://github.com/nz-gravity/LogPSplinePSD/commit/35a155c5efc2631948c44a8f73256fd240b2dc31
.. _c654281: https://github.com/nz-gravity/LogPSplinePSD/commit/c6542814593d742e97bb88392b828d94934e8b46
.. _166aac2: https://github.com/nz-gravity/LogPSplinePSD/commit/166aac2c609f1879e904e2b98ee2352bf07c2ef9
.. _2caa70f: https://github.com/nz-gravity/LogPSplinePSD/commit/2caa70f40418d47ae7d6107ce36abfe140ba09c6
.. _8a7ba42: https://github.com/nz-gravity/LogPSplinePSD/commit/8a7ba42f58ff8e8b239f38e64de7da7b1ffb13d1
.. _104a256: https://github.com/nz-gravity/LogPSplinePSD/commit/104a256b93229ad263bb053fcbc87ec3e8d32bbc
.. _d163705: https://github.com/nz-gravity/LogPSplinePSD/commit/d1637052951ed8a7da5ae090ad0824493293253f
.. _2ecc397: https://github.com/nz-gravity/LogPSplinePSD/commit/2ecc397d5b1b05c25de9fb40b093b9dd72492dce
.. _f5b3f26: https://github.com/nz-gravity/LogPSplinePSD/commit/f5b3f269e18dd14e045e0efae2004d751fbe724b
.. _34ca975: https://github.com/nz-gravity/LogPSplinePSD/commit/34ca975026a6bd1dd659b5afa84917e1bf78f746
.. _6306b1c: https://github.com/nz-gravity/LogPSplinePSD/commit/6306b1c490aa25ac8ca1d520689ec23af3967fff
.. _36a26da: https://github.com/nz-gravity/LogPSplinePSD/commit/36a26daaeeda66297807496ddcd365332e48dc03
.. _bf31899: https://github.com/nz-gravity/LogPSplinePSD/commit/bf318993845ecbaf2b51a2b349db9510e898f5aa
.. _8fa9087: https://github.com/nz-gravity/LogPSplinePSD/commit/8fa908729047ed409471cabb4a913f5e82536672
.. _dc9ee0b: https://github.com/nz-gravity/LogPSplinePSD/commit/dc9ee0bb4d106569bbb179bc24a607a463dcc131
.. _41d564d: https://github.com/nz-gravity/LogPSplinePSD/commit/41d564ddb4f50da118a181eadf8a66799eb4da0d
.. _c64f5a5: https://github.com/nz-gravity/LogPSplinePSD/commit/c64f5a59e006e99638767db87777c5937553b5aa
.. _27b0bd3: https://github.com/nz-gravity/LogPSplinePSD/commit/27b0bd33176bb1298df395f9e35159964bcd95dd
.. _b50d467: https://github.com/nz-gravity/LogPSplinePSD/commit/b50d467b00a503574179840f5f3f104fa64399fe
.. _35f6268: https://github.com/nz-gravity/LogPSplinePSD/commit/35f6268f2dbd0ade8b714194eceba4d6c5247003
.. _4d50468: https://github.com/nz-gravity/LogPSplinePSD/commit/4d50468223bf0894dddabef74a1cefd48ffaae38
.. _193f278: https://github.com/nz-gravity/LogPSplinePSD/commit/193f2780b745524e3110631c49ea9ceab19e2d38
.. _633370e: https://github.com/nz-gravity/LogPSplinePSD/commit/633370e4a5270d08065b5b6089ff01b5e1484cd2
.. _4c3c9e1: https://github.com/nz-gravity/LogPSplinePSD/commit/4c3c9e1e23037d1882e5d96b0cb6804a0607d182
.. _f867b51: https://github.com/nz-gravity/LogPSplinePSD/commit/f867b5171c33000b57cb28314313420d211a004c
.. _3c3ed56: https://github.com/nz-gravity/LogPSplinePSD/commit/3c3ed5699c23c417fc08fcfc51f970ddafdeeaeb
.. _c5041a1: https://github.com/nz-gravity/LogPSplinePSD/commit/c5041a1298e15b007b2d1e82e06753676a0b88d9
.. _9cb304e: https://github.com/nz-gravity/LogPSplinePSD/commit/9cb304e6814efb11643d8bfa41c52a2d6371345d
.. _0081ff6: https://github.com/nz-gravity/LogPSplinePSD/commit/0081ff665755157be17e1a409c6578e40724b83f
.. _34e4304: https://github.com/nz-gravity/LogPSplinePSD/commit/34e4304c0f6a5cbe37606bc7bacc48a1f9e8d1b9
.. _c700da4: https://github.com/nz-gravity/LogPSplinePSD/commit/c700da4d5312c17108ff009e40da9c1aa9fcea75
.. _e1ede71: https://github.com/nz-gravity/LogPSplinePSD/commit/e1ede715778c1d30673e8fe5486a8b37b9f01620
.. _b47f356: https://github.com/nz-gravity/LogPSplinePSD/commit/b47f3566876c3722512e4f1b371980b7d81f76e3
.. _e8f9a1f: https://github.com/nz-gravity/LogPSplinePSD/commit/e8f9a1f979ead5afc29fb808fd273f53e22afcd2
.. _28d5d75: https://github.com/nz-gravity/LogPSplinePSD/commit/28d5d75911bc346381c8e15f9a5309d7f35fc554
.. _7e4c1cf: https://github.com/nz-gravity/LogPSplinePSD/commit/7e4c1cf9ee01ee96013612edd18fac1d41c9f4e8
.. _d56e0c0: https://github.com/nz-gravity/LogPSplinePSD/commit/d56e0c0fc79b4c5efb088ea923b28df818916192
.. _247e24a: https://github.com/nz-gravity/LogPSplinePSD/commit/247e24af7074a42edc2c4208759ff7ec50f7485d
.. _aecc053: https://github.com/nz-gravity/LogPSplinePSD/commit/aecc053d6ab162c0ccabe27edbfe7aff493c13ce
.. _85aea6f: https://github.com/nz-gravity/LogPSplinePSD/commit/85aea6f088a77b66f9314611ad48aae33065e405
.. _eb5389a: https://github.com/nz-gravity/LogPSplinePSD/commit/eb5389ac627caa32ac6f1c22d6baaafe8d7332e5
.. _a17a3c3: https://github.com/nz-gravity/LogPSplinePSD/commit/a17a3c392b151bb61dac58cee4261fb72036355b
.. _b50131f: https://github.com/nz-gravity/LogPSplinePSD/commit/b50131ff4b3215ca68cd8217c0d76263b446197b
.. _f6a710d: https://github.com/nz-gravity/LogPSplinePSD/commit/f6a710dfe2584960d50dc31dc1c1af6c021f0e64
.. _b2a3d0a: https://github.com/nz-gravity/LogPSplinePSD/commit/b2a3d0a88e3d6a292c5cbd1730b5093903a55586
.. _7ba3ed4: https://github.com/nz-gravity/LogPSplinePSD/commit/7ba3ed4b622e2371df05a584ebced21ff244233b
.. _cd51f18: https://github.com/nz-gravity/LogPSplinePSD/commit/cd51f18b8e05af33501c17cfb7c281effa6e70f6
.. _7bb6d29: https://github.com/nz-gravity/LogPSplinePSD/commit/7bb6d298ca6a97045044ac3ca05e8f35f6f25438
.. _c94e430: https://github.com/nz-gravity/LogPSplinePSD/commit/c94e4306f9444622b37a3f78b983dfd4de713d08
.. _3e648ec: https://github.com/nz-gravity/LogPSplinePSD/commit/3e648ec5bf6e3179d882c5d51a4568dc40dbafb2
.. _e42d8fd: https://github.com/nz-gravity/LogPSplinePSD/commit/e42d8fd63e9b5cd837a23669858ef41adadef38c
.. _5d7d721: https://github.com/nz-gravity/LogPSplinePSD/commit/5d7d72170d961fe3d85305c23b141cc8dbb71307
.. _6896728: https://github.com/nz-gravity/LogPSplinePSD/commit/68967284c9ab6796d9add25482036b918a9fd792
.. _4f38bfc: https://github.com/nz-gravity/LogPSplinePSD/commit/4f38bfcc12af2d22c0c8ae0300623d3725a7c447
.. _07ddea1: https://github.com/nz-gravity/LogPSplinePSD/commit/07ddea16688c60700d68bbe4314905dae8d26c0f
.. _deb463a: https://github.com/nz-gravity/LogPSplinePSD/commit/deb463a7f7e2c77d715a0100430327fb8796cba0
.. _64c0590: https://github.com/nz-gravity/LogPSplinePSD/commit/64c05909d4b8a78e594a2092154aa4dc60365af0
.. _541e8da: https://github.com/nz-gravity/LogPSplinePSD/commit/541e8da8c12d18546f4ed2f350c699ae31707b2d
.. _6310b6a: https://github.com/nz-gravity/LogPSplinePSD/commit/6310b6a2ff0f8039d3dc4ebe22db93ea55eb0d91
.. _1e42487: https://github.com/nz-gravity/LogPSplinePSD/commit/1e42487d080c651d324fb41e3b094cbd9e2dfa3b
.. _eb62a25: https://github.com/nz-gravity/LogPSplinePSD/commit/eb62a25ea2a8ad1aa94b96d5e9421b5149ee28d2
.. _7fcf8f3: https://github.com/nz-gravity/LogPSplinePSD/commit/7fcf8f35a21074fe271e3fc192498c78e0a1e932
.. _64a0a47: https://github.com/nz-gravity/LogPSplinePSD/commit/64a0a476e19432d0bfbd82f74c88cd8162f5bc5a
.. _b2b9cd7: https://github.com/nz-gravity/LogPSplinePSD/commit/b2b9cd7ad6c3fcb65e6798de7296194869f3436e
.. _4d359f6: https://github.com/nz-gravity/LogPSplinePSD/commit/4d359f61f7a9ed36110740aace843fc086e45194
.. _28823c3: https://github.com/nz-gravity/LogPSplinePSD/commit/28823c3e0778a4ba168edb63708c9ec7caca6b1e
.. _91ed697: https://github.com/nz-gravity/LogPSplinePSD/commit/91ed69726f5378e9f832144d0d3f8cd80b922143
.. _26cfab9: https://github.com/nz-gravity/LogPSplinePSD/commit/26cfab9b3c89483c843b6e4878cdc895f0a550ef
.. _dfc6edd: https://github.com/nz-gravity/LogPSplinePSD/commit/dfc6eddc8f29a0ecf3a84b5fe55429c5081f0207
.. _aff0d18: https://github.com/nz-gravity/LogPSplinePSD/commit/aff0d18863a755fe9a15c36b4f7fbcdbb705b00b
.. _9d70c82: https://github.com/nz-gravity/LogPSplinePSD/commit/9d70c82708c77c654446ead2d824d90d026671f5
.. _ab07e24: https://github.com/nz-gravity/LogPSplinePSD/commit/ab07e249f261b615c8f0274190e3dfb3f4c9d81b
.. _1d8fa25: https://github.com/nz-gravity/LogPSplinePSD/commit/1d8fa25053055d6d8bebecffe8d0d42fc75b9827
.. _c2a915c: https://github.com/nz-gravity/LogPSplinePSD/commit/c2a915c3dc0b3a5b2213b2fc43a43a368677c25b
.. _fe2b728: https://github.com/nz-gravity/LogPSplinePSD/commit/fe2b728088a596f75a32ee20a61f3623b152f850
.. _299a5b8: https://github.com/nz-gravity/LogPSplinePSD/commit/299a5b892c4eb6d0d4f876526c087d1765c537f0
.. _f26fce2: https://github.com/nz-gravity/LogPSplinePSD/commit/f26fce2709f5deb1d139b0380a21262224970409
.. _853ca48: https://github.com/nz-gravity/LogPSplinePSD/commit/853ca48ef7a97323d2e39739ea0031ea2c29bdd6
.. _8e52ca3: https://github.com/nz-gravity/LogPSplinePSD/commit/8e52ca3e74ed7232c05e942224f6311ae9f334ca
.. _bd0eeee: https://github.com/nz-gravity/LogPSplinePSD/commit/bd0eeee4d95801f0938e486917cf5ab194e0fb02
.. _0428c0f: https://github.com/nz-gravity/LogPSplinePSD/commit/0428c0f3088f8de570f05718202f0044be119a74
.. _24721a1: https://github.com/nz-gravity/LogPSplinePSD/commit/24721a191d1dbb8719000e7d6d0e1d87975ec4e5
.. _8a17868: https://github.com/nz-gravity/LogPSplinePSD/commit/8a17868c7fca31bc2867503a07880846f32a1f03
.. _ef8b862: https://github.com/nz-gravity/LogPSplinePSD/commit/ef8b86298cf5f0ecf70c72b550e4b8f748037a7d
.. _08cae1a: https://github.com/nz-gravity/LogPSplinePSD/commit/08cae1abe74e70a53c8bbce92e5001c71e9f2b6e
.. _9765714: https://github.com/nz-gravity/LogPSplinePSD/commit/9765714bb0f287620e86bcdadbcbd5cb9e3168ca
.. _0c9e226: https://github.com/nz-gravity/LogPSplinePSD/commit/0c9e226aeef1257a709b1f2d24cd3873317c1f76
.. _4667e02: https://github.com/nz-gravity/LogPSplinePSD/commit/4667e02bbf0362f725e64fdc573db7bf06154dba
.. _c4b0979: https://github.com/nz-gravity/LogPSplinePSD/commit/c4b097906c93036d606ae6956431216851b761c7
.. _87a6365: https://github.com/nz-gravity/LogPSplinePSD/commit/87a636589edbb83a0c1229f5232a22664dd09ebc
.. _31b96ff: https://github.com/nz-gravity/LogPSplinePSD/commit/31b96ffe1b0fd6aeb9a0bef9248b6868e61336a7
.. _047dbf7: https://github.com/nz-gravity/LogPSplinePSD/commit/047dbf7baa8ae1d6c2234a4782c4ad97b50d2815
.. _bd38705: https://github.com/nz-gravity/LogPSplinePSD/commit/bd38705e06b6f87d3e33939da8752141b5e5f240
.. _fe45df5: https://github.com/nz-gravity/LogPSplinePSD/commit/fe45df5f9ef304241f6ecd190e177696d601b42f
.. _8282035: https://github.com/nz-gravity/LogPSplinePSD/commit/8282035f5ae3f4aef546dea27a311cae5f5725a4
.. _3ae8df2: https://github.com/nz-gravity/LogPSplinePSD/commit/3ae8df2c841222b0b686f4ef3672d05c5cd035cd
.. _98733ab: https://github.com/nz-gravity/LogPSplinePSD/commit/98733ab8c71fba1e2d4acbe55c9610ed15c81e2f
.. _6d7e21f: https://github.com/nz-gravity/LogPSplinePSD/commit/6d7e21f1e0a599a5b88e37998c31899954dc833b
.. _3ec2933: https://github.com/nz-gravity/LogPSplinePSD/commit/3ec29336ab279da48339a206ddc886e9b8ad08c0
.. _500b28d: https://github.com/nz-gravity/LogPSplinePSD/commit/500b28d037b9caff6fc40b5b8c07c612201447f7
.. _fd1ddfa: https://github.com/nz-gravity/LogPSplinePSD/commit/fd1ddfa1144ebe652ba95f5f3009dbe82faa4d3c
.. _dee923f: https://github.com/nz-gravity/LogPSplinePSD/commit/dee923f7683d52274ec67bfdc9f36c6eeda2afcb
.. _0acac8c: https://github.com/nz-gravity/LogPSplinePSD/commit/0acac8ca339707812806ba551ca611cda7a5cc8b
.. _74d41e3: https://github.com/nz-gravity/LogPSplinePSD/commit/74d41e323e302f22efd68a719a1141d870cada93
.. _f6f24cd: https://github.com/nz-gravity/LogPSplinePSD/commit/f6f24cdd885e948879f692802abfa6f028fdacc0
.. _649f071: https://github.com/nz-gravity/LogPSplinePSD/commit/649f071b8414866c8385f76344d2992687db2ee2
.. _241fa26: https://github.com/nz-gravity/LogPSplinePSD/commit/241fa261b7c7bb7045490b181ab02ab96b83ffd7
.. _217d1d0: https://github.com/nz-gravity/LogPSplinePSD/commit/217d1d0d7881d687233abaa4e8c86cbe3a16027a
.. _6f63046: https://github.com/nz-gravity/LogPSplinePSD/commit/6f6304684cd044ed6be0c8c179c273b366b0cb36
.. _c23ed06: https://github.com/nz-gravity/LogPSplinePSD/commit/c23ed06a13e4bc3cfb79b9beb611da89b6e458fb
.. _2fdaf71: https://github.com/nz-gravity/LogPSplinePSD/commit/2fdaf71b160d7192650cb69cce1a7ce52bc8943a
.. _a2a1505: https://github.com/nz-gravity/LogPSplinePSD/commit/a2a150515c92ef1717a6b7beb5b80cc1c24d8c89
.. _e091162: https://github.com/nz-gravity/LogPSplinePSD/commit/e09116228565f411b2c57d685be0beadddd11072
.. _ffbb214: https://github.com/nz-gravity/LogPSplinePSD/commit/ffbb2142b29a84412b74dbab55881185cbe97c32
.. _56337c6: https://github.com/nz-gravity/LogPSplinePSD/commit/56337c67d2264862d0414264935f075405ccde77
.. _40ae72d: https://github.com/nz-gravity/LogPSplinePSD/commit/40ae72d53f1c3bcb04e33cd8bad4a32c98c41f27
.. _5f983b0: https://github.com/nz-gravity/LogPSplinePSD/commit/5f983b01f2060c5750b23de2d3859a4bf7a070f3
.. _ff0807f: https://github.com/nz-gravity/LogPSplinePSD/commit/ff0807f5b9524a1c98ce56670b61d8e4ba48409c
.. _4be1039: https://github.com/nz-gravity/LogPSplinePSD/commit/4be1039bab91a2bfd8b0000d6ca438c226945420
.. _2c93927: https://github.com/nz-gravity/LogPSplinePSD/commit/2c939275acdbcedc992c1350640cbd66a0438a6b
.. _3db9e55: https://github.com/nz-gravity/LogPSplinePSD/commit/3db9e5554af8d098bdb82269ef25bc634e0d00b7
.. _bd61544: https://github.com/nz-gravity/LogPSplinePSD/commit/bd615445b0d18f54029cb2d788fe2b618e4b45af
.. _778d9a2: https://github.com/nz-gravity/LogPSplinePSD/commit/778d9a228147bec1471aa7843abf3328478a5556
.. _754cbea: https://github.com/nz-gravity/LogPSplinePSD/commit/754cbea21dbca377196946b1b5268b1b474c4af9
.. _a2db6f5: https://github.com/nz-gravity/LogPSplinePSD/commit/a2db6f5003c3a627068b4110ddd5f70dfbf575cb
.. _b10cda8: https://github.com/nz-gravity/LogPSplinePSD/commit/b10cda83532470de467ed3042d61cc6902ae3c02
.. _4ce0997: https://github.com/nz-gravity/LogPSplinePSD/commit/4ce0997b5da1f24cecfd70afd2a15c4bbff9c717
.. _ec7c4b7: https://github.com/nz-gravity/LogPSplinePSD/commit/ec7c4b7ad7a4565589d6f888949a29eb07c91ac4
.. _fe6bbf0: https://github.com/nz-gravity/LogPSplinePSD/commit/fe6bbf0183152d18197854ed02e9004435b3c85d
.. _47a11f5: https://github.com/nz-gravity/LogPSplinePSD/commit/47a11f5ef46a28c4884fbdccbc87dcbb1ff43088
.. _94ab56e: https://github.com/nz-gravity/LogPSplinePSD/commit/94ab56e0407566e3104b11c04cf2f519f6e67346
.. _f865ac0: https://github.com/nz-gravity/LogPSplinePSD/commit/f865ac030d0dd15f655f64d822efca99d2a35877
.. _4e50458: https://github.com/nz-gravity/LogPSplinePSD/commit/4e50458307ef0dddabb5f60c064235c8de7edcad
.. _7185750: https://github.com/nz-gravity/LogPSplinePSD/commit/71857509db9af2ebf2cae094917208c95c45909a
.. _13ce77f: https://github.com/nz-gravity/LogPSplinePSD/commit/13ce77f42ceea075ff89dcde68c64e9921ca7dcc
.. _4610f02: https://github.com/nz-gravity/LogPSplinePSD/commit/4610f024234d6c246e55e5f0f53efe997703576c
.. _119a1f7: https://github.com/nz-gravity/LogPSplinePSD/commit/119a1f743289f93e73d0e43176a8f70360749e59
.. _4286687: https://github.com/nz-gravity/LogPSplinePSD/commit/42866878e7d0ce5cdbf2398cd834e99dad6d8876
.. _624a87b: https://github.com/nz-gravity/LogPSplinePSD/commit/624a87b76c7a21ee964e893648be7b83e570f57a
.. _d7d598d: https://github.com/nz-gravity/LogPSplinePSD/commit/d7d598da476c378776c6bf8487746422c119a56f
.. _86771ee: https://github.com/nz-gravity/LogPSplinePSD/commit/86771ee360f1f5cae088f5b0bb36ee141b473b60
.. _32f41e6: https://github.com/nz-gravity/LogPSplinePSD/commit/32f41e6a8abd771f9926e800a1b4a8b55bc51a2b
.. _600387f: https://github.com/nz-gravity/LogPSplinePSD/commit/600387f6e5c544c2869d2b2f3597786b94912039
.. _6d4dbc4: https://github.com/nz-gravity/LogPSplinePSD/commit/6d4dbc4098fdbee30b84ab5de1ea65d8836845c5
.. _aedfd5a: https://github.com/nz-gravity/LogPSplinePSD/commit/aedfd5ad2792e2fe97c5d6a3f81a022d5d40463f
.. _9a98372: https://github.com/nz-gravity/LogPSplinePSD/commit/9a9837203829c226af0cb275a97ca6466fc7ed44
.. _c9d3799: https://github.com/nz-gravity/LogPSplinePSD/commit/c9d3799326cbab6122929aa37e1f45d70197afd6
.. _3f32876: https://github.com/nz-gravity/LogPSplinePSD/commit/3f32876ac89f41dc96ec72305d59783e9c053c54
.. _4d042c1: https://github.com/nz-gravity/LogPSplinePSD/commit/4d042c1cfc8168642e1d730a9f17825df815c4d3
.. _6ec4029: https://github.com/nz-gravity/LogPSplinePSD/commit/6ec4029eaee53462c3891d95e6c69cdf24a5b5e3
.. _939e855: https://github.com/nz-gravity/LogPSplinePSD/commit/939e855db5facce9242ba611a9f4309b245d2594
.. _537f8d4: https://github.com/nz-gravity/LogPSplinePSD/commit/537f8d41d24b4c169e7365b52b9d5d4645245db3
.. _3d1208b: https://github.com/nz-gravity/LogPSplinePSD/commit/3d1208b955610f9c8b428913e2d9cbc527512bc2
.. _3c6d879: https://github.com/nz-gravity/LogPSplinePSD/commit/3c6d879a6faa7d2cb092358e344fb747a062ee9a
.. _8ca7f3d: https://github.com/nz-gravity/LogPSplinePSD/commit/8ca7f3dfd9096b2658c488c796fa57283f40c43e
.. _21e1be2: https://github.com/nz-gravity/LogPSplinePSD/commit/21e1be23ef2ff8e717dccb6808407c036dbeadde
.. _18b2bfe: https://github.com/nz-gravity/LogPSplinePSD/commit/18b2bfe811fee4c9a924899da68ffb70c022b7b5
.. _02de5a7: https://github.com/nz-gravity/LogPSplinePSD/commit/02de5a7e52b7ea8048571377217ddb9b210f06d0
.. _affd908: https://github.com/nz-gravity/LogPSplinePSD/commit/affd9089da9e7864bea1e3fe77e18b86780221c6
.. _b8f91d7: https://github.com/nz-gravity/LogPSplinePSD/commit/b8f91d7e73854ee669c58f4b6a6347adb62f0af3
.. _f451b80: https://github.com/nz-gravity/LogPSplinePSD/commit/f451b80cd16fd7c1f9e54a5766976bd62503244f
.. _b8a94a2: https://github.com/nz-gravity/LogPSplinePSD/commit/b8a94a2cb82563c02189c23ed89b4ba19b9b3eeb
.. _134c0cd: https://github.com/nz-gravity/LogPSplinePSD/commit/134c0cd77b7b711d8c776449c64679f17198a800
.. _474785e: https://github.com/nz-gravity/LogPSplinePSD/commit/474785ee009568137ba08df694603289c1fe1d72
.. _d8c0a68: https://github.com/nz-gravity/LogPSplinePSD/commit/d8c0a68e83dc085ea6da57d458f049896e381f4b
.. _9143b01: https://github.com/nz-gravity/LogPSplinePSD/commit/9143b01743abae24bc1ac80408099d6e56be0829
.. _ccc1f48: https://github.com/nz-gravity/LogPSplinePSD/commit/ccc1f48b09edc66a3fbc9644d80fc6f6d91e78ed
.. _3107114: https://github.com/nz-gravity/LogPSplinePSD/commit/310711404662c76fa0a574a96cb447c402066574
.. _0314ffd: https://github.com/nz-gravity/LogPSplinePSD/commit/0314ffdb7bfe4e164e06f5927d86806f55289966
.. _c8e8f49: https://github.com/nz-gravity/LogPSplinePSD/commit/c8e8f495a54a0645a5ad57438598bf5e604cb76e
.. _d7de401: https://github.com/nz-gravity/LogPSplinePSD/commit/d7de40191cb1cdd05ad3462e22d30bd3e1e1818f
.. _34951cf: https://github.com/nz-gravity/LogPSplinePSD/commit/34951cf8a364197a185ab445b6b8877d60933e0a
.. _23c3aa0: https://github.com/nz-gravity/LogPSplinePSD/commit/23c3aa05cbdc16386b1a0b5bd08526edcf925c90
.. _ff3b289: https://github.com/nz-gravity/LogPSplinePSD/commit/ff3b289389c61277f4053e40919393e7cb41d772
.. _d7121d6: https://github.com/nz-gravity/LogPSplinePSD/commit/d7121d6c1cd87a65355b4a6f6260578b90223339
.. _f818caa: https://github.com/nz-gravity/LogPSplinePSD/commit/f818caaa86467d5f26fb116a2c29c7a360ff41cf
.. _0666415: https://github.com/nz-gravity/LogPSplinePSD/commit/0666415347785d67b2865efe521648a7a89ee000
.. _4197563: https://github.com/nz-gravity/LogPSplinePSD/commit/4197563ebdd5da00a781dc22556eeb925f1cceaf
.. _922f870: https://github.com/nz-gravity/LogPSplinePSD/commit/922f87003a657d1578a98c3d3d803055f7969fe2
.. _4944aa1: https://github.com/nz-gravity/LogPSplinePSD/commit/4944aa1501d382d8ee4f6e06780c672e588b843d
.. _d93f36b: https://github.com/nz-gravity/LogPSplinePSD/commit/d93f36bcba5a70f2d90b40c3934de265f72cb65f
.. _c9e3c79: https://github.com/nz-gravity/LogPSplinePSD/commit/c9e3c790dff28a51bf9dc97b56bd63ccbcddd43b
.. _fda820d: https://github.com/nz-gravity/LogPSplinePSD/commit/fda820dd58f3072c86876d2a2ae218869f656f6e
.. _3539ffb: https://github.com/nz-gravity/LogPSplinePSD/commit/3539ffb0b1d87445201633488da63743454e0c7f
.. _3274b74: https://github.com/nz-gravity/LogPSplinePSD/commit/3274b74c1f0c59ea43825bdca177f99f8c8fe097
.. _23210a3: https://github.com/nz-gravity/LogPSplinePSD/commit/23210a35eb751832563a69101817ba906b82edba
.. _5685aac: https://github.com/nz-gravity/LogPSplinePSD/commit/5685aac389781eaeeadda6a1c31f2820b61cbed1
.. _8e4ad33: https://github.com/nz-gravity/LogPSplinePSD/commit/8e4ad33d4e99f20a2a76d40dd8539838ed5462ea
.. _1942d60: https://github.com/nz-gravity/LogPSplinePSD/commit/1942d6079393eb78ddcc07a7a4265805bcfcb010
.. _0d619ce: https://github.com/nz-gravity/LogPSplinePSD/commit/0d619ceba76869e3ec3b2d015987a77a1671cf19
.. _cd4026f: https://github.com/nz-gravity/LogPSplinePSD/commit/cd4026f9c50b1384a4cfba70cf8e67f938a254ac
