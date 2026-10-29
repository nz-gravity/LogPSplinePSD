.. _changelog:

=========
CHANGELOG
=========


.. _changelog-v0.3.0:

v0.3.0 (2026-10-29)
===================

Features
--------

* feat: marginalize Gaussian cross-spectrum coefficients (#54)

Integrate stationary cross-spectrum spline weights analytically for both NUTS and VI, then restore full joint conditional draws for spectral reconstruction. Share reduced-state initialization and require float64 for multivariate fits.

Add scalar and matrix completion-of-the-square explanations to the multivariate tutorial, with tests for Gaussian identities, gradients, covariance recovery, and inference outputs. (`0167b8d`_)

.. _0167b8d: https://github.com/nz-gravity/LogPSplinePSD/commit/0167b8d8b9969c467a284fefa2b933f69d9fbc00


.. _changelog-v0.2.1:

v0.2.1 (2026-10-21)
===================

Chores
------

* chore(release): 0.2.1 (`781bc56`_)

* chore: move research studies out of the package repository

Link the dedicated study repositories and ignore local research and test outputs. Remove the obsolete manuscript submodule and configuration entries for retired package modules.

Validation: all 150 tests passed, Ruff checks passed, and documentation built with warnings treated as errors after executing all three tutorials. Tang coordinates, coefficients and powers match the previous main exactly across four schedules, with exact likelihood equality across 12 states. (`1059dec`_)

* chore: enable tagged releases (`63b5c7a`_)

Performance Improvements
------------------------

* perf: compute Tang coordinates without a dummy transform (`18ac7e3`_)

Unknown
-------

* Add scalar VI fitting and notebook examples (#52)

Support scalar tensor and ANOVA power fits with NumPyro SVI using the existing likelihood, smoothing prior and posterior reconstruction. Keep basic guide, optimization and stopping controls, constrained draws, losses and timings.

Demonstrate VI and NUTS curve comparisons in the existing multivariate and time-varying notebooks, and generate the API reference from docstrings. Preserve research checkpoints and comparison machinery in the separate study archive.

Validated with the full package tests, notebook and documentation build, and wrapper parity comparisons. (`0e5c087`_)

.. _781bc56: https://github.com/nz-gravity/LogPSplinePSD/commit/781bc56e562d6f75f416fdccdec01ff624f98af9
.. _1059dec: https://github.com/nz-gravity/LogPSplinePSD/commit/1059dec082fc48de4142d7a2f3a0796eff71a9a3
.. _63b5c7a: https://github.com/nz-gravity/LogPSplinePSD/commit/63b5c7a10d8bd4a3dfed8f1df05b02112321fd1b
.. _18ac7e3: https://github.com/nz-gravity/LogPSplinePSD/commit/18ac7e3cf9e0d0c3e60df225d85d467bdd2d8b63
.. _0e5c087: https://github.com/nz-gravity/LogPSplinePSD/commit/0e5c087fef211a7a89765b77220c651e83bb5f8e


.. _changelog-v0.2.0:

v0.2.0 (2026-10-01)
===================

Bug Fixes
---------

* fix: improving coverage (`7b31713`_)

* fix: remove dead code (`d230c0f`_)

* fix: update workflow (`ab76133`_)

* fix: update workflow (`93b7bae`_)

* fix: update tests (`6f1773c`_)

* fix: vi diagnostics and plotting (`f59177a`_)

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

* docs: describe stationary architecture, migration and validation (`ccb7b5b`_)

* docs: update docs (`4e40a3e`_)

* docs: fix docs (`62c196a`_)

* docs: add plots (`185e176`_)

* docs: add colab button (`f6ef8ae`_)

* docs: add some notes for commits (`2aee2f3`_)

Features
--------

* feat: bump package (`85c0273`_)

* feat(plotting): shade excluded frequency bands in diagnostic plots

Add _shade_excluded_bands helper to both the eigenvalue separation
diagnostic and PSD matrix plots. Excluded regions are highlighted with
a tan axvspan so it is immediately clear which frequency bins were
removed from the likelihood during preprocessing. (`7eab1d9`_)

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
producing a non-integer coarse-grain ratio. (`359d6aa`_)

* feat: add alpha_phi_theta and beta_phi_theta parameters for off-diagonal P-spline blocks (`65e5c0a`_)

Refactoring
-----------

* refactor: separate spline, matrix, likelihood and inference responsibilities (`4f3e164`_)

* refactor: avoid registering large deterministic sites to reduce memory usage (`c701872`_)

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
- README.md: document null-frequency issue, fixes, and run inventory. (`adfb337`_)

* refactor: streamline imports and improve frequency handling in VARMAData; update tests for frequency validation (`1903a98`_)

Testing
-------

* test: capture stationary numerical contracts before refactor (`95a7f46`_)

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
  separation plot; updated diagnostic output path assertion. (`54f2d1c`_)

* test: implement simulation of independent AR(1) channels and add consistency test for univariate and multivariate scaling

Added an AR(1)-based regression test that simulates identical independent channels, runs univariate and multivariate samplers, and checks frequency alignment plus percentile ratio statistics on the diagonal PSDs to guard against scaling mismatches. (`4390cd8`_)

Unknown
-------

* cleanup (`8a87c29`_)

* cleanup (`834842d`_)

* add anova (`26d4828`_)

* Add GridTV ANOVA log-spectrum inference with fixed references

Center the time basis on the native reconstruction grid and use separate sigma roughness priors for the frequency mean and time-varying deviation. Normalize native powers by the fixed reference before pooling and restore the reference in PSDResult reconstruction.

Reuse the power likelihood and NUTS path, retain the tensor model, and cover identifiability, reference semantics, prior scales, and pooling with regression tests. (`74f5c44`_)

* update the roughness (`5d9d245`_)

* refactor testing to simplify (`0ac46c2`_)

* rename pipeline to fit (`b89008d`_)

* simplify the diagnostics (`e1339f9`_)

* simplify the diagnostics (`8aeac4e`_)

* remove psd-compare.py (`a7a1e03`_)

* cleanup (`d413465`_)

* remove separate folder for knots (`63feac9`_)

* remove knots lvk (`27e1759`_)

* remove LVK knot alloc (`cf805d4`_)

* remove benchmarking (`077f4fa`_)

* improve jaxtyping (`02d2a56`_)

* Completed the final cleanup pass in the current checkout. The patch removes 1,568 lines and adds 291 across code, tests, and docs, excluding the pre-existing pyproject.toml edit.
- Removed PSDResult.vi_posterior and vi_spectrum, their NetCDF storage, duplicate reconstruction, and VI overlay plotting. posterior and spectrum now hold the fitted result for either method; vi holds VI diagnostics.
- Deleted diagnostics/_factors.py and its generic DataTree, mapping, sequence, and regex normalization. NUTS summaries and energy plots now build channel views from PSDResult; VI summaries read result.vi directly. Removed unsupported joint multivariate NUTS branches and unused VI loss-input adapters.
- Removed design_weights, w_design, tau, and fit_design_weights() prior centering. Kept analytical_psd as an optional reference spectrum for knot placement, separate from the coefficient prior.
- Kept rectangular GridTV and paired ScatteredTV PowerData and their distinct evaluation paths. PowerData without time coordinates now raises an error. Kept chunked compute_psd_quantiles(). Retained PipelineConfig and PowerSplineConfig.
- Updated [development.md](/Users/avi/Documents/projects/LogPSplinePSD/docs/development.md), the API page, and the multivariate example. The removed file is src/log_psplines/diagnostics/_factors.py; the principal result and diagnostics changes are in [results.py](/Users/avi/Documents/projects/LogPSplinePSD/src/log_psplines/results.py) and [summary_tables.py](/Users/avi/Documents/projects/LogPSplinePSD/src/log_psplines/diagnostics/summary_tables.py).
Initialization benchmark: On the existing small stationary one- and two-channel examples, least squares and least squares plus 5,000 Adam steps gave MSEs of 0.3168533 and 2.175586, respectively, with no material change. Paired NUTS runs had identical posterior draws and divergence counts (0/0 for one channel, 1/1 for two); paired VI runs had identical posterior draws and final ELBOs. On the warmed two-channel case, setup took about 0.007 s with least squares and 0.15 s with Adam. The refinement was removed.
Verification: The final full suite passed with the repository’s original pytest options: 98 passed, 2 skipped. Scoped Ruff and git diff --check passed. (`0b7ef2b`_)

* update pyproj (`d605f02`_)

* run prek (`84f13cf`_)

*  Completed the final VI, NUTS, model, and config cleanup in the current checkout.
- Moved channel_model_kwargs from [nuts.py](/Users/avi/Documents/projects/LogPSplinePSD/src/log_psplines/inference/nuts.py) to [model.py](/Users/avi/Documents/projects/LogPSplinePSD/src/log_psplines/inference/model.py). Removed the unused channel initialization helper. VI and evidence no longer import factor helpers from NUTS.
- Renamed the runners to run_multivariate_nuts and run_multivariate_vi. VI now returns one VIResult containing an xarray posterior, losses, per-block losses, and guide name. Removed returned means, scales, initialization values, parameters, guide objects, latent samples, khat, and runtime. With zero requested draws, it returns the guide median as one draw.
- Removed vi_psd_max_draws and the three unused Welch config fields. Replaced extra_kwargs with lnz_kwargs and made compute_lnz a plain bool defaulting to False.
- Kept PipelineConfig and PowerSplineConfig: both remain flat and understandable, while renaming them would change notebooks without improving the fitting flow. The top-level exports remain focused on fitting, results, data, models, configs, and intended preprocessing helpers. (`d89e1fb`_)

* simplify docs (`8625fe4`_)

* refactoring reconstructuon and plotting, and data storage (`8ede2f6`_)

* clean psd plotter (`435b7a0`_)

* cleanup (`8b85b35`_)

* Merge remote main and preserve power partition fits (`d119231`_)

* Merge pull request #51 from nz-gravity/refactor/arviz-diagnostics-only

Refactor results so ArviZ is diagnostics-only (`61f9b67`_)

* allow undefined R-hat for one-chain factors (`5ff3ddd`_)

* remove stale study import contract (`e369660`_)

* remove stale arviz_utils typecheck overrides (`b0fa1bd`_)

* avoid VI-only plots being labelled as NUTS (`8536a4b`_)

* namespace observed coordinates in native NetCDF storage (`43766c2`_)

* fix native point-estimate dataset test (`d2d09c4`_)

* fix diagnostics adapter for arviz-base API (`3ece5e4`_)

* remove stale ArviZ result-layer docs (`7b9245c`_)

* remove stale ArviZ result-layer docs (`b62199b`_)

* document native result model and diagnostics-only ArviZ (`357344d`_)

* remove obsolete ArviZ result layer (`4c775ad`_)

* remove obsolete ArviZ result layer (`c6a907a`_)

* remove obsolete ArviZ result layer (`ee365f5`_)

* remove obsolete ArviZ result layer (`505b0b8`_)

* remove obsolete ArviZ result layer (`fe8c8b6`_)

* remove obsolete ArviZ result layer (`42fed14`_)

* drop obsolete spline-storage assertion (`629fce5`_)

* test plotting from PSDResult directly (`c142578`_)

* replace ArviZ storage tests with native result tests (`846532b`_)

* remove spline-storage tests with old result layer (`4ada8e9`_)

* migrate MCMC integration tests to PSDResult (`51d0fa7`_)

* remove final idata pipeline assertion (`38a2e86`_)

* migrate pipeline tests to native PSDResult (`b20bc2a`_)

* use native posterior in stationary regression test (`791b63e`_)

* update power tests for native result API (`81a2eb3`_)

* test native result contract (`a8dd05b`_)

* normalize sampler statistic names natively (`1d7cb58`_)

* preserve factor log-likelihood names (`17c9817`_)

* read empirical plotting data from PSDResult (`a7b47b9`_)

* report diagnostics from native results (`c416cf5`_)

* make diagnostics consume PSDResult without arviz_utils (`0de8971`_)

* remove diagnostics dependency on arviz_utils (`54dc507`_)

* extract plotting data directly from PSDResult (`db509f6`_)

* plot from PSDResult and diagnostic ArviZ view (`0081553`_)

* keep ArviZ conversion at diagnostics boundary (`0b4d726`_)

* construct scattered results directly (`985df28`_)

* build native power results without ArviZ packers (`84c44cb`_)

* separate all log-likelihood deterministic sites (`9dbf973`_)

* support scattered power in native results (`ffc9106`_)

* build native PSDResult directly in pipeline (`d87cbaf`_)

* make evidence consume native posterior datasets (`43e97ad`_)

* make PSDResult fully native and self-contained (`440d523`_)

* return native NUTS results without ArviZ (`c7b74b6`_)

* avoid deprecated result tree access in cleanup test (`61352d8`_)

* test native PSDResult and ArviZ adapter (`bca32f9`_)

* expose diagnostics-only ArviZ adapter (`71cdc91`_)

* use lightweight arviz-base adapter (`16063c3`_)

* use model reconstruction utilities (`511b1d4`_)

* use model reconstruction utilities (`560bfd9`_)

* leave reconstruction compatibility shim (`ae66acd`_)

* move spectral reconstruction into models (`d0fab7c`_)

* add minimal ArviZ diagnostics adapter (`00d3165`_)

* refactor results away from ArviZ data model (`ef7a655`_)

* cleanup (`e685d73`_)

* update logo (`fc83a87`_)

* fix docs (`62d5272`_)

* update plots (`1f88ed2`_)

* cleanup docs (`4324316`_)

* add notes (`8770f05`_)

* add notes and cleanup docstrings (`9de972e`_)

* refactor (`4d8d1c5`_)

* clean docs (`fb3279e`_)

* remove LVK (`bfc87e8`_)

* refactoring (`4eba004`_)

* add docs (`04a3461`_)

* add plots (`a8944fb`_)

* cleanup moving periodogram (`a23bcc3`_)

* update docs (`6e365bc`_)

* more relaxed (`eec0cbb`_)

* more relaxed (`f0ede1c`_)

* more relaxed (`1ec3c22`_)

* more relaxed (`da95fc0`_)

* add multimethod (`5138c2b`_)

* hacking on WDM (`0b2609f`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`98a5ea3`_)

* Merge pull request #50 from nz-gravity/copilot/skip-lnz-test

skip: disable failing LnZ tests while LnZ is not in use (`fdcf350`_)

* skip: mark failing LnZ tests as skipped, disable compute_lnz in multivar integration test (`5b0e92c`_)

* KILL UNIVAR.py (`51156f5`_)

* confirm that we are adding eta to the LnZ kwargs (`074ee64`_)

* fix LnZ using a whitening step (`2102222`_)

* nuke old plotter (`930b1fc`_)

* add logging fixes (`6c3c970`_)

* cleanup (`2a68197`_)

* cleanup (`50d71bb`_)

* remove lvk lisa (`2f6e7d9`_)

* more junk removal (`efc3406`_)

* Refactor sampler factory: Move sampler factory helpers to preprocessing.py

- Moved functions related to sampler creation and configuration from sampler_factory.py to preprocessing.py.
- Removed the sampler_factory.py file as it is no longer needed.
- Updated imports in relevant modules to reflect the new locations of the moved functions.
- Simplified the NUTSSampler class by removing the VIInitialisationMixin.
- Adjusted the initialization artifacts and runner functions to accommodate the refactor.
- Updated tests to align with the new structure and ensure functionality remains intact. (`d52d2f8`_)

* adding in more pipeline stuff (`39db8c6`_)

* cleanup (`d56fe94`_)

* cleaup VI plotting (`632e7e3`_)

* add energy plot (`c0a64eb`_)

* cleanup (`41de0ee`_)

* make plotting work for univar (`516b44b`_)

* cleanup loading/saving (`5331cec`_)

* simplify vi (`46b436e`_)

* addd mcmcs (`c10d4fb`_)

* add shorter diagnostics (`c7236ae`_)

* cleanup (`7b46e28`_)

* cleanup (`b026490`_)

* cleanup (`8c43e4a`_)

* break up adapters for VI (`dc14b0e`_)

* cleanup (`8255ac0`_)

* add delta for psline block (`229e804`_)

* edit eta comments (`8dac000`_)

* add enbw and eta to vi (`c0bb068`_)

* fix test (`2a9ee60`_)

* remove psis warning (`a7c508a`_)

* nuke useless test (`d2faa1f`_)

* add ignore (`99d3a52`_)

* add tests (`a281ded`_)

* add runner (`3fd44b0`_)

* add eta validation (`2a2b1a0`_)

* add tests for arious eta settings (`f315439`_)

* Enhance VI diagnostics with additional coarse-grained context and PSD extraction (`2c6f8df`_)

* Merge branch 'main' into eta_tempering (`b19197b`_)

* add some changes to VI init (`8184c01`_)

* REMOVE EXTRA STUDIES (`d21bc02`_)

* cleanup (`49d2603`_)

* cleanup result reading (`2c15bb0`_)

* fixes for VI runner (`ac42af7`_)

* fix factory bug (`962e84f`_)

* Merge pull request #47 from nz-gravity/better_memory_handling

Better memory handling (`38bbc8a`_)

* fix uni vs multi overlap (`d6fe80c`_)

* cleanup (`44a4f69`_)

* lazy loading of posterior-psd (`e67261d`_)

* add comment for pspline (`d5c5144`_)

* removed: large full-frequency deterministic state duplication in memory (`3333e64`_)

* fix mem issue (`f48529f`_)

* add pytest-memray to dev dependencies; enhance testing for PSD reconstruction and weight sampling (`110f31b`_)

* fix mem issue (`c4319af`_)

* nuke old run file (`f757b98`_)

* cleanup for multivar (`2ba88a5`_)

* Merge pull request #46 from nz-gravity/multivar_knot_alloc

Improve multivar knot alloc (`30e3d9a`_)

* fix knot alloc (`d11cc7e`_)

* add fixes for knot alloc (`ebb8025`_)

* Merge remote-tracking branch 'origin' into multivar_knot_alloc (`3463ff4`_)

* Merge pull request #45 from nz-gravity/different_number_of_knots_per_cholesky_element

Allow different number of knots per cholesky-component (`403c0e6`_)

* cleanup (`43a26e7`_)

* allow differnt numbner of cholesky components (`accfd48`_)

* knot alloc: remove spectal method, fixes to cholesky method (`1c5734d`_)

* add more runners (`2355be6`_)

* start eta tempering studies (`f31c930`_)

* add improved analysis (`a706ab8`_)

* Merge pull request #43 from nz-gravity/wishart_hacking

Add frequency exclusion, Wishart floor, and TDI null handling (`8bb77a6`_)

* Refactor multivariate sampler to support per-component theta basis and penalties

- Updated the MultivarBaseSampler to log the number of B-spline basis models.
- Modified the _blocked_channel_model function to accept separate real and imaginary theta basis and penalty components.
- Introduced a method to retrieve per-theta basis and penalty arrays for blocked channels in MultivarBlockedNUTSSampler.
- Adjusted the _channel_model_kwargs method to pass the new per-component theta basis and penalties.
- Enhanced the VI initialization functions to accommodate the new structure of theta weights and basis.
- Updated tests to reflect changes in theta basis handling and ensure compatibility with new component structure.
- Removed legacy code related to shared theta basis and penalties, ensuring a cleaner implementation. (`2d0c3ea`_)

* Enhance multivariate PSD workflow and preprocessing

- Added support for additional command-line arguments in ``run_headnode_segment_sweep.sh`` to customize sampling, warmup, chains, and VI steps.
- Introduced a new script ``run_knot_compare_workflow.sh`` for executing knot comparison workflows with posterior options.
- Modified ``_preprocess_with_run_config`` in ``preprocessing.py`` to apply frequency exclusion after coarse graining, ensuring proper data alignment.
- Updated ``multivar_psplines.py`` to utilize channel-space Wishart matrices for knot scoring, preserving cross-spectral structure.
- Added unit tests in ``test_multivar_knot_methods_unit.py`` to verify the use of Wishart matrices in multivariate density scoring. (`823dd25`_)

* more test jobs (`60af11f`_)

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
- Created a new SLURM script for exploring coverage with varying segment sizes and coarse-graining configurations. (`1493808`_)

* add lisa testing (`1fd93ba`_)

* hacking (`65bc35f`_)

* adjustments to vi (`53d8791`_)

* update L2 logging, fix slurm mempry (`707d6ea`_)

* adjust slurm (`13618a4`_)

* adjust VI (`3339a57`_)

* more VI cleanup (`23deb59`_)

* simplify VI (`469b526`_)

* hacking on VI: adding a coarse-grain VI (`b893126`_)

* add prior plot (`a311488`_)

* add prior plot (`a185c83`_)

* Merge pull request #42 from nz-gravity/testing_different_phi

Use exact transformed-Gamma prior for log_phi in pspline_block (`3631264`_)

* add more comments (`9363997`_)

* ran tests (`ad1b986`_)

* add least_squares_weight_initialiser (`deddfb5`_)

* Simplify docs (`3a5eedb`_)

* remove lisa LnZ test and simplify logs (`f59876d`_)

* changes to lisa run (`52c6e64`_)

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
- collect_results.py: aggregates seed results, handles nested slug/seed layout (`7ec4d48`_)

* Refactor sampler utilities: move functions to pspline_block.py and update imports

Also move LnZ drom mutlivar_block to multivar_base (`a5e664b`_)

* Revert default knot scoring to spectral, default window to rect; document findings

The Cholesky knot scoring refactor (f3bed24) caused a ~5.5pp coverage drop
(0.872 → 0.816) confirmed over 100 seeds. Spectral energy scoring places knots
where PSD curvature is highest, giving better-calibrated CIs. Reverts
3d_study.py defaults to spectral scoring and rect window. Documents the full
knot scoring comparison in the study README. (`60d1a9c`_)

* add more testing (`2d55780`_)

* Add configurable knot scoring (cholesky vs spectral vs uniform) for coverage comparison

Restores the pre-refactor spectral-energy knot scoring as a selectable
option alongside the current Cholesky-based scoring. Adds --knot-method
and --knot-scoring CLI args to 3d_study.py. Includes SLURM scripts for
all three variants (uniform, density+spectral, density+cholesky). (`db6afdf`_)

* Change SLURM script to rect + Nh=4 for VAR3 coverage study (`783ec66`_)

* Add Tukey(0.1) + Nh=4 SLURM script for VAR3 coverage study (`38eebe4`_)

* Reorganise README results table: separate rect baseline from Hann exploratory runs (`fff30e3`_)

* cleanup (`972b2bb`_)

* Add --outdir out_var3_sanity to var3d.slurm for scikit-fda baseline sanity check (`c724b0e`_)

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

181 tests pass, 6 skipped. (`b5235ef`_)

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
to pyproject.toml dependencies.  All 216 tests pass. (`b99320d`_)

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
and fix test_null_space_contains_linear to use the clamped knot vector. (`7255d97`_)

* Restore max(|P|) normalization in GPS penalty

Removing normalization entirely (commit 837ed9c) left raw D^T D values
~780,000 on the diagonal, causing φ to over-smooth catastrophically
(coverage dropped from 0.81 to 0.60).

Restore the max(|P|) = 1 normalization that matches the convention used
by scikit-fda's L2Regularization.penalty_matrix(). This keeps the φ prior
well-calibrated and should recover the 0.81 GPS coverage baseline. (`52da30f`_)

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

All 35 GPS penalty tests pass; full suite 203 passed, 6 skipped. (`837ed9c`_)

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
- Document next step: GPS + Rect + light shrinkage (tau=1.0) (`e692e08`_)

* differnt savepath (`dbda75e`_)

* GPS study: use rect window, add --label flag for per-seed folder signage

- Switch var3d_gps_basis.slurm from hann → rect window (matches the
  existing baseline runs for apples-to-apples comparison)
- Add --label CLI arg to 3d_study.py: appends a short tag to each
  per-seed output folder (e.g. seed_0_short_nb4_N2048_K20_rect_noNh_gps/)
  so GPS vs old-basis results are unambiguous when scanning output dirs
- GPS slurm passes --label gps (`f34043e`_)

* Prep GPS basis study: --outdir flag, fix slurm output dir, drop scikit-fda

- Add --outdir CLI argument to 3d_study.py so different study variants
  write to separate directories (e.g. out_var3_gps_basis vs out_var3)
- Update var3d_gps_basis.slurm to pass --outdir out_var3_gps_basis,
  preventing collision with existing hann-window baseline runs
- Remove scikit-fda from pyproject.toml dependencies (replaced by
  scipy BSpline + GPS penalty in previous GPS penalty PR) (`9a2b816`_)

* Add SLURM script for VAR3-3D study with GPS penalty basis (100 seeds, K=20, Nb=4, hann window) (`d4543e0`_)

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

Ready for VAR3-3D study runs on OzStar. (`f3e8b98`_)

* Update README with GPS penalty coverage probe results (K=10-50 at N=2048) (`bb21312`_)

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
- .pre-commit-config.yaml: fix mypy hook for systems without python symlink (`751683c`_)

* Enhance window parameter handling: support tuple input for wishart_window and add parsing function; introduce SLURM script for tukey window simulation. (`7263da6`_)

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
  suppress pre-existing errors in known-noisy modules via ignore_errors (`b06de50`_)

* Refactors Cholesky mapping and improves diagnostics

Moves the PSD-to-Cholesky transformation into a shared utility for reuse and clarity, updating preprocessing and knot placement to leverage it. Enhances diagnostics plotting by subsampling trace plots for large posteriors, adds better plot layout and customization, and improves figure saving robustness. Updates logging and default plotting behaviors for more informative and manageable outputs. (`f3bed24`_)

* chnage settings (`91ffcba`_)

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
preprocessing diagnostics figures. (`72b59cc`_)

* changes to slurm (`2beee51`_)

* fix stupid typo (`5674e9f`_)

* improve preproc (`bdb03f4`_)

* arviz saving fixes (`01b411f`_)

* arviz saving fixes (`be1ac26`_)

* add fixes for preproc (`f0f48db`_)

* add result collection (`4f908b5`_)

* add 3D study for coarse grain (`286a2b4`_)

* fix tests (`70bb701`_)

* bug fixes (`f5fd9a3`_)

* add new file (`8fe817e`_)

* add new file (`8353456`_)

* add new file (`9c5b39d`_)

* Merge branch 'improved_basis' (`0491954`_)

* add new file (`6724575`_)

* reduce logs (`319faef`_)

* add **version** (`83856b1`_)

* fixed window energy (`69da535`_)

* Merge pull request #40 from nz-gravity/improved_basis

Improves multivariate PSD calibration and ArviZ compatibility (`8f35eca`_)

* typo fix (`166cf77`_)

* Improves multivariate PSD calibration and ArviZ compatibility

Introduces window tapering support for multivariate PSD estimation and corrects posterior calibration by scaling the Whittle likelihood using equivalent noise bandwidth (ENBW). Refactors output and diagnostics to use xarray.DataTree for compatibility with ArviZ >=1.0.0, adds detailed coverage breakdowns, and enhances CLI experiment support for rank-deficiency analysis.

Clarifies credible interval coverage limitations in documentation and enables flexible window configuration to address spectral leakage and posterior miscalibration. (`ebb13cb`_)

* edit settings (`e05c7b9`_)

* Add design_psd + tau soft shrinkage to LISA multivariate study

Wire up the recently-merged soft-shrinkage feature in the LISA
analysis script. With DESIGN_PSD_TAU=1.0, each P-spline component is
softly pulled toward the true LISA sensitivity model (true_psd_source),
which should reduce divergences and improve sampler convergence.

- Add DESIGN_PSD_TAU constant (float | None) for easy tuning/disabling
- Include tau value in the run slug so runs produce distinct directories
- Pass design_psd=true_psd_source and tau=DESIGN_PSD_TAU to run_mcmc() (`174247d`_)

* Merge pull request #38 from nz-gravity/add_soft_shrinkage

add soft shrinkage (`0239d93`_)

* add docs (`3a8796d`_)

* Implement soft shrinkage toward design weights in multivariate sampler and add corresponding tests (`933f6d6`_)

* hacking (`11ff71f`_)

* Enhance multivariate PSD diagnostics and streamline environment variable handling

- Removed environment variable functions in  and replaced with direct assignments for clarity and simplicity.
- Updated hyperparameters and configurations in  for improved model performance.
- Introduced new functions in  for generating per-frequency diagnostic plots when a true PSD is available, including relative error and coverage diagnostics.
- Added validation for positive definiteness of covariance matrices in .
- Integrated true PSD diagnostics into the multivariate sampler's plotting routine.
- Created a new markdown file for documenting large study results.
- Added tests for truth-aware PSD diagnostics to ensure functionality. (`86ce4ef`_)

* hacking on LnZ (`d9bceda`_)

* add 3d stiudy (`c65f035`_)

* fix knot alloc (`83aed50`_)

* adjust LISA datagen (`0d096ca`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`5b759a8`_)

* Add files via upload (`730adec`_)

* Refactor and enhance multivariate P-spline functionality

- Updated the ``_posterior_subset_for_rhat`` function to avoid passing parent coordinates, preventing NaN issues during alignment.
- Introduced a new utility function ``_to_flat_finite_array`` to convert ArviZ/xarray outputs to flat finite float arrays, improving handling of R-hat and ESS calculations.
- Enhanced the ``VARMAData`` class to include empirical stationarity checks, adding flags and metrics for better diagnostics.
- Refactored the PSD matrix plotting functions to support per-panel knot extraction for multivariate cases, improving flexibility in visualizations.
- Implemented a new method for initializing knots based on raw frequency/power arrays, allowing for more robust knot placement strategies.
- Added tests for multivariate knot methods, ensuring support for various knot placement strategies and validating their behavior against empirical data.
- Created a new shell script for running VAR3 simulations, facilitating reproducibility in experiments. (`874f652`_)

* Refactor multivariate utilities and enhance VARMAData validity checks

- Moved _interp_complex_matrix import to multivar_utils.py for better organization.
- Added u_re_im_to_U function to combine real and imaginary components into a complex matrix.
- Updated VARMAData class to include validity checks for stationarity and data integrity.
- Enhanced test coverage for VARMAData validity checks and u_re_im_to_U functionality. (`8cf0897`_)

* Merge branch 'main' into copilot/fix-mypy-static-type-errors (`ef296d5`_)

* more refactoring (`d507ae2`_)

* use lambda and v to match up with paper (`555f786`_)

* Improve code comments for clarity

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`0f7efd7`_)

* Address code review feedback - improve comments and warnings

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`fbdd866`_)

* Simplify sampler loop logic in runtime_benchmark

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`3c51d4e`_)

* Fix mypy type checking errors

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`1f0b992`_)

* Initial plan (`f354738`_)

* make coarse grain clearer (`86151ef`_)

* more cleanup (`8acdc0d`_)

* break up preprocessing (`353e42d`_)

* Add preprocessing module and refactor coarse grain functionality

- Introduced a new preprocessing module with configurations for MCMC and diagnostics.
- Implemented data preprocessing functions including alignment of true PSD and coarse graining.
- Refactored existing coarse grain functions to reside within the new preprocessing structure.
- Updated tests to reflect changes in module imports and removed deprecated sampler parameters.
- Ensured compatibility with existing tests by adjusting configurations and function calls. (`94ccb8c`_)

* remove preprocessed imputs. Always pass in timeseries (`2898850`_)

* Refactor MCMC diagnostics and configuration: enforce full diagnostics mode, remove unused options, and streamline preprocessing checks (`a0844bc`_)

* Add MCMC utility functions and enhance PSD matrix plotting

- Introduced a new module ``mcmc_utils.py`` containing various configurations and utility functions for MCMC sampling, including model configuration, diagnostics, and sampler factory configurations.
- Implemented functions to handle true PSD unpacking, interpolation, and normalization of coarse grain configurations.
- Enhanced the ``_prepare_processed_data`` function to handle different data types and added checks for frequency truncation.
- Updated the PSD matrix plotting functionality in ``psd_matrix.py`` to extract and plot spline knots from inference data, improving visualization of model fits.
- Added options to display knots in various panels of the PSD matrix plot, ensuring better representation of model structure. (`3af80a4`_)

* remove old legacy stuff (`5f27647`_)

* remove old legacy stuff (`1d316ad`_)

* Merge pull request #36 from nz-gravity/hacking

hacking (`88887d5`_)

* Merge remote-tracking branch 'origin/hacking' into hacking (`25af3b0`_)

* hacking (`0fd0481`_)

* hacking (`b9a20ae`_)

* simplify diagnostics (`e754a09`_)

* Unifies and refactors diagnostics and interpolation logic

Consolidates diagnostics code by removing redundant modules and integrating energy/E-BFMI and time-domain checks into a single diagnostics pipeline. Unifies frequency-indexed interpolation for PSDs and matrices, improving consistency and correctness when grids are unsorted or have duplicates. Refactors plotting and summary logic to support expanded NUTS convergence checks, clearer reporting, and more robust handling of high-dimensional outputs. Updates tests accordingly to match new diagnostics and interpolation paths.

Addresses scalability, maintainability, and reliability of core diagnostics and output routines. (`217365f`_)

* Merge pull request #35 from nz-gravity/typing

Type checking (`dcb1434`_)

* Refactor and enhance type checking across the log_psplines module

- Updated type hints to use Optional and Union for better clarity.
- Introduced runtime type checking for various functions to ensure input validity.
- Added assertions to check for None values in critical variables.
- Improved diagnostics handling in the save_vi_diagnostics functions.
- Enhanced the initialization of weights and basis matrices in the initialization module.
- Added tests for runtime type checking to ensure robustness against invalid inputs.
- Refactored plotting functions to ensure compatibility with updated type hints.
- Improved error handling and assertions in the multivariate sampler classes. (`39b99d3`_)

* Refactor coarse graining functions and update imports to use multivar_utils; adjust tests for new configurations (`cb240a5`_)

* Merge branch 'refactoring_weights' (`0985f42`_)

* work more on docs (`73f1a91`_)

* Merge pull request #34 from nz-gravity/refactoring_weights

Refactor freq_weights to Nh (`ed71380`_)

* more refactoring (`8751c19`_)

* Refactor frequency weights handling to use Nh instead of freq_bin_counts across multivariate samplers (`6744a7d`_)

* More changes (`00c329c`_)

* Refactor frequency weights handling to use per-bin counts in multivariate samplers and related documentation updates (`dee569a`_)

* remove LISA specific code (`0be11e4`_)

* Simplify code (`5a9e33f`_)

* Merge pull request #33 from nz-gravity/diagnostics_refactor

Refactor diagnostics modules and remove unused code (`afb258c`_)

* Refactor diagnostics modules and remove unused code

- Deleted psd_metrics.py and psd_posterior_diagnostics.py as they were no longer needed.
- Removed time_domain_moments.py and associated tests for similar reasons.
- Updated run_all.py to check for PSD datasets more efficiently.
- Adjusted diagnostics summary generation to utilize cached attributes for improved performance.
- Enhanced the BaseSampler class to compute and cache full diagnostics after inference.
- Modified multivar_blocked_nuts.py and nuts.py to ensure full diagnostics are cached after data creation.
- Removed outdated tests related to deleted modules and added new tests to verify caching behavior in diagnostics summary. (`6471534`_)

* Merge pull request #31 from nz-gravity/copilot/clean-up-unnecessary-pytests

Consolidate redundant PSD matrix test files (`82589dd`_)

* Address code review feedback: use OUTDIR constant and remove trailing whitespace

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`fd77e9b`_)

* Consolidate PSD matrix tests into single refactor file

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`f12131f`_)

* Initial plan (`449a7a9`_)

* Merge pull request #32 from nz-gravity/copilot/improve-documentation-structure

Reorganize documentation into hierarchical structure (`012c481`_)

* Update docs .gitignore to exclude _build directory (`8afc400`_)

* Add Quick Links section and remove duplicate conventions file

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`959a3e5`_)

* Reorganize documentation structure with clear sections

Co-authored-by: avivajpeyi <15642823+avivajpeyi@users.noreply.github.com> (`eadb594`_)

* Initial plan (`f8b5521`_)

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

.. _7b31713: https://github.com/nz-gravity/LogPSplinePSD/commit/7b317130787c9d8d3b29f5933de4acbde3556a54
.. _d230c0f: https://github.com/nz-gravity/LogPSplinePSD/commit/d230c0fb883117dc6e65b46654103931eec96575
.. _ab76133: https://github.com/nz-gravity/LogPSplinePSD/commit/ab761338e93b3fefc6fb1c1fabaa7f1d3b1cd154
.. _93b7bae: https://github.com/nz-gravity/LogPSplinePSD/commit/93b7bae514ffb3d673104f4208564ae954546e82
.. _6f1773c: https://github.com/nz-gravity/LogPSplinePSD/commit/6f1773c390d3ac2301fe45109ab2eeeddb53a92c
.. _f59177a: https://github.com/nz-gravity/LogPSplinePSD/commit/f59177a60bd5ad71ac876a37074e9d7a1f4e5d12
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
.. _ccb7b5b: https://github.com/nz-gravity/LogPSplinePSD/commit/ccb7b5b4d6dcee5de87625c24e247717020b7aca
.. _4e40a3e: https://github.com/nz-gravity/LogPSplinePSD/commit/4e40a3e380b1f99ba4181f4dabaf9c6ca4caedce
.. _62c196a: https://github.com/nz-gravity/LogPSplinePSD/commit/62c196aff27bd37e910a55521f84b610a9ec4306
.. _185e176: https://github.com/nz-gravity/LogPSplinePSD/commit/185e176503f522df62ad726d7306f1d696d3dcb9
.. _f6ef8ae: https://github.com/nz-gravity/LogPSplinePSD/commit/f6ef8ae79c8e6e52f10b8ba88601ec991b540851
.. _2aee2f3: https://github.com/nz-gravity/LogPSplinePSD/commit/2aee2f37bed4fd09d59f7070794592e12bb1b435
.. _85c0273: https://github.com/nz-gravity/LogPSplinePSD/commit/85c0273e8176f2622e54eef16ccb5139c04de8de
.. _7eab1d9: https://github.com/nz-gravity/LogPSplinePSD/commit/7eab1d92983b87dbf690c030365d83457fbc605d
.. _359d6aa: https://github.com/nz-gravity/LogPSplinePSD/commit/359d6aab46f154e5dbd8e33be78ec3aae04f5b7a
.. _65e5c0a: https://github.com/nz-gravity/LogPSplinePSD/commit/65e5c0a19e6322f761e0e838cf7a8f27fc2746e0
.. _4f3e164: https://github.com/nz-gravity/LogPSplinePSD/commit/4f3e16410a1f1a23a1281c26104d43a28823e74c
.. _c701872: https://github.com/nz-gravity/LogPSplinePSD/commit/c7018721bc6deb8f3d4b0f3a26cee33830f89344
.. _adfb337: https://github.com/nz-gravity/LogPSplinePSD/commit/adfb33780dbf9f51656796aba1c7d438fbd444b5
.. _1903a98: https://github.com/nz-gravity/LogPSplinePSD/commit/1903a9822723db11ee8e449c21d2bea5756da964
.. _95a7f46: https://github.com/nz-gravity/LogPSplinePSD/commit/95a7f464258750563f6dfb0314ec876f29cca22b
.. _54f2d1c: https://github.com/nz-gravity/LogPSplinePSD/commit/54f2d1ce0ad76eb6211e65f5ad43e952014c807e
.. _4390cd8: https://github.com/nz-gravity/LogPSplinePSD/commit/4390cd8aa854d6658ffd65b3eea3ed6e9f31ded4
.. _8a87c29: https://github.com/nz-gravity/LogPSplinePSD/commit/8a87c290a17a66411fa575ffb012db71404ea3ba
.. _834842d: https://github.com/nz-gravity/LogPSplinePSD/commit/834842d8ef07d3664948e6a49346dbca106ea713
.. _26d4828: https://github.com/nz-gravity/LogPSplinePSD/commit/26d4828106aa8240c61361a5572beafb6f0d63d1
.. _74f5c44: https://github.com/nz-gravity/LogPSplinePSD/commit/74f5c449006fdee67f99673a1ca37fc30b73ef31
.. _5d9d245: https://github.com/nz-gravity/LogPSplinePSD/commit/5d9d24558d876fbe9b06c390a99d4cf037894df2
.. _0ac46c2: https://github.com/nz-gravity/LogPSplinePSD/commit/0ac46c2192e80ac6e1a0b23beaad573070d738d7
.. _b89008d: https://github.com/nz-gravity/LogPSplinePSD/commit/b89008d8a327ca73029991d248860c76be905736
.. _e1339f9: https://github.com/nz-gravity/LogPSplinePSD/commit/e1339f95199c64f72a5b23c6713f43256809c2aa
.. _8aeac4e: https://github.com/nz-gravity/LogPSplinePSD/commit/8aeac4e98f684b8358d560d5a72c06287128fcd0
.. _a7a1e03: https://github.com/nz-gravity/LogPSplinePSD/commit/a7a1e03074ad850738be951865ece741943df2e2
.. _d413465: https://github.com/nz-gravity/LogPSplinePSD/commit/d4134659add4ac9b44a10ce36eb04e6a89dda1be
.. _63feac9: https://github.com/nz-gravity/LogPSplinePSD/commit/63feac9258f5289a4188adbbdd67eaef35b651c6
.. _27e1759: https://github.com/nz-gravity/LogPSplinePSD/commit/27e1759bb65d78214c0cbfd63bca05d8759b6384
.. _cf805d4: https://github.com/nz-gravity/LogPSplinePSD/commit/cf805d4cbeaa8057d58471c9b958625205a9f564
.. _077f4fa: https://github.com/nz-gravity/LogPSplinePSD/commit/077f4fafebca03a442549c77269db320cd00df22
.. _02d2a56: https://github.com/nz-gravity/LogPSplinePSD/commit/02d2a56ed662165203a454905ae060f88fcdcb66
.. _0b7ef2b: https://github.com/nz-gravity/LogPSplinePSD/commit/0b7ef2b9d02d5035b0e710e69de035785ef34da7
.. _d605f02: https://github.com/nz-gravity/LogPSplinePSD/commit/d605f02f0c92071fcff1fc8c25cb3d41507208c3
.. _84f13cf: https://github.com/nz-gravity/LogPSplinePSD/commit/84f13cf332ad8607d9ce3558493bb06f022e2efe
.. _d89e1fb: https://github.com/nz-gravity/LogPSplinePSD/commit/d89e1fb70211a3d9d0ced3b587a5c95cff7540a5
.. _8625fe4: https://github.com/nz-gravity/LogPSplinePSD/commit/8625fe483b01172197b19df25bfb1756ff94a865
.. _8ede2f6: https://github.com/nz-gravity/LogPSplinePSD/commit/8ede2f656d6320442f0381ee76ef896b8244ae54
.. _435b7a0: https://github.com/nz-gravity/LogPSplinePSD/commit/435b7a07cd1ca5eaab24e9bc2cec39f14c8969aa
.. _8b85b35: https://github.com/nz-gravity/LogPSplinePSD/commit/8b85b35ca0fef22c9a39f656c58896a3a4095b3f
.. _d119231: https://github.com/nz-gravity/LogPSplinePSD/commit/d119231d49c3b3ed14594fffa0b4fc5c3f396eab
.. _61f9b67: https://github.com/nz-gravity/LogPSplinePSD/commit/61f9b67ff748dcbb36b578862ed894c0ae38dbf9
.. _5ff3ddd: https://github.com/nz-gravity/LogPSplinePSD/commit/5ff3ddddff4aab1321ba520629d89d3e44079c2d
.. _e369660: https://github.com/nz-gravity/LogPSplinePSD/commit/e369660b280bbc77f5539560c4ee86b04a46c00e
.. _b0fa1bd: https://github.com/nz-gravity/LogPSplinePSD/commit/b0fa1bd1bd7a9f5b67365343693c96e9f5a6cdfe
.. _8536a4b: https://github.com/nz-gravity/LogPSplinePSD/commit/8536a4bfe918244feffa02dbda7d8cfc07f70632
.. _43766c2: https://github.com/nz-gravity/LogPSplinePSD/commit/43766c2224df6560ca31e6ff712c5be03e59a1ac
.. _d2d09c4: https://github.com/nz-gravity/LogPSplinePSD/commit/d2d09c4b429288bc18ebe963566022ebf0d60751
.. _3ece5e4: https://github.com/nz-gravity/LogPSplinePSD/commit/3ece5e48360363c0102cf2894cf30fecb05ed72e
.. _7b9245c: https://github.com/nz-gravity/LogPSplinePSD/commit/7b9245c0dc29ee63ba523eb1b69ef1b3b7f723b3
.. _b62199b: https://github.com/nz-gravity/LogPSplinePSD/commit/b62199b4869f02fbec8edf824cdaa2fb8a1cd728
.. _357344d: https://github.com/nz-gravity/LogPSplinePSD/commit/357344d35294cfcebce082650a3cab293081b1bc
.. _4c775ad: https://github.com/nz-gravity/LogPSplinePSD/commit/4c775adb965b45f7d1c63d923b6710a1af6044bd
.. _c6a907a: https://github.com/nz-gravity/LogPSplinePSD/commit/c6a907a3cab9b4a786232ad4b9b264f22c95926b
.. _ee365f5: https://github.com/nz-gravity/LogPSplinePSD/commit/ee365f5d28d55938fa0c06cbd68f710947102cd8
.. _505b0b8: https://github.com/nz-gravity/LogPSplinePSD/commit/505b0b89147b20420d3d34efdc5d73da51ad81ca
.. _fe8c8b6: https://github.com/nz-gravity/LogPSplinePSD/commit/fe8c8b609f4184c2910888ae43434c3bcfbbe77a
.. _42fed14: https://github.com/nz-gravity/LogPSplinePSD/commit/42fed1497b6fd401f6ef269d3a40eca266fe03f3
.. _629fce5: https://github.com/nz-gravity/LogPSplinePSD/commit/629fce598428c7c103be26cde2ca985a4af8b560
.. _c142578: https://github.com/nz-gravity/LogPSplinePSD/commit/c142578bbbeb4ed6b9956bbe582c0cc3d043caa9
.. _846532b: https://github.com/nz-gravity/LogPSplinePSD/commit/846532b2764146b69fbff8f285319950bdabc0cd
.. _4ada8e9: https://github.com/nz-gravity/LogPSplinePSD/commit/4ada8e9d3bbdfa8f1bd7ce7c523d2993ed5f4b72
.. _51d0fa7: https://github.com/nz-gravity/LogPSplinePSD/commit/51d0fa7c5da3408db3e9de1d02c12c2b64000603
.. _38a2e86: https://github.com/nz-gravity/LogPSplinePSD/commit/38a2e86bc653f63270e01c76ccde4e92c7b7fcb9
.. _b20bc2a: https://github.com/nz-gravity/LogPSplinePSD/commit/b20bc2a303d0f3113e29c8e7bf5f4e8da79a59fa
.. _791b63e: https://github.com/nz-gravity/LogPSplinePSD/commit/791b63e83b01449e72f866d0a716f357747f497c
.. _81a2eb3: https://github.com/nz-gravity/LogPSplinePSD/commit/81a2eb3d83e57f158726748667824aab81d13983
.. _a8dd05b: https://github.com/nz-gravity/LogPSplinePSD/commit/a8dd05b9ef73ceabff6996f650ffcbbd314b6746
.. _1d7cb58: https://github.com/nz-gravity/LogPSplinePSD/commit/1d7cb582d2b72530c2d231243817aba2ee7ab9e2
.. _17c9817: https://github.com/nz-gravity/LogPSplinePSD/commit/17c981787a8192eeadd17085860091724f7e93df
.. _a7b47b9: https://github.com/nz-gravity/LogPSplinePSD/commit/a7b47b9792dce420095193c3c6845203c01d51b8
.. _c416cf5: https://github.com/nz-gravity/LogPSplinePSD/commit/c416cf50144cd5234015e6c40d200ab40cd7b615
.. _0de8971: https://github.com/nz-gravity/LogPSplinePSD/commit/0de8971633b83e2db909f76be87cdf45bd573e66
.. _54dc507: https://github.com/nz-gravity/LogPSplinePSD/commit/54dc5074a2440eec50f68eab5e77f7ec0f9b4109
.. _db509f6: https://github.com/nz-gravity/LogPSplinePSD/commit/db509f611271504b0e3a1a7444834c65fa41a054
.. _0081553: https://github.com/nz-gravity/LogPSplinePSD/commit/0081553e399534ca3e37bbd2ef3b14ea26b3e111
.. _0b4d726: https://github.com/nz-gravity/LogPSplinePSD/commit/0b4d726362a8454efae978daf73c3a5098bd35b8
.. _985df28: https://github.com/nz-gravity/LogPSplinePSD/commit/985df28d0e792066ec6b72f580e6b766162d7989
.. _84c44cb: https://github.com/nz-gravity/LogPSplinePSD/commit/84c44cb316132d70ea722daf9f7abd699a173960
.. _9dbf973: https://github.com/nz-gravity/LogPSplinePSD/commit/9dbf97391cccbb7122b3e5f2bfbe09b0e74319cc
.. _ffc9106: https://github.com/nz-gravity/LogPSplinePSD/commit/ffc910674dc42d854863cc2a05fda0d0a6ac8e07
.. _d87cbaf: https://github.com/nz-gravity/LogPSplinePSD/commit/d87cbaf7b852598f77d39500ea8a6dadb75751f6
.. _43e97ad: https://github.com/nz-gravity/LogPSplinePSD/commit/43e97ad2ae8d3195b7d133c3bdd65f4dee48830c
.. _440d523: https://github.com/nz-gravity/LogPSplinePSD/commit/440d52374229cf8fbbcca373cba1c766773c223a
.. _c7b74b6: https://github.com/nz-gravity/LogPSplinePSD/commit/c7b74b6ca8e4e30c1f795927328ea85d2682b418
.. _61352d8: https://github.com/nz-gravity/LogPSplinePSD/commit/61352d86e024f7e5f05ab8e9471618a88af00455
.. _bca32f9: https://github.com/nz-gravity/LogPSplinePSD/commit/bca32f9c372e997146aa45eeb8a96210acbb7fa1
.. _71cdc91: https://github.com/nz-gravity/LogPSplinePSD/commit/71cdc91c39ea63833c6adcb01fbecd74d18e1d0b
.. _16063c3: https://github.com/nz-gravity/LogPSplinePSD/commit/16063c318dd82d8a09a79eea57270cd5043f6c2e
.. _511b1d4: https://github.com/nz-gravity/LogPSplinePSD/commit/511b1d4c91795f539e2bb0a592cb8cbb18f32cf4
.. _560bfd9: https://github.com/nz-gravity/LogPSplinePSD/commit/560bfd93a704daff0f5e267ff350b6460f3298c6
.. _ae66acd: https://github.com/nz-gravity/LogPSplinePSD/commit/ae66acdfbb06dd50043534bb355ddc8d9dd443d0
.. _d0fab7c: https://github.com/nz-gravity/LogPSplinePSD/commit/d0fab7c1edd603d9480db978420e9d22b48c41af
.. _00d3165: https://github.com/nz-gravity/LogPSplinePSD/commit/00d31658f9148afc0a9d4c03686d6e15db96dc69
.. _ef7a655: https://github.com/nz-gravity/LogPSplinePSD/commit/ef7a655327d2d92a3c5a6a2add246e8947a8bcbd
.. _e685d73: https://github.com/nz-gravity/LogPSplinePSD/commit/e685d73c726cf776bf69d3e9860a34aff7c14d6a
.. _fc83a87: https://github.com/nz-gravity/LogPSplinePSD/commit/fc83a87b88966925990b563467591fcf2d46346e
.. _62d5272: https://github.com/nz-gravity/LogPSplinePSD/commit/62d52720ab21743ea186f76ee8b5156cbe74dfe2
.. _1f88ed2: https://github.com/nz-gravity/LogPSplinePSD/commit/1f88ed2c7ea529465ae7f2af5ec6f4c5736f251b
.. _4324316: https://github.com/nz-gravity/LogPSplinePSD/commit/4324316ac00d41f240d9c45e7c29773a57c91bdd
.. _8770f05: https://github.com/nz-gravity/LogPSplinePSD/commit/8770f05b1ab72283bcd21e98dca757e990d621e7
.. _9de972e: https://github.com/nz-gravity/LogPSplinePSD/commit/9de972e0b326b78c04207f2f8cb2f737b70a6a80
.. _4d8d1c5: https://github.com/nz-gravity/LogPSplinePSD/commit/4d8d1c517d93e489d924524ca1c8545d54798c07
.. _fb3279e: https://github.com/nz-gravity/LogPSplinePSD/commit/fb3279e251700b5b72df1095aef9be0648c63031
.. _bfc87e8: https://github.com/nz-gravity/LogPSplinePSD/commit/bfc87e86247dabbfd5b7284618c7471ab0b8e561
.. _4eba004: https://github.com/nz-gravity/LogPSplinePSD/commit/4eba0048323deccf04532345ed987e1bac0b5337
.. _04a3461: https://github.com/nz-gravity/LogPSplinePSD/commit/04a346100a37599b65c8ab7b1a659bb6b1910f33
.. _a8944fb: https://github.com/nz-gravity/LogPSplinePSD/commit/a8944fb438a61341deb4161e7ff3a4b1b64fa762
.. _a23bcc3: https://github.com/nz-gravity/LogPSplinePSD/commit/a23bcc38f97471eeca8dfe2b1cdce8056ae6cb67
.. _6e365bc: https://github.com/nz-gravity/LogPSplinePSD/commit/6e365bc21ed121551f2890aca415f9a7d00cb6e4
.. _eec0cbb: https://github.com/nz-gravity/LogPSplinePSD/commit/eec0cbbdee9e8443274089743673ee4e6aae8eca
.. _f0ede1c: https://github.com/nz-gravity/LogPSplinePSD/commit/f0ede1c5a7835e75c76b34a579c0619c142b683b
.. _1ec3c22: https://github.com/nz-gravity/LogPSplinePSD/commit/1ec3c222307223825248526939a663eed4d59d8e
.. _da95fc0: https://github.com/nz-gravity/LogPSplinePSD/commit/da95fc05b50ef6d5bb042f9800ce91aaea287273
.. _5138c2b: https://github.com/nz-gravity/LogPSplinePSD/commit/5138c2b84ea6c491706501bb0bd5c70a1bbca067
.. _0b2609f: https://github.com/nz-gravity/LogPSplinePSD/commit/0b2609f1b81ad253f5486b08ba62e5bef7d4d595
.. _98a5ea3: https://github.com/nz-gravity/LogPSplinePSD/commit/98a5ea3f7f64f3ffc74d1a47ee1d5185773c2054
.. _fdcf350: https://github.com/nz-gravity/LogPSplinePSD/commit/fdcf350e9a9f62a5a59919678b3b81f9f5363d50
.. _5b0e92c: https://github.com/nz-gravity/LogPSplinePSD/commit/5b0e92c00404ef941fd0888da4fe94e721d7e95d
.. _51156f5: https://github.com/nz-gravity/LogPSplinePSD/commit/51156f529866da1940eca35b63eae665b49cdaa1
.. _074ee64: https://github.com/nz-gravity/LogPSplinePSD/commit/074ee64ccf65360cd5ea9d82c22d292fab4129b2
.. _2102222: https://github.com/nz-gravity/LogPSplinePSD/commit/210222284d368f8e418329cb3266e43b6e76f62f
.. _930b1fc: https://github.com/nz-gravity/LogPSplinePSD/commit/930b1fc801bcab12820903852f7ac1e2921ea473
.. _6c3c970: https://github.com/nz-gravity/LogPSplinePSD/commit/6c3c97087e89f21fdffc1156cf0a709743a9435c
.. _2a68197: https://github.com/nz-gravity/LogPSplinePSD/commit/2a681973a1a9a31dcae6d25438ca6b3f911cc786
.. _50d71bb: https://github.com/nz-gravity/LogPSplinePSD/commit/50d71bb1ca3d02a980310c2c8feaabf3f56b5470
.. _2f6e7d9: https://github.com/nz-gravity/LogPSplinePSD/commit/2f6e7d906fc84d1d6d5f519e4a009379f6b80eb4
.. _efc3406: https://github.com/nz-gravity/LogPSplinePSD/commit/efc340664019a56fcc4d066a226b12ac0dc68a68
.. _d52d2f8: https://github.com/nz-gravity/LogPSplinePSD/commit/d52d2f8d4232fe1b5d067cf348f2710bfa3ea335
.. _39db8c6: https://github.com/nz-gravity/LogPSplinePSD/commit/39db8c61e08355a3858422dc479a9064b09362ce
.. _d56fe94: https://github.com/nz-gravity/LogPSplinePSD/commit/d56fe9471c9a62b83e8fd3f74ae4ac5739780c27
.. _632e7e3: https://github.com/nz-gravity/LogPSplinePSD/commit/632e7e3b22dc77fd8bfea42197d9009a584ca2d0
.. _c0a64eb: https://github.com/nz-gravity/LogPSplinePSD/commit/c0a64ebbb1fbe0a6905978b0171c5172b7961bc5
.. _41de0ee: https://github.com/nz-gravity/LogPSplinePSD/commit/41de0ee8ce196c5123903cf209c130653864a9ee
.. _516b44b: https://github.com/nz-gravity/LogPSplinePSD/commit/516b44ba2c0a912923bc502a47c26409ce91b99b
.. _5331cec: https://github.com/nz-gravity/LogPSplinePSD/commit/5331cecc4c687c18789e1c0dc94d0d5530bf4b05
.. _46b436e: https://github.com/nz-gravity/LogPSplinePSD/commit/46b436efee0d92fe1057139435e316072e2ef846
.. _c10d4fb: https://github.com/nz-gravity/LogPSplinePSD/commit/c10d4fb9f55f01945f1ffcd27d68793fd3b9acb3
.. _c7236ae: https://github.com/nz-gravity/LogPSplinePSD/commit/c7236aee553d7e0bafb3297a3d9b7c669116805e
.. _7b46e28: https://github.com/nz-gravity/LogPSplinePSD/commit/7b46e289a798fc7ba987c06bd7e143833eba7181
.. _b026490: https://github.com/nz-gravity/LogPSplinePSD/commit/b0264908a4291e726375b68058f4e14e43343099
.. _8c43e4a: https://github.com/nz-gravity/LogPSplinePSD/commit/8c43e4ae46bc6971ce19da19fbe3187b9e3ec42b
.. _dc14b0e: https://github.com/nz-gravity/LogPSplinePSD/commit/dc14b0e794cf7fc40bbaad275649873c82dceece
.. _8255ac0: https://github.com/nz-gravity/LogPSplinePSD/commit/8255ac0e24a3524d5778c7043c6d2d12756e848e
.. _229e804: https://github.com/nz-gravity/LogPSplinePSD/commit/229e804674774144f6fc9b441b3dffb16bd67925
.. _8dac000: https://github.com/nz-gravity/LogPSplinePSD/commit/8dac000f0e1eca9a3b110582b5bcbed90714e754
.. _c0bb068: https://github.com/nz-gravity/LogPSplinePSD/commit/c0bb0687994403a61cf53d9e0a81cb5699cb14f4
.. _2a9ee60: https://github.com/nz-gravity/LogPSplinePSD/commit/2a9ee60110a126dc989039598e801830d087cf7c
.. _a7c508a: https://github.com/nz-gravity/LogPSplinePSD/commit/a7c508a9910b4e6ef03d09915ea82ca16619cbef
.. _d2faa1f: https://github.com/nz-gravity/LogPSplinePSD/commit/d2faa1fc1f1cac33f65a2d6bac60882d586e0473
.. _99d3a52: https://github.com/nz-gravity/LogPSplinePSD/commit/99d3a523ffc3fd2360b4224b5aed7e6113ed3cca
.. _a281ded: https://github.com/nz-gravity/LogPSplinePSD/commit/a281ded1378f512c6292d43b7ef7715d043096ab
.. _3fd44b0: https://github.com/nz-gravity/LogPSplinePSD/commit/3fd44b0b8fa5b1534d899e871d250408880c23cd
.. _2a2b1a0: https://github.com/nz-gravity/LogPSplinePSD/commit/2a2b1a0f38f0146e2ed479a5b072abd6f52314ad
.. _f315439: https://github.com/nz-gravity/LogPSplinePSD/commit/f3154391446766461260f47b22937e14201f3a25
.. _2c6f8df: https://github.com/nz-gravity/LogPSplinePSD/commit/2c6f8df995a34d638a5f91c73fcea81d8c587a01
.. _b19197b: https://github.com/nz-gravity/LogPSplinePSD/commit/b19197bd42a1c97a055ef336b66e6357f4c39497
.. _8184c01: https://github.com/nz-gravity/LogPSplinePSD/commit/8184c01519102ad4584da670c573b362de79ee46
.. _d21bc02: https://github.com/nz-gravity/LogPSplinePSD/commit/d21bc02dafc3f1a2c826960d127b670509116084
.. _49d2603: https://github.com/nz-gravity/LogPSplinePSD/commit/49d2603808d5a6f2ca787a41b1c8ee7fda0c2695
.. _2c15bb0: https://github.com/nz-gravity/LogPSplinePSD/commit/2c15bb0101b9e9b4dd7b25b4cd91ba73950df6b2
.. _ac42af7: https://github.com/nz-gravity/LogPSplinePSD/commit/ac42af74612612aef15cbeda3b99bbc7a95e0db4
.. _962e84f: https://github.com/nz-gravity/LogPSplinePSD/commit/962e84f644058d96ebbad3887a6aa39fe24cbd9e
.. _38bbc8a: https://github.com/nz-gravity/LogPSplinePSD/commit/38bbc8aaccfaf44b5207f23a72791467a3513a3f
.. _d6fe80c: https://github.com/nz-gravity/LogPSplinePSD/commit/d6fe80cc34d9206e8272610e52a04978a9119511
.. _44a4f69: https://github.com/nz-gravity/LogPSplinePSD/commit/44a4f691f8ea232261d20fc89663621440398852
.. _e67261d: https://github.com/nz-gravity/LogPSplinePSD/commit/e67261da639955579c22e001e6c794733e6c65e6
.. _d5c5144: https://github.com/nz-gravity/LogPSplinePSD/commit/d5c5144db0bb0b95d0c07402651b393c2677b36a
.. _3333e64: https://github.com/nz-gravity/LogPSplinePSD/commit/3333e64f9988f9628d39554171230744c9632d95
.. _f48529f: https://github.com/nz-gravity/LogPSplinePSD/commit/f48529fa69d7125f7f6144e1fe0bd3a0cdd11986
.. _110f31b: https://github.com/nz-gravity/LogPSplinePSD/commit/110f31b24b73deebc1762461d0360e2bdcb49d52
.. _c4319af: https://github.com/nz-gravity/LogPSplinePSD/commit/c4319af8ebaca6ca1de504a8a1a133bab78d72fa
.. _f757b98: https://github.com/nz-gravity/LogPSplinePSD/commit/f757b98cbf99cb1d2b0eb387b554c2f77c0bf811
.. _2ba88a5: https://github.com/nz-gravity/LogPSplinePSD/commit/2ba88a5cc3845122166b921e4ffcd11a3505f559
.. _30e3d9a: https://github.com/nz-gravity/LogPSplinePSD/commit/30e3d9a220d8af34806b425e93c89a29ba49b051
.. _d11cc7e: https://github.com/nz-gravity/LogPSplinePSD/commit/d11cc7ef3be2cabddd53dfc2b157db5451a5172d
.. _ebb8025: https://github.com/nz-gravity/LogPSplinePSD/commit/ebb8025a48947118095606d1903e5ccbff2d0c21
.. _3463ff4: https://github.com/nz-gravity/LogPSplinePSD/commit/3463ff4cfc7e6fdac48aba459138ddab0a062db5
.. _403c0e6: https://github.com/nz-gravity/LogPSplinePSD/commit/403c0e60cbfccd71a091994c20258b541fc33c55
.. _43a26e7: https://github.com/nz-gravity/LogPSplinePSD/commit/43a26e7dba8f63b49be6bd8bbf097dc2f4aadf89
.. _accfd48: https://github.com/nz-gravity/LogPSplinePSD/commit/accfd48a2f8dd209565bab741edf74223f1ebf5f
.. _1c5734d: https://github.com/nz-gravity/LogPSplinePSD/commit/1c5734d670c32ea3ba02f18171fca660a7cdbe71
.. _2355be6: https://github.com/nz-gravity/LogPSplinePSD/commit/2355be6b68e93e345f4fcca45cf603a983161d02
.. _f31c930: https://github.com/nz-gravity/LogPSplinePSD/commit/f31c9302d78d216052fa13961c85dda32d957a10
.. _a706ab8: https://github.com/nz-gravity/LogPSplinePSD/commit/a706ab803f75b0b1eab00a61cbb339257a9d289b
.. _8bb77a6: https://github.com/nz-gravity/LogPSplinePSD/commit/8bb77a6233c6a5e84bb0b3a0b09349a7d0fcabf0
.. _2d0c3ea: https://github.com/nz-gravity/LogPSplinePSD/commit/2d0c3ea420cd2f8275ef7057fe1e38af1ff08386
.. _823dd25: https://github.com/nz-gravity/LogPSplinePSD/commit/823dd2573f42651b76e5364a8c16c1a9c765eed2
.. _60af11f: https://github.com/nz-gravity/LogPSplinePSD/commit/60af11fbd0cef7f7c99474a10ef597821487700b
.. _1493808: https://github.com/nz-gravity/LogPSplinePSD/commit/1493808c471038e23fa20a5700ae92751de2cca9
.. _1fd93ba: https://github.com/nz-gravity/LogPSplinePSD/commit/1fd93ba55df0a965dda17e14724f9a1937dd870c
.. _65bc35f: https://github.com/nz-gravity/LogPSplinePSD/commit/65bc35f2605b8be38ac662b652e976351d7374ed
.. _53d8791: https://github.com/nz-gravity/LogPSplinePSD/commit/53d8791e1f9acd416abc12ffaaf1ddc3d55ff2eb
.. _707d6ea: https://github.com/nz-gravity/LogPSplinePSD/commit/707d6ea5a897f5d91fb0faa17d38a4ffba4c2d4d
.. _13618a4: https://github.com/nz-gravity/LogPSplinePSD/commit/13618a4a50c385ad28853e41731169b21b020bef
.. _3339a57: https://github.com/nz-gravity/LogPSplinePSD/commit/3339a575051ef4a4dfcbfca3833153eeeb553400
.. _23deb59: https://github.com/nz-gravity/LogPSplinePSD/commit/23deb59015b22a9c5620e45daa36e1d7b7e7d647
.. _469b526: https://github.com/nz-gravity/LogPSplinePSD/commit/469b5264a10d8d741edde5a2154026f3f13039f6
.. _b893126: https://github.com/nz-gravity/LogPSplinePSD/commit/b89312666b05d39f329bdf06bc2747188bd2e9bd
.. _a311488: https://github.com/nz-gravity/LogPSplinePSD/commit/a3114885bcd5daa742682b73768ef5ac28942283
.. _a185c83: https://github.com/nz-gravity/LogPSplinePSD/commit/a185c8314fd0e30cf253ed417f5208224cfc4905
.. _3631264: https://github.com/nz-gravity/LogPSplinePSD/commit/3631264aa1c014df916ddf91f9f01af0ce2ca09b
.. _9363997: https://github.com/nz-gravity/LogPSplinePSD/commit/9363997cac6815f66764518268c206e19ed581e6
.. _ad1b986: https://github.com/nz-gravity/LogPSplinePSD/commit/ad1b9866551177de9c157e1dfd21f7493c262cc6
.. _deddfb5: https://github.com/nz-gravity/LogPSplinePSD/commit/deddfb5705ddf2508bae32c6cb888f6b1c368efb
.. _3a5eedb: https://github.com/nz-gravity/LogPSplinePSD/commit/3a5eedb1816eea6b53741ff701fb957050696737
.. _f59876d: https://github.com/nz-gravity/LogPSplinePSD/commit/f59876d95a168175b497f69117967af7a2bb48f7
.. _52c6e64: https://github.com/nz-gravity/LogPSplinePSD/commit/52c6e64ef2837a097f65e50d8ac039c77d634401
.. _7ec4d48: https://github.com/nz-gravity/LogPSplinePSD/commit/7ec4d48bda38d6ababcde327fa4c5870a9c70c8e
.. _a5e664b: https://github.com/nz-gravity/LogPSplinePSD/commit/a5e664bb53731e33241285ce0a1dd4755a35cc00
.. _60d1a9c: https://github.com/nz-gravity/LogPSplinePSD/commit/60d1a9ce5fe531fc6bb8d877a9bfd76dd47a5828
.. _2d55780: https://github.com/nz-gravity/LogPSplinePSD/commit/2d557800b7cb61b2d50e540213dd71a5e64bf529
.. _db6afdf: https://github.com/nz-gravity/LogPSplinePSD/commit/db6afdfd48d261c292c6151d2f12461a41382c07
.. _783ec66: https://github.com/nz-gravity/LogPSplinePSD/commit/783ec66015c119b21347bedd1b2982eed8537442
.. _38eebe4: https://github.com/nz-gravity/LogPSplinePSD/commit/38eebe40bb827bf7f9128e5866340ffb2f4cae91
.. _fff30e3: https://github.com/nz-gravity/LogPSplinePSD/commit/fff30e3e7858296aaa76d80a3b428727c3ed3bce
.. _972b2bb: https://github.com/nz-gravity/LogPSplinePSD/commit/972b2bb3b5e4224d4e0603236a2dac38af200a8c
.. _c724b0e: https://github.com/nz-gravity/LogPSplinePSD/commit/c724b0e01476adf13022414b258e1cc76ed9d2f1
.. _b5235ef: https://github.com/nz-gravity/LogPSplinePSD/commit/b5235ef6e712ad775d8116c935f8fa7141685829
.. _b99320d: https://github.com/nz-gravity/LogPSplinePSD/commit/b99320d6b65d7ded5fca1a9f11f8778aa10073ca
.. _7255d97: https://github.com/nz-gravity/LogPSplinePSD/commit/7255d9798bfdf2a1ade9e834792456598aaaade3
.. _52da30f: https://github.com/nz-gravity/LogPSplinePSD/commit/52da30f7ebd9b1bfcaf04287d6ddfeaa39b066dc
.. _837ed9c: https://github.com/nz-gravity/LogPSplinePSD/commit/837ed9c9b815cf9473143488b4f676716a63691e
.. _e692e08: https://github.com/nz-gravity/LogPSplinePSD/commit/e692e08f97fc2117ffd1a0db07fefe5d13f5d2b2
.. _dbda75e: https://github.com/nz-gravity/LogPSplinePSD/commit/dbda75ee402e042b3b5d45a58053e7e6beccea10
.. _f34043e: https://github.com/nz-gravity/LogPSplinePSD/commit/f34043e9fab394bbbe0578e8331a66bb0cddb76d
.. _9a2b816: https://github.com/nz-gravity/LogPSplinePSD/commit/9a2b8169b4886af2e96db0bcdbf2a7485f812e60
.. _d4543e0: https://github.com/nz-gravity/LogPSplinePSD/commit/d4543e0c0aed31848e2784bb065c16846012d228
.. _f3e8b98: https://github.com/nz-gravity/LogPSplinePSD/commit/f3e8b98de27709b2f2ea7aeb771be38fba694c7f
.. _bb21312: https://github.com/nz-gravity/LogPSplinePSD/commit/bb2131242755ff0e42f1b9b3b9e0900e4a14b41c
.. _751683c: https://github.com/nz-gravity/LogPSplinePSD/commit/751683ceb277f45db8430cffc31931d51de12237
.. _7263da6: https://github.com/nz-gravity/LogPSplinePSD/commit/7263da6073bde56f16644ad7fc8a8817f7d6c1c0
.. _b06de50: https://github.com/nz-gravity/LogPSplinePSD/commit/b06de50fe230021ddfb515cf812df55dea7d6b71
.. _f3bed24: https://github.com/nz-gravity/LogPSplinePSD/commit/f3bed24eb86d2808ef97a33e25a09d8e2066f829
.. _91ffcba: https://github.com/nz-gravity/LogPSplinePSD/commit/91ffcba35b5de1e0cf81d4fc440225e560b6377f
.. _72b59cc: https://github.com/nz-gravity/LogPSplinePSD/commit/72b59ccaa7b6a019f09cc9668c10cf143282c976
.. _2beee51: https://github.com/nz-gravity/LogPSplinePSD/commit/2beee513c915ca08bd453ddcb014be3dbe045acc
.. _5674e9f: https://github.com/nz-gravity/LogPSplinePSD/commit/5674e9f15c7594739135bd5d10347a30f0095692
.. _bdb03f4: https://github.com/nz-gravity/LogPSplinePSD/commit/bdb03f4f2cffa533276a9b8f5ef42448a1520e05
.. _01b411f: https://github.com/nz-gravity/LogPSplinePSD/commit/01b411fa03be2d087a18a1016ebf570d15600859
.. _be1ac26: https://github.com/nz-gravity/LogPSplinePSD/commit/be1ac264474d6c92198bbda4a8804e12a448860b
.. _f0f48db: https://github.com/nz-gravity/LogPSplinePSD/commit/f0f48dbb8fb1c6fef16a21225cf632b807dfeb4b
.. _4f908b5: https://github.com/nz-gravity/LogPSplinePSD/commit/4f908b55c6227d02bbb46b624a462d5269590433
.. _286a2b4: https://github.com/nz-gravity/LogPSplinePSD/commit/286a2b40d74cecab8da3205063078fb1bd80fbf8
.. _70bb701: https://github.com/nz-gravity/LogPSplinePSD/commit/70bb70127af199b0093c72bf5ca753ae979967d4
.. _f5fd9a3: https://github.com/nz-gravity/LogPSplinePSD/commit/f5fd9a327a4c25e29d9de7c0d4f1a0c1ee6789e4
.. _8fe817e: https://github.com/nz-gravity/LogPSplinePSD/commit/8fe817eab0d5ad05b77fc77252968feb3dbbdf4b
.. _8353456: https://github.com/nz-gravity/LogPSplinePSD/commit/8353456f9cfbbcdfb04db69b0cdd1c6f137bcaaa
.. _9c5b39d: https://github.com/nz-gravity/LogPSplinePSD/commit/9c5b39d95f3a3c1ea2c6a8026cdcbb9e2d0a75cc
.. _0491954: https://github.com/nz-gravity/LogPSplinePSD/commit/04919546628f6884fea87647d0916e7e0d73bd05
.. _6724575: https://github.com/nz-gravity/LogPSplinePSD/commit/67245755fe458b9f44336dcb0a9918fd118fab09
.. _319faef: https://github.com/nz-gravity/LogPSplinePSD/commit/319faefcc9c5ac1542669411b7b9efee6bdd3d20
.. _83856b1: https://github.com/nz-gravity/LogPSplinePSD/commit/83856b13ae25f86520a5e01da1b603db66a8a909
.. _69da535: https://github.com/nz-gravity/LogPSplinePSD/commit/69da5358498837347a7c491e8abd3bd604d544ec
.. _8f35eca: https://github.com/nz-gravity/LogPSplinePSD/commit/8f35eca0f65308258b6b59020994e19d0172f40c
.. _166cf77: https://github.com/nz-gravity/LogPSplinePSD/commit/166cf774f3d7fa82f2e6ee04c506430518aefc99
.. _ebb13cb: https://github.com/nz-gravity/LogPSplinePSD/commit/ebb13cba007f39dc5152bb3a70f4f2c4b6dec964
.. _e05c7b9: https://github.com/nz-gravity/LogPSplinePSD/commit/e05c7b9e4612247733fcce580d409a63623d7176
.. _174247d: https://github.com/nz-gravity/LogPSplinePSD/commit/174247da1c80fbcf564de0e8ff29c585ad489a46
.. _0239d93: https://github.com/nz-gravity/LogPSplinePSD/commit/0239d935e56bca0c570b890eeeae26d6f7dcead8
.. _3a8796d: https://github.com/nz-gravity/LogPSplinePSD/commit/3a8796d76743c1aec49a7e990510efeb79ee5a93
.. _933f6d6: https://github.com/nz-gravity/LogPSplinePSD/commit/933f6d63d04b50638c0f544841a2aecff684aac3
.. _11ff71f: https://github.com/nz-gravity/LogPSplinePSD/commit/11ff71f37632fb335d49a86daea933b85f5d87ba
.. _86ce4ef: https://github.com/nz-gravity/LogPSplinePSD/commit/86ce4efd6e16a987ec220d8894ef47dafcb778fb
.. _d9bceda: https://github.com/nz-gravity/LogPSplinePSD/commit/d9bcedaa2808f05807e7af93d2a6ebeffb6a822d
.. _c65f035: https://github.com/nz-gravity/LogPSplinePSD/commit/c65f0355763af6e2d380ca6414622344e198a337
.. _83aed50: https://github.com/nz-gravity/LogPSplinePSD/commit/83aed50b33a18a7098f5357c71ec908be7536c7c
.. _0d096ca: https://github.com/nz-gravity/LogPSplinePSD/commit/0d096ca4ba5e0adecc3a0ec9656446cf8d1abe07
.. _5b759a8: https://github.com/nz-gravity/LogPSplinePSD/commit/5b759a8b72b11dd6754502f4593bd71a1842e0e9
.. _730adec: https://github.com/nz-gravity/LogPSplinePSD/commit/730adecae7c97b91449c5433fbde7c9396e0c99e
.. _874f652: https://github.com/nz-gravity/LogPSplinePSD/commit/874f652f72969a76fa369b0aecce0fe84babf37b
.. _8cf0897: https://github.com/nz-gravity/LogPSplinePSD/commit/8cf089760953a3f7effb6f493e0c61dd85d6a692
.. _ef296d5: https://github.com/nz-gravity/LogPSplinePSD/commit/ef296d516e40cdb308b773aafb73a828a408fbe4
.. _d507ae2: https://github.com/nz-gravity/LogPSplinePSD/commit/d507ae270a52d628b6478543bcbca9e48a8493cf
.. _555f786: https://github.com/nz-gravity/LogPSplinePSD/commit/555f78655cacb5368f07b5dd8b8d7fe70f757023
.. _0f7efd7: https://github.com/nz-gravity/LogPSplinePSD/commit/0f7efd78287ad6b876436713899c45187ac7757d
.. _fbdd866: https://github.com/nz-gravity/LogPSplinePSD/commit/fbdd8661960cd1d38f55cb013bb0184c463ecc51
.. _3c51d4e: https://github.com/nz-gravity/LogPSplinePSD/commit/3c51d4ec25a323d869b956e8577adff52658d3fd
.. _1f0b992: https://github.com/nz-gravity/LogPSplinePSD/commit/1f0b9923e993968e456d2bc1361192b4227ac950
.. _f354738: https://github.com/nz-gravity/LogPSplinePSD/commit/f35473827d7c9204ba7047dbb330979935720928
.. _86151ef: https://github.com/nz-gravity/LogPSplinePSD/commit/86151efddab0ee13f31604d9f15e4ec9a9bc5667
.. _8acdc0d: https://github.com/nz-gravity/LogPSplinePSD/commit/8acdc0dc7bf8ef07e348aeeb8244bdf6741101a8
.. _353e42d: https://github.com/nz-gravity/LogPSplinePSD/commit/353e42d32bc67b38a2174216a9be93004ac0aaae
.. _94ccb8c: https://github.com/nz-gravity/LogPSplinePSD/commit/94ccb8c1b1780d44ceef92da10c49370092278b7
.. _2898850: https://github.com/nz-gravity/LogPSplinePSD/commit/28988504e770b4e0493a1739018f93f51103c3d0
.. _a0844bc: https://github.com/nz-gravity/LogPSplinePSD/commit/a0844bce9d753d52e9f0cb62015f8df8410d9b52
.. _3af80a4: https://github.com/nz-gravity/LogPSplinePSD/commit/3af80a4f3981485b24ef0be67596eb362d3b5c9f
.. _5f27647: https://github.com/nz-gravity/LogPSplinePSD/commit/5f276470babeb49266f6808b2f5b865217e29af1
.. _1d316ad: https://github.com/nz-gravity/LogPSplinePSD/commit/1d316adb394eb1a37243358cd969fab64d74ed33
.. _88887d5: https://github.com/nz-gravity/LogPSplinePSD/commit/88887d555478f73f6830b258a07ec75693f45abd
.. _25af3b0: https://github.com/nz-gravity/LogPSplinePSD/commit/25af3b0413c4140f0e04e2fe57e1a5a74686d76a
.. _0fd0481: https://github.com/nz-gravity/LogPSplinePSD/commit/0fd04813a9be0c32a38a71c641ffd60840407560
.. _b9a20ae: https://github.com/nz-gravity/LogPSplinePSD/commit/b9a20ae78dec2252d03ffd2bcd06c9790c83e62e
.. _e754a09: https://github.com/nz-gravity/LogPSplinePSD/commit/e754a0960c1d8c556105b0193cdc9cb46f62a79b
.. _217365f: https://github.com/nz-gravity/LogPSplinePSD/commit/217365f4051b3f8d28e80e6754df8b773ee5b84d
.. _dcb1434: https://github.com/nz-gravity/LogPSplinePSD/commit/dcb1434825f86f95a72556d90081cbffd27101ce
.. _39b99d3: https://github.com/nz-gravity/LogPSplinePSD/commit/39b99d3f15c4ff7442c2db957769eb2d5af0b07c
.. _cb240a5: https://github.com/nz-gravity/LogPSplinePSD/commit/cb240a542b09b8117ecc0119f96d9390c86f0f3c
.. _0985f42: https://github.com/nz-gravity/LogPSplinePSD/commit/0985f420bc95b95f6167e57170d00e9dae9039f1
.. _73f1a91: https://github.com/nz-gravity/LogPSplinePSD/commit/73f1a915cf9d07464fbb25d83504f26cbde7ddd1
.. _ed71380: https://github.com/nz-gravity/LogPSplinePSD/commit/ed713804184a8bb7507afabf7cb54dea1ca30c5b
.. _8751c19: https://github.com/nz-gravity/LogPSplinePSD/commit/8751c19047ccac710e37af1e0ff9cb15f49f2736
.. _6744a7d: https://github.com/nz-gravity/LogPSplinePSD/commit/6744a7daffba22a0b113c4fc292cad2747efa845
.. _00c329c: https://github.com/nz-gravity/LogPSplinePSD/commit/00c329c84dfdae659eb88f411548bfe25cb25a27
.. _dee569a: https://github.com/nz-gravity/LogPSplinePSD/commit/dee569a2dc93ca403b00b9cb53e5d3444e9863ef
.. _0be11e4: https://github.com/nz-gravity/LogPSplinePSD/commit/0be11e4ad2470c5c3db0dfbfb0c973cde0800b81
.. _5a9e33f: https://github.com/nz-gravity/LogPSplinePSD/commit/5a9e33f99f6aa3c1025cd3b77c8f58f2e2ef05e1
.. _afb258c: https://github.com/nz-gravity/LogPSplinePSD/commit/afb258c7973464d83ea109bc4c5a17de70ffae6b
.. _6471534: https://github.com/nz-gravity/LogPSplinePSD/commit/6471534808a1bb246bf536e7c3ead664719fb2cf
.. _82589dd: https://github.com/nz-gravity/LogPSplinePSD/commit/82589dd5ca8a8aa9e2205bad1d2bae49258d53a0
.. _fd77e9b: https://github.com/nz-gravity/LogPSplinePSD/commit/fd77e9bd29a31d929c7b6b1c8bf7afce243dc072
.. _f12131f: https://github.com/nz-gravity/LogPSplinePSD/commit/f12131ffeba951762276bdf163f91f8e1e9fd03b
.. _449a7a9: https://github.com/nz-gravity/LogPSplinePSD/commit/449a7a9a0d4e1465a19b8424e13a759ee52fd9d9
.. _012c481: https://github.com/nz-gravity/LogPSplinePSD/commit/012c481b41ed3f43361732ef1f1bb06c7e9c7fc1
.. _8afc400: https://github.com/nz-gravity/LogPSplinePSD/commit/8afc400bfd47a930b85593b735404252edc2427d
.. _959a3e5: https://github.com/nz-gravity/LogPSplinePSD/commit/959a3e566d260141196b6e7c1c85c6caa603acd2
.. _eadb594: https://github.com/nz-gravity/LogPSplinePSD/commit/eadb5944da4754bcdf349add61ad18e1b7db32d2
.. _f8b5521: https://github.com/nz-gravity/LogPSplinePSD/commit/f8b55216754040b615c872466f69f98f3d147f4f
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
