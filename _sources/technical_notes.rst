Technical Notes
===============

These pages document the assumptions and implementation details that matter for
maintaining or extending the inference code.

Core Invariants
---------------

- PSD diagonal entries must stay strictly positive.
- Multivariate spectral matrices must stay Hermitian positive definite at each
  retained frequency.
- Coherence is derived from the spectral matrix and should remain bounded by
  ``[0, 1]`` up to numerical tolerance.
- Randomness should be controlled through explicit seeds or JAX PRNG keys.
- Shape conventions should be documented near public functions and checked in
  non-JIT code paths.

Implementation Map
------------------

``log_psplines.mcmc``
   High-level ``run_mcmc`` convenience function.

``log_psplines.pipeline``
   Pipeline construction, preprocessing, VI, NUTS, saving, and evidence
   estimation.

``log_psplines.datatypes``
   Time-domain and frequency-domain containers.

``log_psplines.psplines``
   Spline basis construction, P-spline penalties, knot placement, and
   multivariate PSD reconstruction.

``log_psplines.preprocessing``
   Frequency selection, Wishart preprocessing, and coarse graining.

``log_psplines.arviz_utils``
   Loading, saving, and extracting posterior spectral summaries.

``log_psplines.diagnostics`` and ``log_psplines.plotting``
   Convergence checks, error metrics, and visual summaries.

Detailed Pages
--------------

- :doc:`conventions`
- :doc:`coarse_grain`
- :doc:`multivar_blocked_nuts`
- :doc:`design_shrinkage`
