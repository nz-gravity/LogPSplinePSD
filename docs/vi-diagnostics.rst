VI diagnostics and saved guides
===============================

``StationaryConfig`` and ``PowerConfig`` expose ``vi_diagnostics`` and
``vi_early_stopping``. Guide-density diagnostics are opt-in; requesting them
keeps the existing training and posterior random streams and legacy stopping
rule. Choose guide and optimization settings against a diagnosed NUTS
reference for the same target. The values below illustrate the API and are
not a validated accuracy preset.

.. code-block:: python

   from log_psplines import PowerConfig, fit

   config = PowerConfig(
       method="vi",
       vi_guide="mvn",
       vi_steps=5000,
       vi_posterior_draws=4096,
       vi_early_stopping=False,
       vi_diagnostics={
           "seeds": (8101, 8102, 8103),
           "num_particles": 4096,
           "chunk_size": 128,
           "checkpoint_steps": (1000, 5000),
       },
   )
   result = fit(data, config, model=spline)
   state = result.vi.diagnostics
   state.save("guide-checkpoint")
   result.to_netcdf("result.nc")

Stationary Cholesky factors retain separate guide checkpoints. Independent
factor diagnostics use distinct seeds, and aligned joint log ratios are
summed before fitting the joint tail diagnostic. Shared latent sites are
rejected. Multivariate timing durations and ``steps_run`` sum all factor
fits; ``steps_run`` counts total updates across channels. Per-factor timing
entries are also retained.

Exploring model choices
-----------------------

VI can screen choices such as spline knots using approximate posterior
spectrum curves, followed by NUTS validation of promising choices. Scalar
and stationary VI use the same prepared Bayesian model as NUTS. Guide
family, optimization steps, learning rate, draw count and stopping settings
change the inference procedure; they do not replace the likelihood,
smoothing prior or spectrum reconstruction.

Cheap exploration needs its own speed and predictive-accuracy benchmark.
The long recipes used in the validation studies prioritize agreement and
do not establish a faster public preset. Compare prediction errors and
model-choice rankings as well as elapsed time. Keep guide-density diagnostics
optional when measuring exploration cost, and report their overhead
separately. Public configs expose the basic VI controls; training particle
counts and callable learning-rate schedules are currently available through
the lower-level ``fit_vi`` API.

Saved Gaussian guides
---------------------

.. code-block:: python

   from log_psplines.diagnostics.variational import (
       VIDiagnosticState, rebuild_guide,
   )

   state = VIDiagnosticState.load("guide-checkpoint")
   guide = rebuild_guide(
       state,
       prepared_model,
       target_fingerprint=matching_fingerprint,
   )
   q_unconstrained = guide.get_posterior(state.params)

Reconstruction requires the identical prepared model, matching target
fingerprint, NumPyro version, latent site order, shapes, dtypes and support
transform schema. The recipe supports built-in diagonal, low-rank and full
Gaussian guides. Numerical checkpoints do not reconstruct model code,
flow/custom guides, or optimizer-state resumption.

``packed_log_densities`` evaluates the NumPyro potential and matching guide
density in unconstrained coordinates, including prior correction factors and
one support Jacobian. Deterministic coefficients and spectrum values are not
additional latent density dimensions. Constant weights, nonfinite densities,
insufficient tails and unsupported diagnostics have distinct statuses. Low
Pareto k concerns importance proposals; it does not establish raw VI
posterior accuracy.

Prepared models and optimization
--------------------------------

The lower-level ``log_psplines.inference.vi.fit_vi`` accepts a prepared model,
``VIDiagnosticConfig``, explicit initialization, and
``optimization_particles`` independently of diagnostic particle counts.
``optimizer_lr`` may be an Optax schedule; numerical
``optimizer_lr_metadata`` describes the schedule in persisted records without
serializing callable code. Supply a fingerprint covering closed-over data,
bases, priors, normalization and source identity. An absent fingerprint is
recorded as unavailable and cannot pass a same-target comparison.

``checkpoint_callback`` requires a diagnostics configuration and receives the
selected diagnostic checkpoints. An empty ``checkpoint_steps`` still saves
the final state. Requesting ``stopping_rule="noise_aware"`` additionally uses
paired fixed-seed objective changes, Monte Carlo noise and location stability.
It requires a guide with posterior mean and variance; flow guides do not
support this stopping rule. Ordinary flow optimization remains available.
Stopping heuristics assess optimization, not posterior accuracy.
