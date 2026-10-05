Power data with a reference spectrum
====================================

Supply a time-frequency power grid, a fixed reference spectrum, and optional
simulation truth to the public ``fit`` function. The package builds the scalar
tensor-product log-P-spline from ``PowerConfig``. Domain-specific construction
of the reference and truth belongs in the calling analysis.

.. code-block:: python

   import numpy as np

   from log_psplines import PowerConfig, PowerData, PowerPartition, fit

   data = PowerData(
       power=summed_power,
       counts=component_counts,
       time=time_grid,
       frequency=frequency_grid,
       units="WDM coefficient variance",
   )
   config = PowerConfig(
       n_interior_knots_time=8,
       n_interior_knots_freq=12,
       roughness_scale=10.0,
       centered=True,
       n_warmup=500,
       n_samples=1000,
   )
   result = fit(
       data,
       config,
       reference=reference_power,
       true_psd=truth_power,
   )
   result.save("results/example")

``reference_power`` and ``truth_power`` must be positive arrays matching the
native ``(time, frequency)`` grid and the units of ``data.power / data.counts``.
The reference is fixed during inference: the spline estimates a smooth log
correction to it. Truth is excluded from the likelihood and used only for
posterior checks and plots. The native reference is restored to every posterior
spectrum draw and truth is retained in ``result.truth`` and the saved NetCDF.

With a ``PowerPartition``, each native power is divided by its own reference
value before pooling. The exact component counts are pooled separately. The
result is reconstructed on the original grid. Use ``reference=None`` for a
free log-P-spline surface, or pass an explicit ``LogPSpline`` or
``ANOVALogPSpline`` model when its basis or decomposition needs to be set by
the caller.

The WDM adapter returns coefficient-variance powers, not PSD per Hz. Convert
the supplied reference and truth to those same units before calling ``fit``;
the package does not infer a transform calibration from an analytic PSD.

ANOVA corrections
~~~~~~~~~~~~~~~~~

Set ``PowerConfig(structure="anova", interaction_scale=0.5)`` to use
``log S = log reference + g(f) + eta(t, f)``. The interaction is centered on
the full native time grid. ``roughness_scale`` controls the HalfNormal
smoothing scales; ``interaction_scale`` controls the interaction amplitude.
This uses the same ``fit(data, config, reference=..., true_psd=...)`` call.

External parametric spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a physical model, supply a deterministic JAX function and independent
scalar priors. The package constructs the Bayesian model and power likelihood:

.. code-block:: python

   import jax.numpy as jnp
   from numpyro import distributions as dist
   from log_psplines import ParametricSpectrum

   template = jnp.asarray(reference_power)
   model = ParametricSpectrum(
       spectrum=lambda parameters, section: (
           jnp.exp(parameters["log_scale"]) * template[:, section]
       ),
       priors={"log_scale": dist.Normal(0.0, 0.5)},
       initial_values={"log_scale": 0.0},
   )
   result = fit(data, config, model=model, true_psd=truth_power)

The function returns a positive variance grid matching ``PowerData``.
``(time, frequency, channel)`` powers with ``channels=("sensor_1", "sensor_2")``
use independent channel likelihoods and shared parameters. Off-diagonal
covariances are not fitted. Data, deterministic templates, response projection
and pooling must already agree; ``reference`` and ``partition`` are therefore
not accepted with ``ParametricSpectrum``. Domain-specific physics stays in
external scripts. Scalar tensor and ANOVA models still take scalar powers.

Large-grid results
~~~~~~~~~~~~~~~~~~

``PowerConfig(spectrum_draws=2, spectrum_chunk_size=4)`` retains every
parameter draw but stores only two spectral draws per chain. The 5/50/95
percentiles, arithmetic mean and geometric mean in ``result.spectrum_summary``
use **all** draws, evaluated in frequency chunks. ``result.quantiles()`` and
plots use those summaries. ``result.psd`` and ``result.spectrum`` contain the
preview only. Other percentiles require reconstruction from the saved draws.
The default ``spectrum_draws=None`` retains all spectral draws.

Spline results save their bases and reference in ``result.model_data``.
``log_psplines.models.reconstruction.power_draws_from_basis`` reconstructs all
scalar draws, optionally for a frequency slice. Parametric reconstruction
requires the caller's deterministic function and external model inputs, which
should be saved alongside the result. NetCDF persistence preserves the
separate lengths of posterior chains and spectral previews.

Scalar variational inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``PowerConfig(method="vi", vi_steps=5000, vi_lr=0.01,
vi_guide="diag", vi_posterior_draws=256)`` for scalar spline powers on a
rectangular grid or at exact paired coordinates. ``lowrank:10`` is also
available; its rank is capped at the latent dimension. NUTS remains the
default. Both methods use the same prepared likelihood, tensor prior and
coefficient reconstruction. Parametric spectra currently support NUTS only.

VI returns one chain of constrained draws and ``result.vi`` loss/timing
information, with no fabricated NUTS sample statistics. The VI diagnostics,
model bases, observations and native units survive ``PSDResult.to_netcdf`` /
``PSDResult.from_netcdf``. Guide and optimizer checks against diagnosed NUTS
are necessary before interpreting VI uncertainty. See :ref:`vi-diagnostics`
for opt-in guide-density diagnostics, checkpoints and their limits.
