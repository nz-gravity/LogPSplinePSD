Time-varying WDM power fits
===========================

The package accepts prepared arrays for each scalar channel. A/E construction,
gap tapering, and WDM guard masks are the caller's responsibility. Fit A and E
independently. No LISA data format or LISA software is required.

``power[t, f]`` is a sum of squared real WDM coefficients in **coefficient
variance** units. ``counts[t, f]`` is the number of retained real components
represented by that sum. A native WDM coefficient has count 1. Excluded cells
have both power and count zero. The optional ``wdm_periodogram`` helper creates
native powers from a uniformly sampled single-channel ``TimeSeries``.
If the target is the paper's one-sided PSD per Hz, multiply every native
coefficient square by ``2 * dt / N`` (where ``N`` is the WDM frequency
resolution parameter) before constructing ``PowerSpectrum``, and label the
units accordingly. Apply that conversion once; the likelihood does not rescale
input power.

.. code-block:: python

   import numpy as np
   from log_psplines import (
       LogPSpline, PowerSpectrum, PowerSplineConfig, SplineBasis,
       fit, mask_power, select_power_partition,
   )

   # The caller has already prepared one channel and supplied a boolean
   # WDM-cell mask. The arrays have shapes (time, frequency).
   with np.load("prepared_a.npz") as arrays:
       native = PowerSpectrum(
           arrays["power"], arrays["counts"],
           arrays["frequency"], arrays["time"],
           units="WDM coefficient variance",
       )
       retained = mask_power(native, arrays["mask"].astype(bool))
       pilot = arrays["training_pilot_log_psd"]

   # The pilot selects boundaries only; the likelihood uses retained.power.
   partition = select_power_partition(
       pilot, retained.time, counts=retained.counts,
       time_bin=4, max_frequency_bin=24, max_log_range=0.25,
   )
   model = LogPSpline(
       frequency=SplineBasis.from_grid(
           retained.frequency / retained.frequency[-1],
           n_interior_knots=8,
       ),
       time=SplineBasis.from_grid(
           (retained.time - retained.time[0])
           / (retained.time[-1] - retained.time[0]),
           n_interior_knots=8,
       ),
   )
   result = fit(
       retained,
       PowerSplineConfig(roughness_scale=10),
       model=model,
       partition=partition,
   )
   result.to_netcdf("a_fit.nc")

The example knot count is deliberately small. Paper-scale analyses can supply
frozen nonuniform frequency knots through ``SplineBasis.from_grid`` with
``interior_knots=...``. Knot placement and
likelihood partitioning are separate choices. A paper-style pilot uses 32
training-only time profiles, a 31-channel frequency median, and the partition
settings shown above. Store or generate that pilot outside the likelihood.

The coarse likelihood evaluates one smooth PSD value per rectangular block.
Its summed power and exact count match the native data under the assumption
that the PSD is effectively constant inside each block. Check narrow features
and posterior coverage against uncompressed fits before using larger bins.
The result stores both the coarse observed statistics and partition starts;
``result.psd`` is reconstructed on the native time/frequency grid. The
diagonal WDM likelihood is a locally stationary composite approximation.
