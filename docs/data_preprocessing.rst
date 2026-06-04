Data and Preprocessing
======================

Accepted Inputs
---------------

The high-level pipeline accepts either time-domain data or precomputed
frequency-domain statistics.

``MultivariateTimeseries``
   Time-domain samples with shape ``(n, p)``. A one-dimensional input is
   promoted to ``(n, 1)``. The sampling frequency is inferred from ``t``.

``MultivarFFT``
   Frequency-domain Wishart sufficient statistics. Use this when you need
   explicit control over FFT construction before calling the pipeline.

Time-Domain Container
---------------------

.. code-block:: python

   import numpy as np
   from log_psplines.datatypes import MultivariateTimeseries

   fs = 64.0
   t = np.arange(512) / fs
   y = np.column_stack([
       np.sin(2.0 * np.pi * 4.0 * t),
       np.cos(2.0 * np.pi * 8.0 * t),
   ])

   ts = MultivariateTimeseries(y=y, t=t)

For PSD estimation, standardising at the boundary is often helpful:

.. code-block:: python

   ts_std = ts.standardise_for_psd()

The original channel standard deviations are carried through so exported PSDs
can be rescaled back to physical units.

Wishart Statistics
------------------

``MultivariateTimeseries.to_wishart_stats`` and
``MultivarFFT.compute_wishart`` split the data into ``Nb`` contiguous blocks,
apply optional detrending and tapering, compute one-sided FFTs, drop DC, and
store a factor ``U`` such that

.. math::

   Y(f_k) = U(f_k) U(f_k)^H.

The pipeline uses ``Y(f_k)`` as the multivariate Whittle/Wishart sufficient
statistic.

.. code-block:: python

   fft = ts.standardise_for_psd().to_wishart_stats(
       Nb=4,
       fmin=1.0,
       fmax=30.0,
       window="hann",
       detrend="constant",
   )

Frequency Selection
-------------------

Frequency selection is applied in this order:

1. Convert time-domain data to the positive ``rfft`` grid.
2. Drop the DC bin.
3. Apply ``fmin`` and ``fmax`` if provided.
4. Remove any ``exclude_freq_bands``.
5. Optionally coarse grain the retained grid.

Coarse Graining
---------------

Coarse graining sums neighbouring Wishart matrices into equal-size consecutive
frequency bins. It is useful when the frequency grid is much denser than the
spectral structure being estimated.

.. code-block:: python

   from log_psplines.pipeline.config import PipelineConfig
   from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

   config = PipelineConfig(
       coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=128, Nh=None),
   )

See :doc:`coarse_grain` for the mathematical details and implementation
constraints.
