Multivariate Example
====================

The multivariate example is provided as a notebook because it includes plots
and a comparison between variational inference and blocked NUTS.

.. note::

   This is the next example after the :doc:`five-minute` univariate run. It is
   intentionally more expensive and is meant to show the spectral matrix and
   coherence workflow rather than the minimum API.

The notebook simulates a two-channel VAR process, runs the multivariate
pipeline, and plots posterior auto-spectra, cross-spectra, and coherence:

:download:`Open the multivariate example notebook <quickstart.ipynb>`

The notebook uses the same public pattern as the short example:

.. code-block:: python

   from log_psplines import PipelineConfig, fit

   result = fit(two_channel_data, PipelineConfig(...))

For the reconstructed matrix, use ``result.spectral_density``. For coherence,
use ``result.coherence``. Both preserve the channel dimensions and posterior
sample axes.

Multivariate time-varying inference is not currently implemented. For scalar
time-varying inference, continue to :doc:`time-varying-psd`.
