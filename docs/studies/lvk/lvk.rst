LVK
===

Demo on estimating LVK detector noise PSD using spline models.


Plots
-----

**Raw PSD Estimate**

.. image:: lvk_psd.png
   :alt: Raw PSD of LVK detector
   :width: 70%

**Fitted Noise Model**

.. image:: lvk_noise.png
   :alt: Fitted noise model
   :width: 70%

**Fitted Noise Model (PDF)**

.. image:: lvk_noise.pdf
   :alt: PDF version of the fitted noise model
   :width: 70%

**Noise + Spline Fit Overlay**

.. image:: lvk_noise_and_splines.png
   :alt: Overlay of noise model and fitted spline
   :width: 70%

**Trace Plot**

.. image:: traceplot.png
   :alt: Trace plot of the parameter estimation
   :width: 70%


Source Code
-----------

.. literalinclude:: run_estimator.py
   :language: python
   :caption: `run_estimator.py` - main script for LVK PSD estimation

.. literalinclude:: data.py
   :language: python
   :caption: `data.py` - helper functions and data preprocessing
