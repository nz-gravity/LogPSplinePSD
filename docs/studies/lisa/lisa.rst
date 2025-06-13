LISA
====

Demo on estimating LISA X channel noise PSD from LDC.

This is the raw periodogram of the LISA data. We take a running median approximation to set our knots:

.. image:: lisa_pdgrm.png
   :alt: Raw periodogram of LISA X-channel data
   :width: 80%

Source Code
-----------

.. literalinclude:: lisa_demo.py
   :language: python
   :caption: `lisa_demo.py` - main script for PSD estimation

Plots
-----

**Trace Plot**

.. image:: traceplot.png
   :alt: Trace plot of MCMC samples
   :width: 70%

**Posterior Predictive Check**

.. image:: lisa_fitted_model.png
   :alt: Posterior predictive fit to LISA data
   :width: 70%

**Spline Fit Only**

.. image:: just_splines.png
   :alt: Spline-only fit of the PSD
   :width: 70%
