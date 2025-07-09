LogPSplinePSD
=============

Log-spline representation of the power spectral density (PSD) in the frequency domain, using penalized B-splines with a discrete penalty on spline coefficients.

**GitHub Repository**: https://github.com/nz-gravity/LogPSplinePSD


Overview
--------

`LogPSplinePSD` implements a Bayesian model for PSD estimation by fitting a log-spline to the periodogram. Main features:

- **Log-frequency representation**: Works on the log-scale of frequencies for numerical stability and improved resolution.
- **P-spline prior**: Applies a discrete difference penalty to log B-spline coefficients, enforcing smoothness in the log-PSD domain.
- **Whittle likelihood**: Employs Whittle's approximation for fast likelihood evaluation on periodogram ordinates.
- **HMC sampling**: Uses NumPyro (JAX) to perform efficient Hamiltonian Monte Carlo inference.

Methodology
-----------

The approach follows the P-spline framework for spectral density estimation described by Maturana-Russel & Meyer (2021) (arXiv:1905.01832):

1. **Basis construction**
   Define order-r B-spline basis functions \\(B_k(\\omega)\\), \\(k=1,\\dots,K+r\\), on an equidistant grid of interior knots in the log-frequency domain.

2. **Penalized prior**
   Apply a discrete \\(D\\)th-order difference penalty to the spline coefficients \\(\\{\\beta_k\\}\\), which induces smoothness in the estimated log-PSD.

3. **Knot placement (optional)**
   For spectra with sharp features, knot locations can be set based on quantiles of the raw periodogram values to allocate flexibility where needed.

4. **Model and likelihood**
The log-PSD is modeled as:

  .. math::

     \log f(\lambda_l) = \sum_k \beta_k \, B_k(\log \lambda_l)


Whittle’s approximation for the periodogram \\(I_n(\\lambda_l)\\) yields the log-likelihood:

.. math::

     \log L(\beta) \propto -\sum_{l=1}^{\nu} \left[ \log f(\lambda_l) + \frac{I_n(\lambda_l)}{f(\lambda_l)} \right]



5. **Inference**
   Jointly sample the spline coefficients and weights using NumPyro’s NUTS sampler.

This fixed-basis P-spline approach avoids reversible-jump MCMC over knot numbers and positions, reducing computational cost while retaining flexibility to capture complex spectral features.

Finally, one can also provide a 'parametric model' of the PSD as a function that can then be 'corrected' non-parametrically by the spline model.
This is useful for cases where a known functional form (e.g., power-law) is expected, but additional flexibility is needed to account for deviations in the data.

Installation
------------
::

    pip install LogPSplinePSD


Basic Usage
-----------

See `demo.py <https://github.com/nz-gravity/LogPSplinePSD/tree/main/docs>`_


.. literalinclude:: demo.py
   :language: python
   :linenos:

.. image:: https://github.com/nz-gravity/LogPSplinePSD/raw/main/docs/demo.png
   :alt: Demo Image
   :align: center



Author
------

NZ Gravity

Acknowledgements
----------------

Part of the NZ-Gravity and International LISA Consortium efforts on gravitational-wave data analysis.
